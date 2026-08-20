[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $PlanPath,
    [Parameter(Mandatory = $true)][string] $ServerPath,
    [Parameter(Mandatory = $true)][string] $ModelPath,
    [Parameter(Mandatory = $true)][string] $OutputDir,
    [Parameter(Mandatory = $true)][string] $RuntimeLabel,
    [string] $PythonPath = '',
    [int] $GpuIndex = 0,
    [int] $MinFreeRamGiB = 18,
    [int] $MinFreeVramMiB = 5000,
    [int] $MaxUsedVramMiB = 27000,
    [int] $PollMilliseconds = 250,
    [int] $MaxRuntimeSeconds = 2400,
    [int] $JobTimeoutSeconds = 1800,
    [int] $ArmTimeoutSeconds = 1200,
    [int] $ReadinessTimeoutSeconds = 300,
    [int] $HttpTimeoutSeconds = 900,
    [int] $StopTimeoutSeconds = 30
)

$ErrorActionPreference = 'Stop'

function Get-Sha256([string] $Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-LoopbackPortFree([int] $Port) {
    $listener = $null
    try {
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) { $listener.Stop() }
    }
}

function Write-AtomicJson([string] $Path, [object] $Value) {
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $temporary = "$Path.$PID.$([guid]::NewGuid().ToString('N')).tmp"
    $json = $Value | ConvertTo-Json -Depth 16
    [IO.File]::WriteAllText($temporary, $json, (New-Object Text.UTF8Encoding($false)))
    [IO.File]::Move($temporary, $Path)
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$driver = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'qwen38_dspark_qualification.py')).Path
$guard = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'Invoke-ExclusiveGpuTask.ps1')).Path
$wrapper = $MyInvocation.MyCommand.Path
if (-not $PythonPath) { $PythonPath = (Get-Command python.exe -ErrorAction Stop).Source }
$python = (Resolve-Path -LiteralPath $PythonPath).Path
$plan = (Resolve-Path -LiteralPath $PlanPath).Path
$server = (Resolve-Path -LiteralPath $ServerPath).Path
$model = (Resolve-Path -LiteralPath $ModelPath).Path
$output = if ([IO.Path]::IsPathRooted($OutputDir)) {
    [IO.Path]::GetFullPath($OutputDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}

foreach ($item in @($server, $model)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) { throw "Required artifact is not a file: $item" }
}
if ((Split-Path -Leaf $output) -cne 'p1-final-v5') {
    throw 'OutputDir must end in p1-final-v5 for the parity diagnostic layout.'
}
if ($MaxRuntimeSeconds -le $JobTimeoutSeconds) {
    throw 'MaxRuntimeSeconds must be greater than JobTimeoutSeconds.'
}
if ($MinFreeRamGiB -lt 18 -or $MinFreeVramMiB -lt 5000 -or $MaxUsedVramMiB -gt 27000) {
    throw 'Target-trace guard limits may only be made stricter than 18 GiB RAM / 5000 MiB free / 27000 MiB used.'
}

$parent = Split-Path -Parent $output
$leaf = Split-Path -Leaf $output
if (-not $parent -or -not $leaf) { throw 'OutputDir must name a directory below a parent directory.' }
if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
}
$suffix = '{0}-{1}-{2}' -f ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')), $PID, ([guid]::NewGuid().ToString('N'))
$guardStdout = Join-Path $parent "$leaf.guard-$suffix.stdout.log"
$guardStderr = Join-Path $parent "$leaf.guard-$suffix.stderr.log"
$externalReceiptPath = Join-Path $parent "$leaf.guard-$suffix.receipt.json"
$internalReceiptPath = Join-Path $output 'guard-receipt.json'
$relevantProcessNames = @(
    'llama-server', 'llama-cli', 'test-backend-ops', 'test-dflash-fusion-determinism',
    'test-fp8-e4m3', 'test-qwen38-recurrent-parity', 'test-recurrent-state-rollback'
)
$mutex = [Threading.Mutex]::new($false, 'Global\AiLoaderQwen38TargetTrace')
$mutexHeld = $false
$receiptWritten = $false
$guardStarted = $false
$stage = 'mutex'
try {
    try {
        $mutexHeld = $mutex.WaitOne(0)
    } catch [Threading.AbandonedMutexException] {
        $mutexHeld = $true
    }
    if (-not $mutexHeld) { throw 'Another Qwen3.8 target trace is already running.' }

    # CPU-only validation happens before the exclusive GPU lease.
    $stage = 'preflight'
    & $python $driver validate --plan $plan --tier P1
    if ($LASTEXITCODE -ne 0) { throw "Qualification plan validation failed with exit code $LASTEXITCODE" }
    $preflightProcesses = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in $relevantProcessNames })
    if ($preflightProcesses.Count -ne 0) { throw 'Preflight refused: a relevant llama/GPU test process already exists.' }
    if (-not (Test-LoopbackPortFree 18136)) { throw 'Preflight refused: port 18136 is occupied.' }
    if ((Test-Path -LiteralPath $output) -and @(Get-ChildItem -LiteralPath $output -Force).Count -ne 0) {
        throw 'OutputDir must be absent or empty; this runner never resumes or overwrites a trace.'
    }

    $runnerArguments = @(
        $driver, 'target-trace', '--plan', $plan, '--server', $server, '--model', $model,
        '--output-dir', $output, '--runtime-label', $RuntimeLabel,
        '--job-timeout-s', [string] $JobTimeoutSeconds,
        '--arm-timeout-s', [string] $ArmTimeoutSeconds,
        '--readiness-timeout-s', [string] $ReadinessTimeoutSeconds,
        '--http-timeout-s', [string] $HttpTimeoutSeconds,
        '--stop-timeout-s', [string] $StopTimeoutSeconds
    )
    if ($runnerArguments -contains '--draft-model' -or $runnerArguments -contains '--profile' -or
            $runnerArguments -contains 'run') {
        throw 'Internal target-trace command unexpectedly selected a full qualification input or command.'
    }

    $runtimeEnvironmentPrior = @{}
    $guardResult = $null
    $guardFailure = $null
    try {
        foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
            $name = [string] $entry.Key
            if ($name.StartsWith('LLAMA_', [StringComparison]::OrdinalIgnoreCase) -or
                    $name.StartsWith('GGML_', [StringComparison]::OrdinalIgnoreCase) -or
                    $name.StartsWith('CUDA_', [StringComparison]::OrdinalIgnoreCase)) {
                $runtimeEnvironmentPrior[$name] = [string] $entry.Value
                Remove-Item -LiteralPath ("Env:" + $name) -ErrorAction SilentlyContinue
            }
        }
        try {
            # Exactly one guard owns exactly one target-only server lifecycle.
            $stage = 'guard'
            $guardStarted = $true
            $guardResult = & $guard `
                -Executable $python `
                -TaskArguments $runnerArguments `
                -TaskWorkingDirectory $repoRoot `
                -GpuIndex $GpuIndex `
                -MinFreeRamGiB $MinFreeRamGiB `
                -MinFreeVramMiB $MinFreeVramMiB `
                -MaxUsedVramMiB $MaxUsedVramMiB `
                -PollMilliseconds $PollMilliseconds `
                -MaxRuntimeSeconds $MaxRuntimeSeconds `
                -StdoutPath $guardStdout `
                -StderrPath $guardStderr
        } catch {
            $guardFailure = $_.Exception
        }
    } finally {
        foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
            $name = [string] $entry.Key
            if ($name.StartsWith('LLAMA_', [StringComparison]::OrdinalIgnoreCase) -or
                    $name.StartsWith('GGML_', [StringComparison]::OrdinalIgnoreCase) -or
                    $name.StartsWith('CUDA_', [StringComparison]::OrdinalIgnoreCase)) {
                Remove-Item -LiteralPath ("Env:" + $name) -ErrorAction SilentlyContinue
            }
        }
        foreach ($entry in $runtimeEnvironmentPrior.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
        }
    }

    $stage = 'postflight'
    $remainingProcesses = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in $relevantProcessNames })
    $remainingLeases = @(Get-ChildItem -LiteralPath $repoRoot -Filter 'gpu-exclusive.lock' -File -Recurse -ErrorAction SilentlyContinue)
    $postflight = [ordered]@{
        relevant_processes = $remainingProcesses.Count
        port_18136_free = Test-LoopbackPortFree 18136
        recursive_gpu_leases = $remainingLeases.Count
    }
    $guardCompleted = -not $guardFailure -and $guardResult -and
        $guardResult.Status -ceq 'completed' -and [int] $guardResult.ExitCode -eq 0
    if (-not $guardCompleted -and -not $guardFailure) {
        $guardFailure = [InvalidOperationException]::new('Exclusive GPU guard returned an invalid receipt.')
    }
    $validationStatus = 'not_run'
    $validationFailure = $null
    if ($guardCompleted -and $postflight.relevant_processes -eq 0 -and $postflight.port_18136_free -and
            $postflight.recursive_gpu_leases -eq 0) {
        try {
            $stage = 'evidence-validation'
            $validationLines = @(& $python $driver validate-target-trace `
                --plan $plan --server $server --model $model --output-dir $output `
                --gpu-uuid ([string] $guardResult.GpuUuid))
            if ($LASTEXITCODE -ne 0 -or $validationLines.Count -eq 0) {
                throw "Target trace evidence validation failed with exit code $LASTEXITCODE"
            }
            $validated = $validationLines[-1] | ConvertFrom-Json
            $validation = Get-Content -LiteralPath (Join-Path $output 'validation.json') -Raw | ConvertFrom-Json
            if ($validated.schema -cne 'ai-loader-qwen38-target-trace-validation/v1' -or
                    -not [bool] $validated.valid -or [string] $validated.gpu_uuid -cne [string] $guardResult.GpuUuid -or
                    [string] $validated.identity -cne [string] $validation.identity) {
                throw 'Target trace evidence validator returned an invalid identity result.'
            }
            $validationStatus = 'passed'
        } catch {
            $validationStatus = 'failed'
            $validationFailure = $_.Exception
        }
    }

    $postflightFailure = if ($postflight.relevant_processes -ne 0 -or -not $postflight.port_18136_free -or
            $postflight.recursive_gpu_leases -ne 0) { [InvalidOperationException]::new('Postflight proof failed after target trace.') } else { $null }
    $overallFailure = if ($guardFailure) { $guardFailure } elseif ($postflightFailure) { $postflightFailure } else { $validationFailure }
    $guardStatus = if ($guardCompleted) { 'completed' } elseif ($guardStarted) { 'failed' } else { 'not_started' }
    $receipt = [ordered]@{
        schema = 'ai-loader-qwen38-target-trace-guard-receipt/v1'
        created_at = [DateTimeOffset]::UtcNow.ToString('o')
        stage = if ($overallFailure) { $stage } else { 'completed' }
        guard_status = $guardStatus
        trace_validation = $validationStatus
        guard_started = $guardStarted
        error = if ($overallFailure) { $overallFailure.Message } else { $null }
        runner_sha256 = Get-Sha256 $driver
        wrapper_sha256 = Get-Sha256 $wrapper
        guard_sha256 = Get-Sha256 $guard
        server_sha256 = Get-Sha256 $server
        model_sha256 = Get-Sha256 $model
        manifest_sha256 = if (Test-Path -LiteralPath (Join-Path $output 'manifest.json') -PathType Leaf) { Get-Sha256 (Join-Path $output 'manifest.json') } else { $null }
        validation_sha256 = if (Test-Path -LiteralPath (Join-Path $output 'validation.json') -PathType Leaf) { Get-Sha256 (Join-Path $output 'validation.json') } else { $null }
        guard_stdout_sha256 = if (Test-Path -LiteralPath $guardStdout -PathType Leaf) { Get-Sha256 $guardStdout } else { $null }
        guard_stderr_sha256 = if (Test-Path -LiteralPath $guardStderr -PathType Leaf) { Get-Sha256 $guardStderr } else { $null }
        guard = $guardResult
        postflight = $postflight
    }
    Write-AtomicJson $externalReceiptPath $receipt
    $receiptWritten = $true
    if (-not $overallFailure) { Write-AtomicJson $internalReceiptPath $receipt }
    if ($overallFailure) {
        throw [InvalidOperationException]::new("$($overallFailure.Message) Guard/postflight receipt: $externalReceiptPath", $overallFailure)
    }

    Write-Host "Fresh P1 target-only trace: $output"
    Write-Host "Guard/postflight receipt: $internalReceiptPath"
    Write-Host 'Review validation.json, then update the immutable parity-diagnostic anchors before running it.'
    $stage = 'completed'
} catch {
    $caught = $_.Exception
    if ($mutexHeld -and -not $receiptWritten) {
        $failureProcesses = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in $relevantProcessNames })
        $failureLeases = @(Get-ChildItem -LiteralPath $repoRoot -Filter 'gpu-exclusive.lock' -File -Recurse -ErrorAction SilentlyContinue)
        $failurePostflight = [ordered]@{
            relevant_processes = $failureProcesses.Count
            port_18136_free = Test-LoopbackPortFree 18136
            recursive_gpu_leases = $failureLeases.Count
        }
        $failureReceipt = [ordered]@{
            schema = 'ai-loader-qwen38-target-trace-guard-receipt/v1'
            created_at = [DateTimeOffset]::UtcNow.ToString('o')
            stage = $stage
            guard_started = $guardStarted
            guard_status = if ($guardStarted) { 'failed' } else { 'not_started' }
            trace_validation = 'not_run'
            error = $caught.Message
            runner_sha256 = Get-Sha256 $driver
            wrapper_sha256 = Get-Sha256 $wrapper
            guard_sha256 = Get-Sha256 $guard
            server_sha256 = Get-Sha256 $server
            model_sha256 = Get-Sha256 $model
            manifest_sha256 = if (Test-Path -LiteralPath (Join-Path $output 'manifest.json') -PathType Leaf) { Get-Sha256 (Join-Path $output 'manifest.json') } else { $null }
            validation_sha256 = if (Test-Path -LiteralPath (Join-Path $output 'validation.json') -PathType Leaf) { Get-Sha256 (Join-Path $output 'validation.json') } else { $null }
            guard_stdout_sha256 = if (Test-Path -LiteralPath $guardStdout -PathType Leaf) { Get-Sha256 $guardStdout } else { $null }
            guard_stderr_sha256 = if (Test-Path -LiteralPath $guardStderr -PathType Leaf) { Get-Sha256 $guardStderr } else { $null }
            guard = $null
            postflight = $failurePostflight
        }
        Write-AtomicJson $externalReceiptPath $failureReceipt
        $receiptWritten = $true
        throw [InvalidOperationException]::new("$($caught.Message) Guard/postflight receipt: $externalReceiptPath", $caught)
    }
    throw
} finally {
    if ($mutexHeld) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
