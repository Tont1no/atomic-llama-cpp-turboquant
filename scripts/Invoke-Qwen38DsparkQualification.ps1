[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $PlanPath,
    [Parameter(Mandatory = $true)][ValidateSet('P1', 'P4', 'P8')][string] $Tier,
    [Parameter(Mandatory = $true)][string] $ServerPath,
    [Parameter(Mandatory = $true)][string] $ModelPath,
    [Parameter(Mandatory = $true)][string] $DraftModelPath,
    [Parameter(Mandatory = $true)][string] $ProfilePath,
    [Parameter(Mandatory = $true)][string] $OutputDir,
    [Parameter(Mandatory = $true)][string] $RuntimeLabel,
    [string] $PythonPath = '',
    [int] $GpuIndex = 0,
    [int] $MinFreeRamGiB = 18,
    [int] $MinFreeVramMiB = 5000,
    [int] $MaxUsedVramMiB = 27000,
    [int] $PollMilliseconds = 250,
    [int] $MaxRuntimeSeconds = 5400,
    [int] $JobTimeoutSeconds = 4800,
    [int] $ArmTimeoutSeconds = 1200,
    [int] $ReadinessTimeoutSeconds = 300,
    [int] $HttpTimeoutSeconds = 900,
    [int] $StopTimeoutSeconds = 30
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$driver = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'qwen38_dspark_qualification.py')).Path
$guard = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'Invoke-ExclusiveGpuTask.ps1')).Path
if (-not $PythonPath) { $PythonPath = (Get-Command python.exe -ErrorAction Stop).Source }
$python = (Resolve-Path -LiteralPath $PythonPath).Path
$plan = (Resolve-Path -LiteralPath $PlanPath).Path
$server = (Resolve-Path -LiteralPath $ServerPath).Path
$model = (Resolve-Path -LiteralPath $ModelPath).Path
$draft = (Resolve-Path -LiteralPath $DraftModelPath).Path
$profile = (Resolve-Path -LiteralPath $ProfilePath).Path
$output = if ([IO.Path]::IsPathRooted($OutputDir)) {
    [IO.Path]::GetFullPath($OutputDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}

foreach ($item in @($server, $model, $draft, $profile)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) { throw "Required artifact is not a file: $item" }
}
if ($MaxRuntimeSeconds -le $JobTimeoutSeconds) {
    throw 'MaxRuntimeSeconds must be greater than JobTimeoutSeconds.'
}
if ($MinFreeRamGiB -lt 18 -or $MinFreeVramMiB -lt 5000 -or $MaxUsedVramMiB -gt 27000) {
    throw 'Qualification guard limits may only be made stricter than 18 GiB RAM / 5000 MiB free / 27000 MiB used.'
}

# This is deliberately before OutputDir creation and before taking the lease.
& $python $driver validate --plan $plan --tier $Tier
if ($LASTEXITCODE -ne 0) { throw "Qualification plan validation failed with exit code $LASTEXITCODE" }
if (Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue) {
    throw 'Preflight refused: a llama-server process already exists.'
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
$runnerArguments = @(
    $driver, 'run', '--plan', $plan, '--tier', $Tier,
    '--server', $server, '--model', $model, '--draft-model', $draft,
    '--profile', $profile, '--output-dir', $output, '--runtime-label', $RuntimeLabel,
    '--job-timeout-s', [string] $JobTimeoutSeconds,
    '--arm-timeout-s', [string] $ArmTimeoutSeconds,
    '--readiness-timeout-s', [string] $ReadinessTimeoutSeconds,
    '--http-timeout-s', [string] $HttpTimeoutSeconds,
    '--stop-timeout-s', [string] $StopTimeoutSeconds
)

# Exactly one guard invocation owns the whole target -> static(s) -> SPS run.
$guardReceipt = & $guard `
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

if (Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue) {
    throw 'Postflight failed: a llama-server process remains after guarded cleanup.'
}
if (-not (Test-Path -LiteralPath (Join-Path $output 'summary.json') -PathType Leaf)) {
    throw 'Guarded qualification returned without a final summary.'
}
$receiptPath = Join-Path $output 'guard-receipt.json'
$temporaryReceipt = "$receiptPath.$PID.tmp"
$receiptJson = @{
    Schema = 'ai-loader-qwen38-dspark-qualification-guard-receipt/v1'
    Tier = $Tier
    DynamicRs = $false
    GuardStdout = $guardStdout
    GuardStderr = $guardStderr
    Receipt = $guardReceipt
    CompletedAtUtc = [DateTime]::UtcNow.ToString('o')
} | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText($temporaryReceipt, $receiptJson, (New-Object Text.UTF8Encoding($false)))
[IO.File]::Move($temporaryReceipt, $receiptPath)
