$ErrorActionPreference = 'Stop'

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) { throw $Message }
}

function Assert-Argument {
    param([string[]] $Arguments, [string] $Name, [string] $Expected)
    $index = [array]::IndexOf($Arguments, $Name)
    Assert-True ($index -ge 0) "missing runner argument: $Name"
    Assert-True ($index + 1 -lt $Arguments.Count) "runner argument has no value: $Name"
    Assert-True ($Arguments[$index + 1] -eq $Expected) "wrong value for ${Name}: $($Arguments[$index + 1])"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ("dspark-wrapper-" + [guid]::NewGuid().ToString('N'))
$scripts = Join-Path $testRoot 'scripts'
$capturePath = Join-Path $testRoot 'capture.json'
$oldCapture = $env:DSPARK_WRAPPER_CAPTURE

try {
    New-Item -ItemType Directory -Path $scripts -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\Invoke-DsparkSpsProfile.ps1') -Destination $scripts
    Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\dspark_sps_profile.py') -Destination $scripts
    $fakeGuard = @'
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Executable,
    [string[]] $TaskArguments = @(),
    [string] $TaskWorkingDirectory = '',
    [int] $MinFreeRamGiB = 0,
    [int] $MinFreeVramMiB = 0,
    [int] $MaxUsedVramMiB = 0,
    [int] $GpuIndex = 0,
    [int] $PollMilliseconds = 0,
    [int] $MaxRuntimeSeconds = 0,
    [string] $StdoutPath = '',
    [string] $StderrPath = ''
)
@{
    Executable = $Executable
    TaskArguments = @($TaskArguments)
    TaskWorkingDirectory = $TaskWorkingDirectory
    MinFreeRamGiB = $MinFreeRamGiB
    MinFreeVramMiB = $MinFreeVramMiB
    MaxUsedVramMiB = $MaxUsedVramMiB
    GpuIndex = $GpuIndex
    PollMilliseconds = $PollMilliseconds
    MaxRuntimeSeconds = $MaxRuntimeSeconds
    StdoutPath = $StdoutPath
    StderrPath = $StderrPath
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $env:DSPARK_WRAPPER_CAPTURE -Encoding utf8
'@
    Set-Content -LiteralPath (Join-Path $scripts 'Invoke-ExclusiveGpuTask.ps1') -Value $fakeGuard -Encoding utf8

    $python = (Get-Command python.exe -ErrorAction Stop).Source
    $driver = Join-Path $scripts 'dspark_sps_profile.py'
    $plan = Join-Path $testRoot 'plan.json'
    & $python $driver plan --output $plan --active-slots 1 --context-ceilings 2048 `
        --prefix-caps 0,7 --repetitions 2 --samples-per-cell 2 --warmup-shape-runs 3 | Out-Null
    Assert-True ($LASTEXITCODE -eq 0) 'CPU plan generation failed'

    $server = Join-Path $testRoot 'server.exe'
    $model = Join-Path $testRoot 'target.gguf'
    $draft = Join-Path $testRoot 'draft.gguf'
    [IO.File]::WriteAllText($server, 'mock-server')
    [IO.File]::WriteAllText($model, 'mock-target')
    [IO.File]::WriteAllText($draft, 'mock-draft')
    $env:DSPARK_WRAPPER_CAPTURE = $capturePath

    & (Join-Path $scripts 'Invoke-DsparkSpsProfile.ps1') `
        -PlanPath $plan -ServerPath $server -ModelPath $model -DraftModelPath $draft `
        -OutputDir (Join-Path $testRoot 'output') -RuntimeLabel 'cuda-mock' -PythonPath $python `
        -GpuIndex 3 -MinFreeRamGiB 19 -MinFreeVramMiB 5000 -MaxUsedVramMiB 27000 `
        -PollMilliseconds 321 -MaxRuntimeSeconds 999 -JobTimeoutSeconds 900 `
        -ArmTimeoutSeconds 101 -ReadinessTimeoutSeconds 31 -HttpTimeoutSeconds 41 `
        -ProgressTimeoutSeconds 51 -StopTimeoutSeconds 11 -Port 19136 -ContextSize 32768 `
        -Parallel 4 -BatchSize 1024 -UbatchSize 256 -TargetKv q8_0 -DraftKv f16 `
        -PromptTokenId 17 -OutputTokens 512 -PromptSafetyTokens 23 `
        -DisableUnifiedKv -DisableCudaGraphs -NoResume

    $capture = Get-Content -Raw -LiteralPath $capturePath | ConvertFrom-Json
    Assert-True ($capture.MaxRuntimeSeconds -eq 999) 'MaxRuntimeSeconds was not forwarded to the guard'
    Assert-True ($capture.GpuIndex -eq 3) 'GpuIndex was not forwarded to the guard'
    Assert-True ($capture.MinFreeRamGiB -eq 19) 'RAM guard was not forwarded'
    Assert-True ($capture.MinFreeVramMiB -eq 5000) 'free VRAM guard was not forwarded'
    Assert-True ($capture.MaxUsedVramMiB -eq 27000) 'used VRAM guard was not forwarded'
    Assert-True ($capture.PollMilliseconds -eq 321) 'guard polling interval was not forwarded'
    $runner = @($capture.TaskArguments)
    Assert-Argument $runner '--runtime-label' 'cuda-mock'
    Assert-Argument $runner '--job-timeout-s' '900'
    Assert-Argument $runner '--arm-timeout-s' '101'
    Assert-Argument $runner '--readiness-timeout-s' '31'
    Assert-Argument $runner '--http-timeout-s' '41'
    Assert-Argument $runner '--progress-timeout-s' '51'
    Assert-Argument $runner '--stop-timeout-s' '11'
    Assert-Argument $runner '--port' '19136'
    Assert-Argument $runner '--ctx-size' '32768'
    Assert-True (-not ($runner -contains '--recurrent-cache-type')) 'unsupported recurrent-cache option was forwarded'
    Assert-True ($runner -contains '--disable-unified-kv') 'unified-KV switch was not forwarded'
    Assert-True ($runner -contains '--disable-cuda-graphs') 'CUDA-graph switch was not forwarded'
    Assert-True ($runner -contains '--no-resume') 'resume switch was not forwarded'
    'DSpark SPS wrapper CPU-only tests: PASS'
} finally {
    $env:DSPARK_WRAPPER_CAPTURE = $oldCapture
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
