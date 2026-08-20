[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $PlanPath,

    [Parameter(Mandatory = $true)]
    [string] $ServerPath,

    [Parameter(Mandatory = $true)]
    [string] $ModelPath,

    [Parameter(Mandatory = $true)]
    [string] $DraftModelPath,

    [Parameter(Mandatory = $true)]
    [string] $OutputDir,

    [Parameter(Mandatory = $true)]
    [string] $RuntimeLabel,

    [string] $PythonPath = '',

    [int] $GpuIndex = 0,

    [int] $MinFreeRamGiB = 16,

    [int] $MinFreeVramMiB = 4096,

    [int] $MaxUsedVramMiB = 28672,

    [int] $PollMilliseconds = 500,

    [int] $MaxRuntimeSeconds = 21600,

    [int] $JobTimeoutSeconds = 18000,

    [int] $ArmTimeoutSeconds = 1200,

    [int] $ReadinessTimeoutSeconds = 300,

    [int] $HttpTimeoutSeconds = 900,

    [int] $ProgressTimeoutSeconds = 300,

    [int] $StopTimeoutSeconds = 30,

    [int] $Port = 18136,

    [int] $ContextSize = 65536,

    [int] $Parallel = 8,

    [int] $BatchSize = 2048,

    [int] $UbatchSize = 512,

    [ValidateSet('q4_0', 'q8_0', 'f16')]
    [string] $TargetKv = 'q4_0',

    [ValidateSet('q4_0', 'q8_0', 'f16')]
    [string] $DraftKv = 'q4_0',

    [int] $PromptTokenId = 1,

    [int] $OutputTokens = 0,

    [int] $PromptSafetyTokens = 16,

    [switch] $DisableUnifiedKv,

    [switch] $DisableCudaGraphs,

    [switch] $NoResume
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$driverPath = Join-Path $PSScriptRoot 'dspark_sps_profile.py'
$guardPath = Join-Path $PSScriptRoot 'Invoke-ExclusiveGpuTask.ps1'

if (-not $PythonPath) {
    $PythonPath = (Get-Command python.exe -ErrorAction Stop).Source
}

$resolvedPython = (Resolve-Path -LiteralPath $PythonPath).Path
$resolvedPlan = (Resolve-Path -LiteralPath $PlanPath).Path
$resolvedServer = (Resolve-Path -LiteralPath $ServerPath).Path
$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
$resolvedDraft = (Resolve-Path -LiteralPath $DraftModelPath).Path
$resolvedDriver = (Resolve-Path -LiteralPath $driverPath).Path
$resolvedGuard = (Resolve-Path -LiteralPath $guardPath).Path
$resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}

if (-not (Test-Path -LiteralPath $resolvedServer -PathType Leaf)) {
    throw "ServerPath is not a file: $resolvedServer"
}
if (-not (Test-Path -LiteralPath $resolvedModel -PathType Leaf)) {
    throw "ModelPath is not a file: $resolvedModel"
}
if (-not (Test-Path -LiteralPath $resolvedDraft -PathType Leaf)) {
    throw "DraftModelPath is not a file: $resolvedDraft"
}
if ($MaxRuntimeSeconds -gt 0 -and $MaxRuntimeSeconds -le $JobTimeoutSeconds) {
    throw 'MaxRuntimeSeconds must be zero or greater than JobTimeoutSeconds.'
}

# Validate the immutable plan before acquiring the GPU lease.
& $resolvedPython $resolvedDriver validate --plan $resolvedPlan
if ($LASTEXITCODE -ne 0) {
    throw "SPS profile plan validation failed with exit code $LASTEXITCODE"
}
$plan = Get-Content -Raw -LiteralPath $resolvedPlan | ConvertFrom-Json

New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
$guardStdout = Join-Path $resolvedOutput 'guard.stdout.log'
$guardStderr = Join-Path $resolvedOutput 'guard.stderr.log'
$runnerArguments = @(
    $resolvedDriver,
    'run',
    '--plan', $resolvedPlan,
    '--server', $resolvedServer,
    '--model', $resolvedModel,
    '--draft-model', $resolvedDraft,
    '--output-dir', $resolvedOutput,
    '--runtime-label', $RuntimeLabel,
    '--port', [string] $Port,
    '--ctx-size', [string] $ContextSize,
    '--parallel', [string] $Parallel,
    '--batch-size', [string] $BatchSize,
    '--ubatch-size', [string] $UbatchSize,
    '--target-kv', $TargetKv,
    '--draft-kv', $DraftKv,
    '--max-draft-tokens-per-slot', [string] $plan.max_draft_tokens_per_slot,
    '--samples-per-cell', [string] $plan.samples_per_cell,
    '--warmup-shape-runs', [string] $plan.warmup_shape_runs,
    '--prompt-token-id', [string] $PromptTokenId,
    '--output-tokens', [string] $OutputTokens,
    '--prompt-safety-tokens', [string] $PromptSafetyTokens,
    '--job-timeout-s', [string] $JobTimeoutSeconds,
    '--arm-timeout-s', [string] $ArmTimeoutSeconds,
    '--readiness-timeout-s', [string] $ReadinessTimeoutSeconds,
    '--http-timeout-s', [string] $HttpTimeoutSeconds,
    '--progress-timeout-s', [string] $ProgressTimeoutSeconds,
    '--stop-timeout-s', [string] $StopTimeoutSeconds
)
if ($DisableUnifiedKv) {
    $runnerArguments += '--disable-unified-kv'
}
if ($DisableCudaGraphs) {
    $runnerArguments += '--disable-cuda-graphs'
}
if ($NoResume) {
    $runnerArguments += '--no-resume'
}

# The whole matrix is one guarded root task. The runner keeps every server arm
# sequential inside this child process.
& $resolvedGuard `
    -Executable $resolvedPython `
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
