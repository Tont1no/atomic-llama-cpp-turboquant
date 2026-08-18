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

    [string] $PythonPath = '',

    [int] $GpuIndex = 0,

    [int] $MinFreeRamGiB = 16,

    [int] $MinFreeVramMiB = 4096,

    [int] $MaxUsedVramMiB = 28672,

    [int] $PollMilliseconds = 500
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
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDir, $repoRoot)

if (-not (Test-Path -LiteralPath $resolvedServer -PathType Leaf)) {
    throw "ServerPath is not a file: $resolvedServer"
}
if (-not (Test-Path -LiteralPath $resolvedModel -PathType Leaf)) {
    throw "ModelPath is not a file: $resolvedModel"
}
if (-not (Test-Path -LiteralPath $resolvedDraft -PathType Leaf)) {
    throw "DraftModelPath is not a file: $resolvedDraft"
}

# Validate the immutable plan before acquiring the GPU lease.
& $resolvedPython $resolvedDriver validate --plan $resolvedPlan
if ($LASTEXITCODE -ne 0) {
    throw "SPS profile plan validation failed with exit code $LASTEXITCODE"
}

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
    '--output-dir', $resolvedOutput
)

# The whole matrix is one guarded root task. The future runner must keep all
# server arms sequential inside this child process.
& $resolvedGuard `
    -Executable $resolvedPython `
    -TaskArguments $runnerArguments `
    -TaskWorkingDirectory $repoRoot `
    -GpuIndex $GpuIndex `
    -MinFreeRamGiB $MinFreeRamGiB `
    -MinFreeVramMiB $MinFreeVramMiB `
    -MaxUsedVramMiB $MaxUsedVramMiB `
    -PollMilliseconds $PollMilliseconds `
    -StdoutPath $guardStdout `
    -StderrPath $guardStderr
