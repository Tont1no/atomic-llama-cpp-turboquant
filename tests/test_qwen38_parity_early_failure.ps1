[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $DiagnosticExecutable
)

$ErrorActionPreference = 'Stop'

function Assert-True([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw "ASSERTION FAILED: $Message" }
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("qwen38-parity-error-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot | Out-Null
$stdoutPath = Join-Path $tempRoot 'stdout.jsonl'
$stderrPath = Join-Path $tempRoot 'stderr.log'
$hash = ('0' * 64)
$environment = [ordered]@{
    LLAMACPP_QWEN38_PARITY_GUARD = 'ONE_CONTEXT_NO_SERVER'
    LLAMACPP_QWEN38_PARITY_REFERENCE_TOKENS = ((0..255) -join ',')
    LLAMACPP_QWEN38_PARITY_EXECUTABLE_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_MODEL_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_REFERENCE_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_MANIFEST_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_COMMAND_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_RECEIPT_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_RUNTIME_BUNDLE_MATCH = 'true'
    LLAMACPP_QWEN38_PARITY_BUNDLE_COUNT = '1'
    LLAMACPP_QWEN38_PARITY_CURRENT_BUNDLE_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_REFERENCE_BUNDLE_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_CURRENT_SERVER_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_REFERENCE_SERVER_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_CURRENT_LLAMA_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_REFERENCE_LLAMA_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_CURRENT_COMMON_SHA256 = $hash
    LLAMACPP_QWEN38_PARITY_REFERENCE_COMMON_SHA256 = $hash
    AI_LOADER_EXCLUSIVE_GPU_GUARD = '1'
    AI_LOADER_EXCLUSIVE_GPU_UUID = 'CPU-TEST'
    AI_LOADER_EXCLUSIVE_GPU_LEASE_PATH = 'fake-lease'
    AI_LOADER_EXCLUSIVE_GPU_LEASE_NONCE = 'fake-nonce'
    AI_LOADER_EXCLUSIVE_GPU_OWNER_PID = '1'
    AI_LOADER_EXCLUSIVE_GPU_JOB_NAME = 'fake-job'
    CUDA_VISIBLE_DEVICES = 'CPU-TEST'
}
$prior = @{}

try {
    foreach ($entry in $environment.GetEnumerator()) {
        $prior[$entry.Key] = [System.Environment]::GetEnvironmentVariable($entry.Key, 'Process')
        [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    $arguments = @(
        '--model', (Join-Path $tempRoot 'must-not-load.gguf'),
        '--ctx-size', '2048', '--parallel', '1',
        '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
        '--flash-attn', 'on', '--batch-size', '2048', '--ubatch-size', '128',
        '--n-gpu-layers', 'all', '--device', 'none', '--split-mode', 'none', '--kv-unified'
        '--fit', 'off', '--cache-ram', '0', '--ctx-checkpoints', '0',
        '--no-cache-idle-slots', '--no-cache-prompt', '--no-webui', '--metrics',
        '--reasoning', 'off', '-lv', '4'
    )
    & $DiagnosticExecutable @arguments 1> $stdoutPath 2> $stderrPath
    $exitCode = $LASTEXITCODE

    Assert-True ($exitCode -eq 2) 'early argument-contract failure must return exit code 2'
    Assert-True ((Get-Item -LiteralPath $stdoutPath).Length -eq 0) 'early failure must not emit success JSONL'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $tempRoot 'must-not-load.gguf'))) `
        'regression must not create or load a model fixture'

    $records = @()
    foreach ($line in Get-Content -LiteralPath $stderrPath) {
        try {
            $record = $line | ConvertFrom-Json -ErrorAction Stop
            if ($record.type -ceq 'error') { $records += $record }
        } catch {
            # Parser/backend warnings may precede the structured terminal record.
        }
    }
    Assert-True ($records.Count -eq 1) 'stderr must contain exactly one structured error record'
    Assert-True ($records[0].stage -ceq 'argument_contract') 'error stage must identify argument validation'
    Assert-True ([string] $records[0].message -like '*exact P1 target settings*') `
        'error message must state the actionable P1 contract'
} finally {
    foreach ($entry in $environment.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($entry.Key, $prior[$entry.Key], 'Process')
    }
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
