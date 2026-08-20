$ErrorActionPreference = 'Stop'

function Assert-True([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw $Message }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('qwen38-target-trace-wrapper-' + [guid]::NewGuid().ToString('N'))
$scripts = Join-Path $testRoot 'scripts'
$capture = Join-Path $testRoot 'capture.json'
$oldCapture = $env:QWEN38_TARGET_TRACE_CAPTURE
$oldLlama = $env:LLAMA_ARG_CTX_SIZE
$oldGgml = $env:GGML_CUDA_DISABLE_GRAPHS
$oldCuda = $env:CUDA_LAUNCH_BLOCKING
$oldProjection = $env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS
try {
    New-Item -ItemType Directory -Path $scripts -Force | Out-Null
    foreach ($name in @('Invoke-Qwen38TargetTrace.ps1', 'qwen38_dspark_qualification.py', 'dspark_sps_profile.py')) {
        Copy-Item -LiteralPath (Join-Path $repoRoot "scripts\$name") -Destination $scripts
    }
    @'
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Executable,
    [string[]] $TaskArguments = @(), [string] $TaskWorkingDirectory = '',
    [int] $MinFreeRamGiB = 0, [int] $MinFreeVramMiB = 0, [int] $MaxUsedVramMiB = 0,
    [int] $GpuIndex = 0, [int] $PollMilliseconds = 0, [int] $MaxRuntimeSeconds = 0,
    [string] $StdoutPath = '', [string] $StderrPath = ''
)
function Sha([string] $Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function TextSha([string] $Text) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes($Text)
        ([BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally { $algorithm.Dispose() }
}
function WriteJson([string] $Path, [object] $Value) {
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 16), (New-Object Text.UTF8Encoding($false)))
}
function Arg([string] $Name) {
    $index = [array]::IndexOf($TaskArguments, $Name)
    if ($index -lt 0) { throw "missing $Name" }
    $TaskArguments[$index + 1]
}
$output = Arg '--output-dir'
$server = Arg '--server'
$model = Arg '--model'
$runner = $TaskArguments[0]
$wrapper = Join-Path $PSScriptRoot 'Invoke-Qwen38TargetTrace.ps1'
if ($env:QWEN38_TARGET_TRACE_FAKE_FAIL -eq '1') { throw 'fake target trace guard failure' }
$root = Join-Path $output 'arms\target-only'
New-Item -ItemType Directory -Path $root -Force | Out-Null
$serverArgs = @(
    $server, '--model', $model, '--host', '127.0.0.1', '--port', '18136',
    '--alias', 'qwen38-dspark-qualification', '--ctx-size', '2048', '--parallel', '1',
    '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--flash-attn', 'on',
    '--batch-size', '2048', '--ubatch-size', '128', '--n-gpu-layers', 'all',
    '--device', 'CUDA0', '--split-mode', 'none', '--fit', 'off', '--cache-ram', '0',
    '--ctx-checkpoints', '0', '--no-cache-idle-slots', '--no-cache-prompt', '--no-webui',
    '--metrics', '--reasoning', 'off', '-lv', '4', '--kv-unified'
)
$commandPath = Join-Path $root 'command.json'
WriteJson $commandPath ([ordered]@{schema='ai-loader-qwen38-dspark-qualification-arm/v1';tier='P1';arm='target-only';dynamic_rs=$false;arguments=$serverArgs})
$tokens = @(0..255)
$tokenSha = TextSha ($tokens | ConvertTo-Json -Compress)
$waves = @()
for ($wave = 0; $wave -lt 7; ++$wave) {
    $waves += ,[ordered]@{wave=$wave;clients=@([ordered]@{client=0;seed=20260819;tokens=$tokens;token_sha256=$tokenSha;content_sha256='content'})}
}
$wavesPath = Join-Path $root 'waves.json'
WriteJson $wavesPath ([ordered]@{schema='ai-loader-qwen38-dspark-qualification-arm/v1';waves=$waves;metric_snapshots=@()})
$receiptPath = Join-Path $root 'receipt.json'
WriteJson $receiptPath ([ordered]@{
    schema='ai-loader-qwen38-dspark-qualification-arm/v1';tier='P1';arm='target-only';dynamic_rs=$false
    artifacts=[ordered]@{
        command=[ordered]@{path='command.json';sha256=Sha $commandPath}
        waves=[ordered]@{path='waves.json';sha256=Sha $wavesPath}
    }
})
$summaryPath = Join-Path $output 'summary.json'
WriteJson $summaryPath ([ordered]@{schema='ai-loader-qwen38-target-trace-summary/v1';status='completed';tier='P1';mode='target-only'})
$bundle = @()
Get-ChildItem -LiteralPath (Split-Path -Parent $server) -Filter '*.dll' -File | Sort-Object Name | ForEach-Object {
    $bundle += ,[ordered]@{name=$_.Name;sha256=Sha $_.FullName}
}
$manifestPath = Join-Path $output 'manifest.json'
WriteJson $manifestPath ([ordered]@{
    schema='ai-loader-qwen38-dspark-qualification-manifest/v1';identity=('1' * 64)
    components=[ordered]@{
        schema='ai-loader-qwen38-dspark-qualification-identity/v1';tier='P1';trace_mode='target-only';dynamic_rs=$false
        runner_sha256=Sha $runner;wrapper_sha256=Sha $wrapper;guard_sha256=Sha $MyInvocation.MyCommand.Path
        server_sha256=Sha $server;target_model_sha256=Sha $model;runtime_bundle=$bundle
        server_environment=[ordered]@{
            schema='ai-loader-qwen38-server-environment/v1';cuda_visible_devices='GPU-test';cuda_device_order='PCI_BUS_ID'
            dynamic_rs='0';adaptive_draft='0';scrubbed_prefixes=@('LLAMA_','GGML_','CUDA_')
        }
        arm_arguments=@([ordered]@{name='target-only';arguments=$serverArgs})
    }
})
$validationPath = Join-Path $output 'validation.json'
WriteJson $validationPath ([ordered]@{
    schema='ai-loader-qwen38-target-trace-validation/v1';identity=('1' * 64);qualification_identity=('1' * 64);valid=$true;token_trace_sha256=$tokenSha
    manifest_sha256=Sha $manifestPath;waves_sha256=Sha $wavesPath;command_sha256=Sha $commandPath
    receipt_sha256=Sha $receiptPath;summary_sha256=Sha $summaryPath
})
$contamination = @([Environment]::GetEnvironmentVariables('Process').Keys | Where-Object {
    ([string]$_).StartsWith('LLAMA_', [StringComparison]::OrdinalIgnoreCase) -or
    ([string]$_).StartsWith('GGML_', [StringComparison]::OrdinalIgnoreCase) -or
    ([string]$_).StartsWith('CUDA_', [StringComparison]::OrdinalIgnoreCase)
})
@{
    count=1;task_arguments=@($TaskArguments);min_ram=$MinFreeRamGiB;min_vram=$MinFreeVramMiB
    max_used=$MaxUsedVramMiB;poll=$PollMilliseconds;contamination=@($contamination)
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $env:QWEN38_TARGET_TRACE_CAPTURE -Encoding utf8
[pscustomobject]@{Status='completed';ExitCode=0;Schema='fake-guard/v1';GpuUuid='GPU-TEST';GpuIndex=$GpuIndex}
'@ | Set-Content -LiteralPath (Join-Path $scripts 'Invoke-ExclusiveGpuTask.ps1') -Encoding utf8
    $fakePython = Join-Path $testRoot 'fake-python.cmd'
    @'
@echo off
echo {"schema":"ai-loader-qwen38-target-trace-validation/v1","valid":true,"gpu_uuid":"GPU-TEST","identity":"1111111111111111111111111111111111111111111111111111111111111111"}
exit /b 0
'@ | Set-Content -LiteralPath $fakePython -Encoding ascii

    $serverDir = Join-Path $testRoot 'runtime'
    New-Item -ItemType Directory -Path $serverDir -Force | Out-Null
    $server = Join-Path $serverDir 'llama-server.exe'
    $model = Join-Path $testRoot 'target.gguf'
    [IO.File]::WriteAllText($server, 'server')
    [IO.File]::WriteAllText($model, 'model')
    [IO.File]::WriteAllText((Join-Path $serverDir 'ggml.dll'), 'ggml')
    [IO.File]::WriteAllText((Join-Path $serverDir 'llama.dll'), 'llama')
    $modelSha = (Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant()
    $common = [ordered]@{
        host='127.0.0.1';port=18136;batch_size=2048;ubatch_size=128;target_kv='q8_0';draft_kv='q8_0'
        flash_attn=$true;n_gpu_layers='all';device='CUDA0';split_mode='none';fit='off';cache_ram=0
        ctx_checkpoints=0;cache_idle_slots=$false;cache_prompt=$false;kv_unified=$true;metrics=$true
        reasoning='off';log_verbosity=4
    }
    $workload = [ordered]@{
        endpoint='/completion';stream=$true;prompt_token_id=1;prompt_tokens_per_request=736
        output_tokens_per_request=256;temperature=0;seed_base=20260819;seed_rule='seed_base + client_index'
        ignore_eos=$true;cache_prompt=$false;warmup_waves=1;measured_waves=6;request_barrier=$true
        exact_stop_type='limit';truncated=$false
    }
    $profile = Join-Path $testRoot 'unused-profile.json'
    $tiers = @(
        [ordered]@{name='P1';parallel=1;ctx_size=2048;profile_output=$profile;arms=@(
            [ordered]@{name='target-only';speculation='none'},
            [ordered]@{name='static-dspark';speculation='draft-dspark';n_max=7;dynamic_rs=$false},
            [ordered]@{name='sps-execution';speculation='draft-dspark';n_max=7;sps_profile='p1';dynamic_rs=$false})},
        [ordered]@{name='P4';parallel=4;ctx_size=8192;profile_output=$profile;arms=@(
            [ordered]@{name='target-only';speculation='none'},
            [ordered]@{name='static-dspark-matched-cap3';speculation='draft-dspark';n_max=3;dynamic_rs=$false},
            [ordered]@{name='static-dspark-best-cap2';speculation='draft-dspark';n_max=2;dynamic_rs=$false},
            [ordered]@{name='sps-execution';speculation='draft-dspark';n_max=3;sps_profile='p4';dynamic_rs=$false})},
        [ordered]@{name='P8';parallel=8;ctx_size=8192;profile_output=$profile;arms=@(
            [ordered]@{name='target-only';speculation='none'},
            [ordered]@{name='static-dspark-cap1';speculation='draft-dspark';n_max=1;dynamic_rs=$false},
            [ordered]@{name='sps-execution';speculation='draft-dspark';n_max=1;sps_profile='p8';dynamic_rs=$false})}
    )
    $plan = Join-Path $testRoot 'plan.json'
    [IO.File]::WriteAllText($plan, ([ordered]@{
        schema='ai-loader-qwen38-dspark-production-qualification/v1';status='test';created_for='CPU target trace wrapper test'
        execution_policy=[ordered]@{
            gpu_processes_max=1;server_arms_sequential_only=$true;outer_guard_required=$true;port=18136
            preflight_requires_port_free=$true;preflight_requires_zero_llama_processes=$true
            postflight_requires_zero_llama_processes=$true;stop_on_guard_violation=$true
            stop_on_any_http_or_server_error=$true;max_used_vram_mib=27000;min_free_vram_mib=5000
            min_free_system_ram_gib=18;poll_interval_ms=250
        }
        artifacts=[ordered]@{server=$server;target_model=$model;draft_model=(Join-Path $testRoot 'unused-draft.gguf');target_model_sha256=$modelSha;draft_model_sha256=('0'*64)}
        common_server_args=$common;deterministic_workload=$workload;tiers=$tiers
        correctness_gates=@{};reported_metrics=@();promotion_gates=@{};estimated_runtime_minutes=@{}
    } | ConvertTo-Json -Depth 14), (New-Object Text.UTF8Encoding($false)))

    $env:QWEN38_TARGET_TRACE_CAPTURE = $capture
    $env:LLAMA_ARG_CTX_SIZE = '9999'
    $env:GGML_CUDA_DISABLE_GRAPHS = '1'
    $env:CUDA_LAUNCH_BLOCKING = '1'
    $env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS = '1'
    $python = $fakePython
    $output = Join-Path $testRoot 'fresh\p1-final-v5'
    & (Join-Path $scripts 'Invoke-Qwen38TargetTrace.ps1') `
        -PlanPath $plan -ServerPath $server -ModelPath $model -OutputDir $output `
        -RuntimeLabel 'cpu-wrapper-test' -PythonPath $python -GpuIndex 2 `
        -MaxRuntimeSeconds 1900 -JobTimeoutSeconds 1800

    $record = Get-Content -LiteralPath $capture -Raw | ConvertFrom-Json
    Assert-True ($record.count -eq 1) 'target trace wrapper did not invoke exactly one guard'
    Assert-True ($record.min_ram -eq 18 -and $record.min_vram -eq 5000 -and $record.max_used -eq 27000) 'guard limits changed'
    Assert-True ($record.poll -eq 250) 'guard poll interval changed'
    Assert-True (@($record.contamination).Count -eq 0) 'runtime environment prefixes were not scrubbed before the guard'
    $arguments = @($record.task_arguments)
    Assert-True (@($arguments | Where-Object { $_ -ceq 'target-trace' }).Count -eq 1) 'target-trace command was not selected exactly once'
    Assert-True (-not ($arguments -contains 'run')) 'full qualification command leaked into target trace'
    Assert-True (-not ($arguments -contains '--draft-model')) 'draft model leaked into target trace'
    Assert-True (-not ($arguments -contains '--profile')) 'SPS profile leaked into target trace'
    Assert-True (Test-Path -LiteralPath (Join-Path $output 'guard-receipt.json') -PathType Leaf) 'guard/postflight receipt was not persisted'
    $receipt = Get-Content -LiteralPath (Join-Path $output 'guard-receipt.json') -Raw | ConvertFrom-Json
    Assert-True ($receipt.guard_status -ceq 'completed' -and $receipt.trace_validation -ceq 'passed') 'guard receipt did not prove validation success'
    Assert-True ($receipt.postflight.relevant_processes -eq 0 -and $receipt.postflight.port_18136_free -and $receipt.postflight.recursive_gpu_leases -eq 0) 'postflight proof is incomplete'
    Assert-True ($env:LLAMA_ARG_CTX_SIZE -ceq '9999' -and $env:GGML_CUDA_DISABLE_GRAPHS -ceq '1' -and
            $env:CUDA_LAUNCH_BLOCKING -ceq '1' -and
            $env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS -ceq '1') 'caller environment was not restored'

    $env:QWEN38_TARGET_TRACE_FAKE_FAIL = '1'
    $failureOutput = Join-Path $testRoot 'failure\p1-final-v5'
    $failureRejected = $false
    try {
        & (Join-Path $scripts 'Invoke-Qwen38TargetTrace.ps1') `
            -PlanPath $plan -ServerPath $server -ModelPath $model -OutputDir $failureOutput `
            -RuntimeLabel 'cpu-wrapper-failure-test' -PythonPath $python `
            -MaxRuntimeSeconds 1900 -JobTimeoutSeconds 1800
    } catch {
        $failureRejected = $_.Exception.Message -like '*fake target trace guard failure*Guard/postflight receipt:*'
    }
    Assert-True $failureRejected 'guard failure did not surface the persisted receipt path'
    $failureReceipts = @(Get-ChildItem -LiteralPath (Split-Path -Parent $failureOutput) -Filter 'p1-final-v5.guard-*.receipt.json' -File)
    Assert-True ($failureReceipts.Count -eq 1) 'guard failure did not persist exactly one external receipt'
    $failureReceipt = Get-Content -LiteralPath $failureReceipts[0].FullName -Raw | ConvertFrom-Json
    Assert-True ($failureReceipt.guard_status -ceq 'failed' -and $failureReceipt.trace_validation -ceq 'not_run') 'failure receipt status is incorrect'
    Assert-True ($failureReceipt.postflight.relevant_processes -eq 0 -and $failureReceipt.postflight.port_18136_free -and $failureReceipt.postflight.recursive_gpu_leases -eq 0) 'failure receipt lacks postflight proof'
    Remove-Item Env:QWEN38_TARGET_TRACE_FAKE_FAIL

    $preflightOutput = Join-Path $testRoot 'preflight\p1-final-v5'
    New-Item -ItemType Directory -Path $preflightOutput -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $preflightOutput 'occupied.txt'), 'occupied')
    $preflightRejected = $false
    try {
        & (Join-Path $scripts 'Invoke-Qwen38TargetTrace.ps1') `
            -PlanPath $plan -ServerPath $server -ModelPath $model -OutputDir $preflightOutput `
            -RuntimeLabel 'cpu-wrapper-preflight-test' -PythonPath $python `
            -MaxRuntimeSeconds 1900 -JobTimeoutSeconds 1800
    } catch {
        $preflightRejected = $_.Exception.Message -like '*never resumes or overwrites*Guard/postflight receipt:*'
    }
    Assert-True $preflightRejected 'preflight failure did not surface its postflight receipt'
    $preflightReceipts = @(Get-ChildItem -LiteralPath (Split-Path -Parent $preflightOutput) -Filter 'p1-final-v5.guard-*.receipt.json' -File)
    Assert-True ($preflightReceipts.Count -eq 1) 'preflight failure did not persist exactly one receipt'
    $preflightReceipt = Get-Content -LiteralPath $preflightReceipts[0].FullName -Raw | ConvertFrom-Json
    Assert-True ($preflightReceipt.stage -ceq 'preflight' -and -not $preflightReceipt.guard_started -and $preflightReceipt.guard_status -ceq 'not_started') 'preflight receipt incorrectly claims the guard started'
    Assert-True ($preflightReceipt.postflight.relevant_processes -eq 0 -and $preflightReceipt.postflight.port_18136_free -and $preflightReceipt.postflight.recursive_gpu_leases -eq 0) 'preflight receipt lacks postflight proof'
    'Qwen38 target-only trace wrapper CPU-only tests: PASS'
} finally {
    if ($null -eq $oldCapture) { Remove-Item Env:QWEN38_TARGET_TRACE_CAPTURE -ErrorAction SilentlyContinue } else { $env:QWEN38_TARGET_TRACE_CAPTURE = $oldCapture }
    if ($null -eq $oldLlama) { Remove-Item Env:LLAMA_ARG_CTX_SIZE -ErrorAction SilentlyContinue } else { $env:LLAMA_ARG_CTX_SIZE = $oldLlama }
    if ($null -eq $oldGgml) { Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue } else { $env:GGML_CUDA_DISABLE_GRAPHS = $oldGgml }
    if ($null -eq $oldCuda) { Remove-Item Env:CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue } else { $env:CUDA_LAUNCH_BLOCKING = $oldCuda }
    if ($null -eq $oldProjection) { Remove-Item Env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS -ErrorAction SilentlyContinue } else { $env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS = $oldProjection }
    Remove-Item Env:QWEN38_TARGET_TRACE_FAKE_FAIL -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $testRoot) { Remove-Item -LiteralPath $testRoot -Recurse -Force }
}
