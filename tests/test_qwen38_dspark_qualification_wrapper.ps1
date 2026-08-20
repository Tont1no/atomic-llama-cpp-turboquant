$ErrorActionPreference = 'Stop'

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) { throw $Message }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('qwen38-qualification-wrapper-' + [guid]::NewGuid().ToString('N'))
$scripts = Join-Path $testRoot 'scripts'
$capture = Join-Path $testRoot 'capture.json'
$oldCapture = $env:QWEN38_QUALIFICATION_CAPTURE
try {
    New-Item -ItemType Directory -Path $scripts -Force | Out-Null
    foreach ($name in @('Invoke-Qwen38DsparkQualification.ps1', 'qwen38_dspark_qualification.py', 'dspark_sps_profile.py')) {
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
$prior = if (Test-Path -LiteralPath $env:QWEN38_QUALIFICATION_CAPTURE) {
    Get-Content -Raw -LiteralPath $env:QWEN38_QUALIFICATION_CAPTURE | ConvertFrom-Json
} else { $null }
$count = if ($prior) { [int]$prior.Count + 1 } else { 1 }
$outIndex = [array]::IndexOf($TaskArguments, '--output-dir')
$out = $TaskArguments[$outIndex + 1]
New-Item -ItemType Directory -Path $out -Force | Out-Null
'{}' | Set-Content -LiteralPath (Join-Path $out 'summary.json') -Encoding utf8
@{
    Count = $count; Executable = $Executable; TaskArguments = @($TaskArguments)
    TaskWorkingDirectory = $TaskWorkingDirectory; MinFreeRamGiB = $MinFreeRamGiB
    MinFreeVramMiB = $MinFreeVramMiB; MaxUsedVramMiB = $MaxUsedVramMiB
    GpuIndex = $GpuIndex; PollMilliseconds = $PollMilliseconds
    MaxRuntimeSeconds = $MaxRuntimeSeconds; StdoutPath = $StdoutPath; StderrPath = $StderrPath
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $env:QWEN38_QUALIFICATION_CAPTURE -Encoding utf8
[pscustomobject]@{ Schema = 'fake-guard/v1'; ExitCode = 0 }
'@ | Set-Content -LiteralPath (Join-Path $scripts 'Invoke-ExclusiveGpuTask.ps1') -Encoding utf8

    $server = Join-Path $testRoot 'server.exe'
    $model = Join-Path $testRoot 'target.gguf'
    $draft = Join-Path $testRoot 'draft.gguf'
    $profile = Join-Path $testRoot 'profile.json'
    foreach ($path in @($server, $model, $draft, $profile)) { [IO.File]::WriteAllText($path, 'mock') }
    $common = [ordered]@{
        host='127.0.0.1'; port=18136; batch_size=2048; ubatch_size=128; target_kv='q8_0'; draft_kv='q8_0'
        flash_attn=$true; n_gpu_layers='all'; device='CUDA0'; split_mode='none'; fit='off'; cache_ram=0
        ctx_checkpoints=0; cache_idle_slots=$false; cache_prompt=$false; kv_unified=$true; metrics=$true
        reasoning='off'; log_verbosity=4
    }
    $workload = [ordered]@{
        endpoint='/completion'; stream=$true; prompt_token_id=1; prompt_tokens_per_request=736
        output_tokens_per_request=256; temperature=0; seed_base=20260819; seed_rule='seed_base + client_index'
        ignore_eos=$true; cache_prompt=$false; warmup_waves=1; measured_waves=6; request_barrier=$true
        exact_stop_type='limit'; truncated=$false
    }
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
    $planJson = [ordered]@{
        schema='ai-loader-qwen38-dspark-production-qualification/v1'
        status='test'; created_for='CPU wrapper test'
        execution_policy=[ordered]@{
            gpu_processes_max=1; server_arms_sequential_only=$true; outer_guard_required=$true; port=18136
            preflight_requires_port_free=$true; preflight_requires_zero_llama_processes=$true
            postflight_requires_zero_llama_processes=$true; stop_on_guard_violation=$true
            stop_on_any_http_or_server_error=$true; max_used_vram_mib=27000; min_free_vram_mib=5000
            min_free_system_ram_gib=18; poll_interval_ms=250
        }
        artifacts=[ordered]@{server=$server;target_model=$model;draft_model=$draft}
        common_server_args=$common; deterministic_workload=$workload; tiers=$tiers
        correctness_gates=@{}; reported_metrics=@(); promotion_gates=@{}; estimated_runtime_minutes=@{}
    } | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText($plan, $planJson, (New-Object Text.UTF8Encoding($false)))

    $env:QWEN38_QUALIFICATION_CAPTURE = $capture
    $python = (Get-Command python.exe -ErrorAction Stop).Source
    $output = Join-Path $testRoot 'result'
    & (Join-Path $scripts 'Invoke-Qwen38DsparkQualification.ps1') `
        -PlanPath $plan -Tier P4 -ServerPath $server -ModelPath $model -DraftModelPath $draft `
        -ProfilePath $profile -OutputDir $output -RuntimeLabel 'cpu-wrapper-test' -PythonPath $python `
        -GpuIndex 2 -MaxRuntimeSeconds 999 -JobTimeoutSeconds 900

    $record = Get-Content -Raw -LiteralPath $capture | ConvertFrom-Json
    Assert-True ($record.Count -eq 1) 'wrapper did not invoke exactly one outer guard'
    Assert-True ($record.MinFreeRamGiB -eq 18) 'RAM guard default changed'
    Assert-True ($record.MinFreeVramMiB -eq 5000) 'free VRAM guard default changed'
    Assert-True ($record.MaxUsedVramMiB -eq 27000) 'used VRAM guard default changed'
    Assert-True ($record.PollMilliseconds -eq 250) 'guard poll default changed'
    $runner = @($record.TaskArguments)
    Assert-True (($runner | Where-Object { $_ -eq 'run' }).Count -eq 1) 'runner command is not singular'
    Assert-True ($runner -contains '--tier') 'tier was not forwarded'
    Assert-True ($runner -contains 'P4') 'P4 was not forwarded'
    Assert-True (-not ($runner -contains '--dynamic-rs')) 'wrapper exposed Dynamic-RS enablement'
    Assert-True (Test-Path -LiteralPath (Join-Path $output 'guard-receipt.json') -PathType Leaf) 'guard receipt was not written'

    $rejected17 = $false
    try {
        & (Join-Path $scripts 'Invoke-Qwen38DsparkQualification.ps1') `
            -PlanPath $plan -Tier P4 -ServerPath $server -ModelPath $model -DraftModelPath $draft `
            -ProfilePath $profile -OutputDir (Join-Path $testRoot 'result-17') `
            -RuntimeLabel 'cpu-wrapper-test-17' -PythonPath $python `
            -MinFreeRamGiB 17 -MaxRuntimeSeconds 999 -JobTimeoutSeconds 900
    } catch {
        $rejected17 = $_.Exception.Message -like '*18 GiB RAM*'
    }
    Assert-True $rejected17 'wrapper did not reject a 17 GiB RAM floor'
    $afterReject = Get-Content -Raw -LiteralPath $capture | ConvertFrom-Json
    Assert-True ($afterReject.Count -eq 1) '17 GiB rejection reached the outer guard'
    'Qwen38 DSpark qualification wrapper CPU-only tests: PASS'
} finally {
    $env:QWEN38_QUALIFICATION_CAPTURE = $oldCapture
    if (Test-Path -LiteralPath $testRoot) { Remove-Item -LiteralPath $testRoot -Recurse -Force }
}
