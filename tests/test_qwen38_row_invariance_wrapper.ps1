[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Assert-True([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw "ASSERTION FAILED: $Message" }
}

function Get-Sha256([string] $Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Set-Anchor([string] $Text, [string] $Variable, [string] $Hash) {
    $pattern = "(?m)^\`$$([regex]::Escape($Variable)) = '[0-9a-f]{64}'$"
    $replacement = "`$$Variable = '$Hash'"
    $updated = [regex]::Replace($Text, $pattern, $replacement)
    Assert-True ($updated -cne $Text) "anchor $Variable was not replaced"
    return $updated
}

function Read-Receipt([string] $OutputRoot, [string] $Mode = 'head-candidate') {
    return Get-Content -LiteralPath (Join-Path $OutputRoot "$Mode.guard-receipt.json") -Raw | ConvertFrom-Json
}

$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$sourceWrapper = Join-Path $workspaceRoot 'scripts\Invoke-Qwen38RowInvariance.ps1'
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("qwen38-row-wrapper-" + [Guid]::NewGuid().ToString('N'))
$scripts = Join-Path $testRoot 'scripts'
$bin = Join-Path $testRoot 'bin'
$output = Join-Path $testRoot 'output'
$prior = [ordered]@{}
$contaminants = [ordered]@{
    LLAMA_TEST_ROW_WRAPPER = 'llama-contamination'
    GGML_BACKEND_PATH = 'fake-backend-path'
    CUDA_LAUNCH_BLOCKING = '1'
}

try {
    New-Item -ItemType Directory -Path $scripts, $bin, $output -Force | Out-Null
    $wrapper = Join-Path $scripts 'Invoke-Qwen38RowInvariance.ps1'
    Copy-Item -LiteralPath $sourceWrapper -Destination $wrapper

    $fakeGuard = @'
param(
    [string] $Executable, [string[]] $TaskArguments, [string] $TaskWorkingDirectory,
    [int] $GpuIndex, [int] $MinFreeRamGiB, [int] $MinFreeVramMiB,
    [int] $MaxUsedVramMiB, [int] $MaxRuntimeSeconds,
    [string] $StdoutPath, [string] $StderrPath
)
if ($env:LLAMA_TEST_ROW_WRAPPER -or $env:GGML_BACKEND_PATH -or $env:CUDA_LAUNCH_BLOCKING) {
    throw 'row-invariance wrapper leaked a scrubbed environment variable'
}
if ($env:LLAMACPP_QWEN38_ROW_INVARIANCE_GUARD -cne 'EXCLUSIVE_GPU') {
    throw 'row-invariance wrapper did not bind its sentinel'
}
if ($env:FAKE_ROW_GUARD_FAILURE -eq '1') { throw 'synthetic guarded failure' }

function New-RouteRecord([string] $Id, [hashtable] $Selected) {
    $record = [ordered]@{
        type = 'routes'; id = $Id; fp8 = 0; fp8_batch = 0; nvfp4_mmvq_m1 = 0; nvfp4_mmvq_m8 = 0
        nvfp4_fused_ffn = 0; nvfp4_row_invariant_head_scaled = 0; nvfp4_row_invariant_ffn_fused = 0
        nvfp4_row_invariant_ffn_down_scaled = 0; gdn = 0; gdn_fused_cache = 0; rms_norm = 0
        rms_norm_mul = 0; ssm_conv = 0; ssm_conv_silu = 0; l2_norm = 0; silu = 0
        silu_mul = 0; bf16_mmvf = 0; bf16_cublas = 0
        bf16_row_invariant_beta = 0; bf16_row_invariant_alpha = 0
    }
    foreach ($name in $Selected.Keys) { $record[$name] = $Selected[$name] }
    if ($env:FAKE_ROW_EXTRA_OPERATOR_ROUTE -eq '1' -and $Id -like 'qwen35-*') {
        $record.nvfp4_mmvq_m1 = 1
    }
    return $record
}

$mode = if ($TaskArguments -contains '--projection-candidate') { 'projection-candidate' }
    elseif ($TaskArguments -contains '--fp8') { 'fp8' }
    elseif ($TaskArguments -contains '--gdn') { 'gdn' }
    elseif ($TaskArguments -contains '--bf16-candidate') { 'bf16-candidate' }
    elseif ($TaskArguments -contains '--bf16-projections') { 'bf16-projections' }
    elseif ($TaskArguments -contains '--rms') { 'rms' }
    elseif ($TaskArguments -contains '--ssm-conv') { 'ssm-conv' }
    elseif ($TaskArguments -contains '--l2') { 'l2' }
    elseif ($TaskArguments -contains '--gated-norm') { 'gated-norm' }
    elseif ($TaskArguments -contains '--fattn-vec-pb1') { 'fattn-vec-pb1' }
    else { 'head-candidate' }
if ($mode -ceq 'fattn-vec-pb1') {
    $id = 'qwen35-fattn-d256-q8-gqa6-vec-cols2-pb1'
    $lines = @('{"type":"config","mode":"fattn-vec-pb1","cases":1,"replays":8,"order":"alternating","fixture":"deterministic_qwen35_fattn_exact","comparison":"bitwise_f32"}')
    $lines += (@{ type = 'route_negative_control'; id = $id; m = 3; tagged = $false
        pb1_candidate = 0; fattn_vec = 0; fattn_mma_f16 = 1; fattn_tile = 0
        content_free = $true } | ConvertTo-Json -Compress)
    foreach ($m in 3..8) {
        $pb = if ($env:FAKE_ROW_BAD_FATTN_PB1 -eq '1' -and $m -eq 3) { 2 } else { 1 }
        $lines += (@{ type = 'fattn_pb1_route_probe'; id = $id; m = $m; scalar_rows = $m
            scalar_q_cols = 1; scalar_ncols = 1; scalar_ntiles_x = 1; scalar_ntiles_kv = 4
            scalar_parallel_blocks = 1; batch_q_cols = $m; batch_ncols = 2
            batch_ntiles_x = [int](($m + 1) / 2); batch_ntiles_kv = 4
            batch_parallel_blocks = $pb; pb1_candidate = $m + 1; fattn_vec = $m + 1
            fattn_mma_f16 = 0; fattn_tile = 0; causal_mask_attested = $true
            content_free = $true } | ConvertTo-Json -Compress)
        $lines += (@{ type = 'width_route'; id = $id; m = $m; replays = 8; finite = $true
            comparison = 'bitwise_f32'; scalar_rows_independent = $true
            causal_mask_attested = $true } | ConvertTo-Json -Compress)
    }
    $lines += (@{ type = 'case'; id = $id; status = 'passed'; device_scope = 'sm120'; d = 256
        n_kv = 1024; h_q = 24; h_kv = 4; gqa = 6; q_type = 'f32'; k_type = 'q8_0'
        v_type = 'q8_0'; mask_type = 'f16'; causal_mask_attested = $true; scale = 0.0625
        max_bias = 0.0; softcap = 0.0; prec = 'f32'; scalar_m = 1; min_batch_m = 3
        max_batch_m = 8; widths = 6; replays = 8; ncols_scalar = 1; ncols_batch = 2
        forced_parallel_blocks = 1; finite = $true; comparison = 'bitwise_f32' } | ConvertTo-Json -Compress)
    $lines += '{"type":"complete","status":"passed","mode":"fattn-vec-pb1","cases":1}'
} elseif ($mode -ceq 'projection-candidate') {
    $ids = @(
        'nvfp4-qwen35-ffn-gate-up-swiglu-row-invariant',
        'nvfp4-qwen35-ffn-down-row-invariant'
    )
    $lines = @('{"type":"config","mode":"projection-candidate","cases":2,"replays":8,"order":"alternating","fixture":"deterministic_cancellation_sensitive","comparison":"bitwise_f32"}')
    foreach ($id in $ids) {
        $routeField = if ($id -like '*gate-up*') { 'nvfp4_row_invariant_ffn_fused' } else { 'nvfp4_row_invariant_ffn_down_scaled' }
        $lines += (@{ type = 'route_negative_control'; id = $id; selected_route = 0 } | ConvertTo-Json -Compress)
        foreach ($m in 2..16) {
            $before = $m - 2
            $lines += (@{ type = 'width_route'; id = $id; m = $m; scalar_route = 2
                selected_before = $before; selected_after = $before + 1; selected_delta = 1; generic_m8 = 0 } | ConvertTo-Json -Compress)
        }
        $lines += (@{ type = 'width_coverage'; id = $id; status = 'passed'; min_m = 2; max_m = 16
            comparison = 'bitwise_f32'; route_checked_each_width = $true } | ConvertTo-Json -Compress)
        $lines += (@{ type = 'case'; id = $id; status = 'passed'; replays = 8; fixtures = 3; output_scale = $true } | ConvertTo-Json -Compress)
        $lines += (New-RouteRecord $id @{ $routeField = 15 } | ConvertTo-Json -Compress)
    }
    $lines += '{"type":"complete","status":"passed","mode":"projection-candidate","cases":2}'
} elseif ($mode -ceq 'fp8') {
    $ids = @('fp8-k5120-n1024', 'fp8-k5120-n6144', 'fp8-k6144-n5120', 'fp8-k5120-n10240', 'fp8-k5120-n12288')
    $lines = @('{"type":"config","mode":"fp8","cases":5,"replays":8,"order":"alternating","fixture":"deterministic_cancellation_sensitive","comparison":"bitwise_f32"}')
    foreach ($id in $ids) {
        foreach ($m in 2..8) {
            $before = $m - 2
            if ($env:FAKE_ROW_BAD_FP8_CHAIN -eq '1' -and $id -eq $ids[0] -and $m -eq 4) { $before = 99 }
            $lines += (@{ type = 'width_route'; id = $id; m = $m; fp8_batch_before = $before
                fp8_batch_after = $before + 1; fp8_batch_delta = 1; finite = $true; comparison = 'bitwise_f32'; replays = 8 } | ConvertTo-Json -Compress)
        }
        $lines += (@{ type = 'width_coverage'; id = $id; status = 'passed'; min_m = 2; max_m = 8
            widths = 7; replays = 8; finite = $true; comparison = 'bitwise_f32'; route_checked_each_width = $true } | ConvertTo-Json -Compress)
        $lines += (@{ type = 'case'; id = $id; status = 'passed'; replays = 8; min_batch_m = 2
            max_batch_m = 8; widths = 7; finite = $true; comparison = 'bitwise_f32' } | ConvertTo-Json -Compress)
        $extraNvfp4 = if ($env:FAKE_ROW_EXTRA_NVFP4_ROUTE -eq '1') { 1 } else { 0 }
        $lines += (New-RouteRecord $id @{ fp8 = 16; fp8_batch = 8; nvfp4_fused_ffn = $extraNvfp4 } | ConvertTo-Json -Compress)
    }
    $lines += '{"type":"complete","status":"passed","mode":"fp8","cases":5}'
} elseif ($mode -ceq 'gdn') {
    $fused = if ($env:FAKE_ROW_BAD_GDN_ROUTE -eq '1') { 0 } else { 1 }
    $ordinaryGdn = if ($env:FAKE_ROW_GDN_FALLBACK -eq '1') { 1 } else { 0 }
    $lines = @('{"type":"config","mode":"gdn","cases":2,"replays":8,"order":"alternating","fixture":"deterministic_cancellation_sensitive","comparison":"bitwise_f32"}')
    $lines += '{"type":"case","id":"gdn-k1-vs-k8","status":"passed","replays":8,"slot":0,"finite":true,"comparison":"bitwise_f32"}'
    $extraNvfp4 = if ($env:FAKE_ROW_EXTRA_NVFP4_ROUTE -eq '1') { 1 } else { 0 }
    $lines += (New-RouteRecord 'gdn-k1-vs-k8' @{ nvfp4_row_invariant_head_scaled = $extraNvfp4; gdn = $ordinaryGdn; gdn_fused_cache = $fused } | ConvertTo-Json -Compress)
    $lines += '{"type":"case","id":"gdn-seq1-vs-batch8","status":"passed","replays":8,"snapshot_order":"newest_first","finite":true,"comparison":"bitwise_f32"}'
    $lines += (New-RouteRecord 'gdn-seq1-vs-batch8' @{ nvfp4_row_invariant_head_scaled = $extraNvfp4; gdn = $ordinaryGdn; gdn_fused_cache = $fused } | ConvertTo-Json -Compress)
    $lines += '{"type":"complete","status":"passed","mode":"gdn","cases":2}'
} elseif ($mode -ceq 'bf16-projections') {
    $ids = @('qwen35-bf16-beta-k5120-n48', 'qwen35-bf16-alpha-k5120-n48')
    $lines = @('{"type":"config","mode":"bf16-projections","cases":2,"replays":8,"order":"alternating","fixture":"deterministic_nonzero_shape_exact","comparison":"bitwise_f32"}')
    foreach ($id in $ids) {
        foreach ($m in 2..8) {
            $before = $m - 2
            $scalarCublas = if ($env:FAKE_ROW_BAD_BF16_PROBE -eq '1' -and $id -eq $ids[0] -and $m -eq 2) { 1 } else { 0 }
            $lines += (@{ type = 'bf16_route_probe'; id = $id; m = $m; scalar_mmvf_delta = 8
                scalar_cublas_delta = $scalarCublas; batch_mmvf_delta = 0; batch_cublas_delta = 1
                input_bf16_roundtrip_sensitive = $true } | ConvertTo-Json -Compress)
            $lines += (@{ type = 'width_route'; id = $id; m = $m; bf16_mmvf = 8
                bf16_cublas_before = $before; bf16_cublas_after = $before + 1; bf16_cublas_delta = 1
                finite = $true; comparison = 'bitwise_f32'; replays = 8 } | ConvertTo-Json -Compress)
        }
        $lines += (@{ type = 'width_coverage'; id = $id; status = 'passed'; min_m = 2; max_m = 8
            widths = 7; replays = 8; finite = $true; comparison = 'bitwise_f32'; route_checked_each_width = $true } | ConvertTo-Json -Compress)
        $lines += (@{ type = 'case'; id = $id; status = 'passed'; semantic = $(if ($id -like '*alpha*') { 'alpha' } else { 'beta' })
            replays = 8; min_batch_m = 2; max_batch_m = 8; widths = 7; finite = $true; comparison = 'bitwise_f32' } | ConvertTo-Json -Compress)
        $lines += (New-RouteRecord $id @{ bf16_mmvf = 112; bf16_cublas = 56 } | ConvertTo-Json -Compress)
    }
    $lines += '{"type":"complete","status":"passed","mode":"bf16-projections","cases":2}'
} elseif ($mode -ceq 'bf16-candidate') {
    $ids = @('qwen35-bf16-beta-k5120-n48-row-invariant', 'qwen35-bf16-alpha-k5120-n48-row-invariant')
    $lines = @('{"type":"config","mode":"bf16-candidate","cases":2,"replays":8,"order":"alternating","fixture":"deterministic_nonzero_shape_exact","comparison":"bitwise_f32"}')
    foreach ($id in $ids) {
        $alpha = $id -like '*alpha*'
        $semantic = if ($alpha) { 'alpha' } else { 'beta' }
        $transformed = if ($alpha) { 'add_softplus_mul' } else { 'sigmoid' }
        $candidateField = if ($alpha) { 'bf16_row_invariant_alpha' } else { 'bf16_row_invariant_beta' }
        $lines += (@{ type = 'route_negative_control'; id = $id; untagged_m8_cublas = 1
            tagged_noncontiguous_m8_cublas = 1; candidate_beta = 0; candidate_alpha = 0 } | ConvertTo-Json -Compress)
        foreach ($m in 2..8) {
            $candidateDelta = if ($env:FAKE_ROW_BAD_BF16_CANDIDATE_PROBE -eq '1' -and $id -eq $ids[0] -and $m -eq 2) { 0 } else { 1 }
            $lines += (@{ type = 'bf16_candidate_route_probe'; id = $id; m = $m; scalar_mmvf_delta = 8
                scalar_candidate_delta = 0; scalar_cublas_delta = 0; batch_mmvf_delta = 0
                batch_candidate_delta = $candidateDelta; batch_cublas_delta = 0
                input_bf16_roundtrip_sensitive = $true } | ConvertTo-Json -Compress)
            $lines += (@{ type = 'width_route'; id = $id; m = $m; replays = 8; finite = $true
                comparison = 'bitwise_f32'; raw_projection_bitwise = $true; transformed_path = $transformed } | ConvertTo-Json -Compress)
        }
        $samples = @(1..16 | ForEach-Object { [double] $_ })
        $candidateSamples = @(1..16 | ForEach-Object { [double] $_ / 2.0 })
        $lines += (@{ type = 'microbenchmark'; id = $id; m = 8; warmups = 4; samples_per_arm = 16
            order = 'AB_BA'; baseline = 'untagged_cublas'; candidate = 'tagged_row_invariant_mmvf'
            baseline_median_us = 8.0; baseline_p95_us = 16.0; candidate_median_us = 4.0
            candidate_p95_us = 8.0; median_ratio = 0.5; baseline_samples_us = $samples
            candidate_samples_us = $candidateSamples } | ConvertTo-Json -Compress)
        $lines += (@{ type = 'width_coverage'; id = $id; status = 'passed'; min_m = 2; max_m = 8
            widths = 7; replays = 8; finite = $true; comparison = 'bitwise_f32'; route_checked_each_width = $true } | ConvertTo-Json -Compress)
        $lines += (@{ type = 'case'; id = $id; status = 'passed'; semantic = $semantic
            transformed_path = $transformed; replays = 8; min_batch_m = 2; max_batch_m = 8
            widths = 7; finite = $true; comparison = 'bitwise_f32'; raw_projection_bitwise = $true } | ConvertTo-Json -Compress)
        $selected = @{ bf16_mmvf = 112; bf16_cublas = 16; $candidateField = 56 }
        $lines += (New-RouteRecord $id $selected | ConvertTo-Json -Compress)
    }
    $lines += '{"type":"complete","status":"passed","mode":"bf16-candidate","cases":2}'
} elseif ($mode -cin @('rms', 'l2', 'gated-norm')) {
    $ids = if ($mode -ceq 'rms') { @('qwen35-rms-norm-e5120') } elseif ($mode -ceq 'l2') {
        @('qwen35-l2-norm-q-s128-h16', 'qwen35-l2-norm-k-s128-h16')
    } else { @('qwen35-post-gdn-gated-norm-s128-h48') }
    $lines = @((@{ type = 'config'; mode = $mode; cases = $ids.Count; replays = 8; order = 'alternating'
        fixture = 'deterministic_nonzero_shape_exact'; comparison = 'bitwise_f32' } | ConvertTo-Json -Compress))
    foreach ($id in $ids) {
        foreach ($m in 2..8) {
            $before = ($m - 2) * 2
            $secondary = if ($mode -ceq 'gated-norm') { 1 } else { 0 }
            $lines += (@{ type = 'norm_route_probe'; id = $id; m = $m; scalar_selected_delta = 1
                batch_selected_delta = 1; scalar_secondary_delta = $secondary; batch_secondary_delta = $secondary
                scalar_fallback_delta = 0; batch_fallback_delta = 0 } | ConvertTo-Json -Compress)
            $widthRecord = [ordered]@{ type = 'width_route'; id = $id; m = $m; selected_before = $before
                selected_after = $before + 2; selected_delta = 2; scalar_selected_delta = 1
                batch_selected_delta = 1
                scalar_secondary_delta = $secondary; batch_secondary_delta = $secondary
                scalar_fallback_delta = 0; batch_fallback_delta = 0; finite = $true
                comparison = 'bitwise_f32'; replays = 8 }
            if ($env:FAKE_ROW_MISSING_NORM_ZERO -eq '1' -and $mode -ceq 'rms' -and $m -eq 2) {
                $widthRecord.Remove('scalar_fallback_delta')
            }
            $lines += ($widthRecord | ConvertTo-Json -Compress)
        }
        $lines += (@{ type = 'width_coverage'; id = $id; status = 'passed'; min_m = 2; max_m = 8
            widths = 7; replays = 8; finite = $true; comparison = 'bitwise_f32'; route_checked_each_width = $true } | ConvertTo-Json -Compress)
        $lines += (@{ type = 'case'; id = $id; status = 'passed'; replays = 8; min_batch_m = 2
            max_batch_m = 8; widths = 7; finite = $true; comparison = 'bitwise_f32' } | ConvertTo-Json -Compress)
        $selected = if ($mode -ceq 'rms') { @{ rms_norm_mul = 8 } } elseif ($mode -ceq 'l2') {
            @{ l2_norm = 8 }
        } else { @{ rms_norm_mul = 8; silu_mul = 8 } }
        $lines += (New-RouteRecord $id $selected | ConvertTo-Json -Compress)
    }
    $lines += (@{ type = 'complete'; status = 'passed'; mode = $mode; cases = $ids.Count } | ConvertTo-Json -Compress)
} elseif ($mode -ceq 'ssm-conv') {
    $id = 'qwen35-ssm-conv-d4-c10240'
    $lines = @('{"type":"config","mode":"ssm-conv","cases":1,"replays":8,"order":"alternating","fixture":"deterministic_nonzero_shape_exact","comparison":"bitwise_f32"}')
    $lines += (@{ type = 'ssm_conv_evolution'; id = $id; d_conv = 4; channels = 10240
        initial_cache_rows = 3; tokens = 8; scalar_steps = 8; batch_rows = 8; output_exact = $true
        evolved_cache_exact = $true; cache_mapping = 'eight_slots_newest_first'; finite = $true; replays = 8
        output_fnv1a64 = '0123456789abcdef'; cache_fnv1a64 = 'fedcba9876543210' } | ConvertTo-Json -Compress)
    $lines += (@{ type = 'case'; id = $id; status = 'passed'; replays = 8; finite = $true; comparison = 'bitwise_f32' } | ConvertTo-Json -Compress)
    $lines += (New-RouteRecord $id @{ ssm_conv_silu = 9 } | ConvertTo-Json -Compress)
    $lines += '{"type":"complete","status":"passed","mode":"ssm-conv","cases":1}'
} else {
    $lines = @(
        '{"type":"config","mode":"head-candidate","cases":1,"replays":8,"order":"alternating","fixture":"deterministic_cancellation_sensitive","comparison":"bitwise_f32"}',
        '{"type":"route_negative_control","id":"nvfp4-qwen35-lm-head-row-invariant","row_invariant_head":0}',
        '{"type":"case","id":"nvfp4-qwen35-lm-head-row-invariant","status":"passed","replays":8,"fixtures":3,"output_scale":true}',
        '{"type":"routes","id":"nvfp4-qwen35-lm-head-row-invariant","nvfp4_mmvq_m1":8,"nvfp4_mmvq_m8":0,"nvfp4_row_invariant_head_scaled":8}',
        '{"type":"complete","status":"passed","mode":"head-candidate","cases":1}'
    )
    $widthLines = foreach ($m in 2..16) {
        $before = $m - 2
        if ($env:FAKE_ROW_BAD_CHAIN -eq '1' -and $m -eq 4) { $before = 99 }
        @{ type = 'width_route'; id = 'nvfp4-qwen35-lm-head-row-invariant'; m = $m
           observed = @{ m1 = 2; tagged_scaled_before = $before; tagged_scaled_after = $before + 1; tagged_scaled_delta = 1; generic_m8 = 0 }
           expected = @{ m1_min = 1; tagged_scaled_delta_min = 1; generic_m8 = 0 } } | ConvertTo-Json -Compress
    }
    $lines = @($lines[0..1]) + @($widthLines) + @(
        '{"type":"width_coverage","id":"nvfp4-qwen35-lm-head-row-invariant","status":"passed","min_m":2,"max_m":16,"comparison":"bitwise_f32","route_checked_each_width":true}'
    ) + @($lines[2..4])
}
[System.IO.File]::WriteAllLines($StdoutPath, $lines)
[System.IO.File]::WriteAllText($StderrPath, '')
if ($env:FAKE_ROW_MUTATE_RUNTIME -eq '1') {
    [System.IO.File]::AppendAllText((Join-Path $TaskWorkingDirectory 'ggml-cuda.dll'), 'mutation')
}
[pscustomobject]@{
    Status = 'completed'; ExitCode = 0; PeakUsedVramMiB = 1
    MinimumFreeVramMiB = 99999; MinimumFreeRamMiB = 99999
    GpuIndex = if ($env:FAKE_ROW_WRONG_GPU -eq '1') { $GpuIndex + 1 } else { $GpuIndex }
    GpuUuid = if ($env:FAKE_ROW_EMPTY_UUID -eq '1') { '' } else { 'GPU-TEST' }
    RuntimeSeconds = 0.01
    Lease = (Join-Path $TaskWorkingDirectory 'nonexistent-test-lease.lock')
}
'@
    $fakeGuardPath = Join-Path $scripts 'Invoke-ExclusiveGpuTask.ps1'
    [System.IO.File]::WriteAllText($fakeGuardPath, $fakeGuard)

    $executable = Join-Path $bin 'test-qwen38-row-invariance.exe'
    [System.IO.File]::WriteAllText($executable, 'fake diagnostic executable')
    $dllNames = @(
        'cublas64_13.dll', 'cublasLt64_13.dll', 'ggml-base.dll', 'ggml-cpu.dll',
        'ggml-cuda.dll', 'ggml.dll', 'llama-common.dll', 'llama-server-impl.dll',
        'llama.dll', 'mtmd.dll', 'cudart64_13.dll'
    )
    foreach ($name in $dllNames) {
        [System.IO.File]::WriteAllText((Join-Path $bin $name), "fake runtime $name")
    }

    $wrapperText = Get-Content -LiteralPath $wrapper -Raw
    $wrapperText = Set-Anchor $wrapperText 'TrustedExecutableSha256' (Get-Sha256 $executable)
    $wrapperText = Set-Anchor $wrapperText 'TrustedGuardSha256' (Get-Sha256 $fakeGuardPath)
    $wrapperText = Set-Anchor $wrapperText 'TrustedCudartSha256' (Get-Sha256 (Join-Path $bin 'cudart64_13.dll'))
    foreach ($name in $dllNames | Where-Object { $_ -cne 'cudart64_13.dll' }) {
        $escapedName = [regex]::Escape("'$name'")
        $pattern = "(?m)^(\s*$escapedName\s*=\s*)'[0-9a-f]{64}'$"
        $replacement = "`$1'$(Get-Sha256 (Join-Path $bin $name))'"
        $updated = [regex]::Replace($wrapperText, $pattern, $replacement)
        Assert-True ($updated -cne $wrapperText) "runtime anchor $name was not replaced"
        $wrapperText = $updated
    }
    [System.IO.File]::WriteAllText($wrapper, $wrapperText)

    foreach ($name in $contaminants.Keys) {
        $prior[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $contaminants[$name], 'Process')
    }

    & $wrapper -Executable $executable -OutputDirectory $output -Mode head-candidate
    $receipt = Read-Receipt $output
    Assert-True ($receipt.stage -ceq 'completed') 'success stage was not completed'
    Assert-True ($receipt.guard_status -ceq 'completed') 'success guard was not completed'
    Assert-True ($receipt.diagnostic_validation -ceq 'passed') 'success diagnostic did not validate'
    Assert-True (@($receipt.runtime.PSObject.Properties).Count -eq 11) 'runtime receipt did not bind 11 modules'
    Assert-True ([string]::IsNullOrWhiteSpace([string] $receipt.wrapper_sha256) -eq $false) 'wrapper SHA missing'
    foreach ($name in $contaminants.Keys) {
        Assert-True ([Environment]::GetEnvironmentVariable($name, 'Process') -ceq $contaminants[$name]) "environment $name was not restored"
    }

    & $wrapper -Executable $executable -OutputDirectory $output -Mode projection-candidate
    $projectionReceipt = Read-Receipt $output 'projection-candidate'
    Assert-True ($projectionReceipt.stage -ceq 'completed') 'projection success stage was not completed'
    Assert-True ($projectionReceipt.guard_status -ceq 'completed') 'projection guard was not completed'
    Assert-True ($projectionReceipt.diagnostic_validation -ceq 'passed') 'projection diagnostic did not validate'

    & $wrapper -Executable $executable -OutputDirectory $output -Mode fp8
    $fp8Receipt = Read-Receipt $output 'fp8'
    Assert-True ($fp8Receipt.stage -ceq 'completed' -and $fp8Receipt.guard_status -ceq 'completed' -and
            $fp8Receipt.diagnostic_validation -ceq 'passed') 'FP8-only diagnostic did not validate'

    & $wrapper -Executable $executable -OutputDirectory $output -Mode gdn
    $gdnReceipt = Read-Receipt $output 'gdn'
    Assert-True ($gdnReceipt.stage -ceq 'completed' -and $gdnReceipt.guard_status -ceq 'completed' -and
            $gdnReceipt.diagnostic_validation -ceq 'passed') 'GDN-only diagnostic did not validate'

    foreach ($operatorMode in @('bf16-projections', 'bf16-candidate', 'rms', 'ssm-conv', 'l2', 'gated-norm', 'fattn-vec-pb1')) {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode $operatorMode
        $operatorReceipt = Read-Receipt $output $operatorMode
        Assert-True ($operatorReceipt.stage -ceq 'completed' -and
                $operatorReceipt.guard_status -ceq 'completed' -and
                $operatorReceipt.diagnostic_validation -ceq 'passed') `
            "$operatorMode selective diagnostic did not validate"
    }

    $env:FAKE_ROW_BAD_FATTN_PB1 = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode fattn-vec-pb1 -Force
        throw 'invalid FATTN PB1 launch geometry unexpectedly passed'
    } catch {
        Assert-True ($_.Exception.Message -match 'FATTN PB1 route/launch geometry') `
            'invalid FATTN PB1 launch geometry did not fail semantic validation'
    } finally {
        Remove-Item Env:FAKE_ROW_BAD_FATTN_PB1 -ErrorAction SilentlyContinue
    }
    $fattnReceipt = Read-Receipt $output 'fattn-vec-pb1'
    Assert-True ($fattnReceipt.guard_status -ceq 'completed' -and
            $fattnReceipt.diagnostic_validation -ceq 'failed') `
        'invalid FATTN PB1 receipt statuses are invalid'

    $env:FAKE_ROW_BAD_BF16_PROBE = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode bf16-projections -Force
        throw 'contradictory BF16 route probe unexpectedly passed'
    } catch {
        Assert-True ($_.Exception.Message -match 'BF16 MMVF-versus-cuBLAS route probe') `
            'contradictory BF16 route probe did not fail semantic validation'
    } finally {
        Remove-Item Env:FAKE_ROW_BAD_BF16_PROBE -ErrorAction SilentlyContinue
    }
    $bf16Receipt = Read-Receipt $output 'bf16-projections'
    Assert-True ($bf16Receipt.guard_status -ceq 'completed' -and $bf16Receipt.diagnostic_validation -ceq 'failed') `
        'bad BF16 route probe receipt statuses are invalid'

    $env:FAKE_ROW_BAD_BF16_CANDIDATE_PROBE = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode bf16-candidate -Force
        throw 'contradictory BF16 candidate route probe unexpectedly passed'
    } catch {
        Assert-True ($_.Exception.Message -match 'candidate route probe') `
            'contradictory BF16 candidate route probe did not fail semantic validation'
    } finally {
        Remove-Item Env:FAKE_ROW_BAD_BF16_CANDIDATE_PROBE -ErrorAction SilentlyContinue
    }
    $bf16CandidateReceipt = Read-Receipt $output 'bf16-candidate'
    Assert-True ($bf16CandidateReceipt.guard_status -ceq 'completed' -and $bf16CandidateReceipt.diagnostic_validation -ceq 'failed') `
        'bad BF16 candidate route probe receipt statuses are invalid'

    $env:FAKE_ROW_EXTRA_OPERATOR_ROUTE = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode ssm-conv -Force
        throw 'unrelated route unexpectedly passed the SSM_CONV locator'
    } catch {
        Assert-True ($_.Exception.Message -match 'isolated-route proof') `
            'unrelated route did not invalidate SSM_CONV isolation evidence'
    } finally {
        Remove-Item Env:FAKE_ROW_EXTRA_OPERATOR_ROUTE -ErrorAction SilentlyContinue
    }
    $ssmReceipt = Read-Receipt $output 'ssm-conv'
    Assert-True ($ssmReceipt.guard_status -ceq 'completed' -and $ssmReceipt.diagnostic_validation -ceq 'failed') `
        'contaminated SSM_CONV receipt statuses are invalid'

    $env:FAKE_ROW_MISSING_NORM_ZERO = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode rms -Force
        throw 'missing zero-valued norm route field unexpectedly passed'
    } catch {
        Assert-True ($_.Exception.Message -match 'width-route schema') `
            'missing zero-valued norm route field did not fail the exact schema'
    } finally {
        Remove-Item Env:FAKE_ROW_MISSING_NORM_ZERO -ErrorAction SilentlyContinue
    }
    $rmsReceipt = Read-Receipt $output 'rms'
    Assert-True ($rmsReceipt.guard_status -ceq 'completed' -and $rmsReceipt.diagnostic_validation -ceq 'failed') `
        'missing norm field receipt statuses are invalid'

    $env:FAKE_ROW_BAD_FP8_CHAIN = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode fp8 -Force
        throw 'inconsistent FP8 route chain unexpectedly passed'
    } catch {
        Assert-True ($_.Exception.Message -match 'FP8 width/route/finite/bitwise') 'inconsistent FP8 evidence did not fail semantic validation'
    } finally {
        Remove-Item Env:FAKE_ROW_BAD_FP8_CHAIN -ErrorAction SilentlyContinue
    }
    $fp8Receipt = Read-Receipt $output 'fp8'
    Assert-True ($fp8Receipt.guard_status -ceq 'completed' -and $fp8Receipt.diagnostic_validation -ceq 'failed') `
        'bad FP8 evidence receipt statuses are invalid'

    $env:FAKE_ROW_BAD_GDN_ROUTE = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode gdn -Force
        throw 'missing GDN fused route unexpectedly passed'
    } catch {
        Assert-True ($_.Exception.Message -match 'GDN route/finite/bitwise/semantic|isolated-route proof') 'missing GDN route did not fail semantic validation'
    } finally {
        Remove-Item Env:FAKE_ROW_BAD_GDN_ROUTE -ErrorAction SilentlyContinue
    }
    $gdnReceipt = Read-Receipt $output 'gdn'
    Assert-True ($gdnReceipt.guard_status -ceq 'completed' -and $gdnReceipt.diagnostic_validation -ceq 'failed') `
        'bad GDN evidence receipt statuses are invalid'

    $env:FAKE_ROW_GDN_FALLBACK = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode gdn -Force
        throw 'ordinary GDN fallback unexpectedly passed the fused-cache locator'
    } catch {
        Assert-True ($_.Exception.Message -match 'GDN route/finite/bitwise/semantic|isolated-route proof') `
            'ordinary GDN fallback did not invalidate fused-cache evidence'
    } finally {
        Remove-Item Env:FAKE_ROW_GDN_FALLBACK -ErrorAction SilentlyContinue
    }
    $gdnReceipt = Read-Receipt $output 'gdn'
    Assert-True ($gdnReceipt.guard_status -ceq 'completed' -and $gdnReceipt.diagnostic_validation -ceq 'failed') `
        'GDN fallback receipt statuses are invalid'

    foreach ($guardIdentityFailure in @('FAKE_ROW_WRONG_GPU', 'FAKE_ROW_EMPTY_UUID')) {
        Set-Item -LiteralPath "Env:$guardIdentityFailure" -Value '1'
        try {
            & $wrapper -Executable $executable -OutputDirectory $output -Mode gdn -Force
            throw "$guardIdentityFailure unexpectedly passed"
        } catch {
            Assert-True ($_.Exception.Message -match 'wrong GPU index or an empty GPU UUID') `
                "$guardIdentityFailure did not fail the guard identity contract"
        } finally {
            Remove-Item -LiteralPath "Env:$guardIdentityFailure" -ErrorAction SilentlyContinue
        }
        $identityReceipt = Read-Receipt $output 'gdn'
        Assert-True ($identityReceipt.guard_status -ceq 'completed' -and
                $identityReceipt.diagnostic_validation -ceq 'not_run') `
            "$guardIdentityFailure receipt statuses are invalid"
    }

    $env:FAKE_ROW_EXTRA_NVFP4_ROUTE = '1'
    foreach ($isolatedMode in @('fp8', 'gdn')) {
        try {
            & $wrapper -Executable $executable -OutputDirectory $output -Mode $isolatedMode -Force
            throw "extra NVFP4 route unexpectedly passed in $isolatedMode mode"
        } catch {
            Assert-True ($_.Exception.Message -match 'proof is invalid') `
                "extra NVFP4 route did not invalidate $isolatedMode isolation evidence"
        }
        $isolatedReceipt = Read-Receipt $output $isolatedMode
        Assert-True ($isolatedReceipt.guard_status -ceq 'completed' -and
                $isolatedReceipt.diagnostic_validation -ceq 'failed') `
            "extra-route $isolatedMode receipt statuses are invalid"
    }
    Remove-Item Env:FAKE_ROW_EXTRA_NVFP4_ROUTE -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_ROW_GDN_FALLBACK -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_ROW_WRONG_GPU -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_ROW_EMPTY_UUID -ErrorAction SilentlyContinue

    $env:FAKE_ROW_BAD_CHAIN = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode head-candidate -Force
        throw 'inconsistent route chain unexpectedly passed'
    } catch {
        Assert-True ($_.Exception.Message -match 'Tagged LM-head route') 'inconsistent route chain did not fail semantic validation'
    } finally {
        Remove-Item Env:FAKE_ROW_BAD_CHAIN -ErrorAction SilentlyContinue
    }
    $receipt = Read-Receipt $output
    Assert-True ($receipt.guard_status -ceq 'completed' -and $receipt.diagnostic_validation -ceq 'failed') 'route-chain failure statuses are invalid'

    $env:FAKE_ROW_GUARD_FAILURE = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode head-candidate -Force
        throw 'synthetic guard failure unexpectedly passed'
    } catch {
        Assert-True ($_.Exception.Message -match 'synthetic guarded failure') 'wrong guarded failure propagated'
    } finally {
        Remove-Item Env:FAKE_ROW_GUARD_FAILURE -ErrorAction SilentlyContinue
    }
    $receipt = Read-Receipt $output
    Assert-True ($receipt.guard_started -eq $true -and $receipt.guard_status -ceq 'failed') 'guard failure status is invalid'
    Assert-True ($receipt.postflight.relevant_processes -eq 0 -and $receipt.postflight.remaining_gpu_leases -eq 0) 'guard failure postflight is invalid'

    $env:FAKE_ROW_MUTATE_RUNTIME = '1'
    try {
        & $wrapper -Executable $executable -OutputDirectory $output -Mode head-candidate -Force
        throw 'runtime mutation unexpectedly passed'
    } catch {
        Assert-True ($_.Exception.Message -match 'runtime identity changed before or during the guarded run') 'runtime mutation did not fail closed'
    } finally {
        Remove-Item Env:FAKE_ROW_MUTATE_RUNTIME -ErrorAction SilentlyContinue
    }
    $receipt = Read-Receipt $output
    Assert-True ($receipt.stage -ceq 'runtime_recheck') 'runtime mutation stage is invalid'
    Assert-True ($receipt.guard_status -ceq 'completed' -and $receipt.diagnostic_validation -ceq 'not_run') 'runtime mutation statuses are invalid'

    'Qwen3.8 row-invariance wrapper CPU fake-guard tests passed.'
} finally {
    Remove-Item Env:FAKE_ROW_BAD_FP8_CHAIN -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_ROW_BAD_GDN_ROUTE -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_ROW_BAD_BF16_PROBE -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_ROW_EXTRA_OPERATOR_ROUTE -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_ROW_MISSING_NORM_ZERO -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_ROW_EXTRA_NVFP4_ROUTE -ErrorAction SilentlyContinue
    foreach ($name in $contaminants.Keys) {
        if ($null -eq $prior[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name, $prior[$name], 'Process')
        }
    }
    if (Test-Path -LiteralPath $testRoot) {
        $resolvedTemp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\')
        $resolvedTest = [System.IO.Path]::GetFullPath($testRoot)
        if ($resolvedTest.StartsWith($resolvedTemp + '\', [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedTest -Recurse -Force
        }
    }
}
