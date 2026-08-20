[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Assert-True([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw "ASSERTION FAILED: $Message" }
}

function Write-Json([string] $Path, $Value) {
    [System.IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 20))
}

function Get-Sha256([string] $Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-StringSha256([string] $Value) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Value)
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Restore-ProcessEnvironment([string] $Name, $Value) {
    if ($null -eq $Value -or [string]::IsNullOrEmpty([string] $Value)) {
        Remove-Item -LiteralPath ("Env:" + $Name) -ErrorAction SilentlyContinue
    } else {
        [System.Environment]::SetEnvironmentVariable($Name, [string] $Value, 'Process')
    }
}

$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$sourceWrapper = Join-Path $workspaceRoot 'scripts\Invoke-Qwen38ParityDiagnostic.ps1'
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("qwen38-parity-wrapper-" + [guid]::NewGuid().ToString('N'))
$scripts = Join-Path $testRoot 'scripts'
$arm = Join-Path $testRoot 'qualification-row-invariant-20260820\p1-final-v5\arms\target-only'
New-Item -ItemType Directory -Path $scripts, $arm -Force | Out-Null
$wrapper = Join-Path $scripts 'Invoke-Qwen38ParityDiagnostic.ps1'
Copy-Item -LiteralPath $sourceWrapper -Destination $wrapper

$fakeGuard = @'
param(
    [string] $Executable, [string[]] $TaskArguments, [string] $TaskWorkingDirectory,
    [int] $GpuIndex, [int] $MinFreeRamGiB, [int] $MinFreeVramMiB,
    [int] $MaxUsedVramMiB, [int] $MaxRuntimeSeconds,
    [string] $StdoutPath, [string] $StderrPath
)
if ($env:LLAMA_ARG_SPEC_TYPE) {
    throw 'wrapper leaked a server-scoped LLAMA_ARG_* value into the guarded child'
}
if ($env:LLAMA_TRACE -or $env:GGML_CUDA_DISABLE_GRAPHS -or $env:CUDA_LAUNCH_BLOCKING) {
    throw 'wrapper leaked LLAMA/GGML/CUDA runtime contamination into the guarded child'
}
if ($env:FAKE_PARITY_GUARD_FAILURE -eq '1') {
    throw 'synthetic guarded child failure'
}
$candidate = [string] $env:LLAMACPP_QWEN38_PARITY_DIAGNOSTIC_CANDIDATE
$projectionEnabled = $env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS -ceq '1'
$bf16RecurrentEnabled = $env:LLAMA_QWEN35_BF16_ROW_INVARIANT_RECURRENT_PROJECTIONS -ceq '1'
$fattnVecEnabled = $env:LLAMA_QWEN35_FATTN_D256_Q8_GQA6_VEC_QCOLS3_8 -ceq '1'
$layerBoundary = $env:LLAMACPP_QWEN38_PARITY_LAYER_BOUNDARY_BISECTION -ceq '1'
$internalRefine = $env:LLAMACPP_QWEN38_PARITY_RECURRENT_INTERNAL_REFINE -ceq '1'
$fullAttentionInternalRefine = $env:LLAMACPP_QWEN38_PARITY_FULL_ATTENTION_INTERNAL_REFINE -ceq '1'
if (($candidate -ceq 'QWEN35_NVFP4_FFN_PROJECTIONS_V1' -or
        $candidate -ceq 'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PROJECTIONS_V2' -or
        $candidate -ceq 'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PLUS_FATTN_VEC_V3') -ne $projectionEnabled -or
        ($candidate -ceq 'QWEN35_BF16_RECURRENT_PROJECTIONS_V1' -or
         $candidate -ceq 'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PROJECTIONS_V2' -or
         $candidate -ceq 'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PLUS_FATTN_VEC_V3') -ne $bf16RecurrentEnabled -or
        ($candidate -ceq 'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PLUS_FATTN_VEC_V3') -ne $fattnVecEnabled) {
    throw 'candidate identity and graph opt-ins were not bound together'
}
$config = [ordered]@{
    type = 'config'; exclusive_gpu_guard = $true; repeat_stability_required = $true
    reference_sampler = 'production_cpu_chain'; temperature = 0; seed = 20260819
    ignore_eos = $true; backend_sampling = $false; eog_biases = 5
    server_warmup_replayed = $true; server_threadpools_attached = $true
    full_reference_trajectory_required = $true
    prediction_start = [int] $env:LLAMACPP_QWEN38_PARITY_PREDICTION_START
    prediction_count = [int] $env:LLAMACPP_QWEN38_PARITY_PREDICTION_COUNT
    rollback_start = [int] $env:LLAMACPP_QWEN38_PARITY_ROLLBACK_START
    prompt_tokens_accepted_per_pass = 736; generated_tokens_accepted_per_pass = 256
    executable_sha256 = $env:LLAMACPP_QWEN38_PARITY_EXECUTABLE_SHA256
    model_sha256 = $env:LLAMACPP_QWEN38_PARITY_MODEL_SHA256
    reference_sha256 = $env:LLAMACPP_QWEN38_PARITY_REFERENCE_SHA256
    manifest_sha256 = $env:LLAMACPP_QWEN38_PARITY_MANIFEST_SHA256
    command_sha256 = $env:LLAMACPP_QWEN38_PARITY_COMMAND_SHA256
    receipt_sha256 = $env:LLAMACPP_QWEN38_PARITY_RECEIPT_SHA256
    reference_runtime_bundle_match = [bool]::Parse($env:LLAMACPP_QWEN38_PARITY_RUNTIME_BUNDLE_MATCH)
    runtime_bundle_count = [int] $env:LLAMACPP_QWEN38_PARITY_BUNDLE_COUNT
    current_bundle_sha256 = $env:LLAMACPP_QWEN38_PARITY_CURRENT_BUNDLE_SHA256
    reference_bundle_sha256 = $env:LLAMACPP_QWEN38_PARITY_REFERENCE_BUNDLE_SHA256
    current_server_sha256 = $env:LLAMACPP_QWEN38_PARITY_CURRENT_SERVER_SHA256
    reference_server_sha256 = $env:LLAMACPP_QWEN38_PARITY_REFERENCE_SERVER_SHA256
    current_llama_sha256 = $env:LLAMACPP_QWEN38_PARITY_CURRENT_LLAMA_SHA256
    reference_llama_sha256 = $env:LLAMACPP_QWEN38_PARITY_REFERENCE_LLAMA_SHA256
    current_common_sha256 = $env:LLAMACPP_QWEN38_PARITY_CURRENT_COMMON_SHA256
    reference_common_sha256 = $env:LLAMACPP_QWEN38_PARITY_REFERENCE_COMMON_SHA256
    diagnostic_candidate = $candidate
    qwen35_nvfp4_ffn_projections = $projectionEnabled
    qwen35_bf16_recurrent_projections = $bf16RecurrentEnabled
    qwen35_fattn_d256_q8_gqa6_vec_qcols3_8 = $fattnVecEnabled
    qwen35_nvfp4_head_hint_compiled = $true
    layer_boundary_bisection = $layerBoundary
    layer_boundary_width = if ($layerBoundary) { 8 } else { 0 }
    recurrent_internal_refine = $internalRefine
    recurrent_internal_layer = if ($internalRefine) { 0 } else { -1 }
    full_attention_internal_refine = $fullAttentionInternalRefine
    full_attention_internal_layer = if ($fullAttentionInternalRefine) { 3 } else { -1 }
    layer_type_map_attested = $layerBoundary
    recurrent_layer_map_hash = if ($layerBoundary) { 'a01852462c981ca5' } else { '0000000000000000' }
    recurrent_layer_map_source = if ($layerBoundary) { 'qwen35.attention.recurrent_layers' } else { '' }
    recurrent_layers = if ($layerBoundary) { 48 } else { 0 }
    full_attention_layers = if ($layerBoundary) { 16 } else { 0 }
}
if ($env:FAKE_PARITY_MALFORMED -eq '1') {
    [System.IO.File]::WriteAllLines($StdoutPath, @('{bad json'))
} else {
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add(($config | ConvertTo-Json -Compress))
    $lines.Add('{"type":"reference_validation","first_prediction":0,"last_prediction":255,"predictions_checked":256,"prompt_tokens_accepted_per_pass":736,"generated_tokens_accepted_per_pass":256,"repeat_stable":true}')
    if ($layerBoundary) {
        $badLayer = if ($fullAttentionInternalRefine) { 3 } else { 0 }
        $badLayerType = if ($fullAttentionInternalRefine) { 'full_attention' } else { 'recurrent' }
        foreach ($layer in 0..63) {
            $mismatch = if ($layer -eq $badLayer) { 1 } else { 0 }
            $first = if ($mismatch) { 176 } else { -1 }
            $lines.Add((@{
                type = 'layer_block'; layer = $layer
                layer_type = if ((($layer + 1) % 4) -eq 0) { 'full_attention' } else { 'recurrent' }
                boundary = 'block_output'; rows = 8; row_bytes = 20480
                first_mismatch_prediction = $first; mismatch_count = $mismatch
                scalar_window_hash = '1111111111111111'; batch_window_hash = if ($mismatch) { '2222222222222222' } else { '1111111111111111' }
                finite = $true; exact_row_comparison = $true; repeat_stable = $true
            } | ConvertTo-Json -Compress))
        }
        $lines.Add('{"type":"observer_invariance","stage":"coarse","scalar_top2_match_unobserved":true,"batch_top2_match_unobserved":true,"scalar_partial_state_digest_match_unobserved":true,"batch_partial_state_digest_match_unobserved":true,"repeat_stable":true}')
        foreach ($boundary in @('block_input', 'attention_output', 'attention_residual', 'ffn_output', 'block_output')) {
            $mismatch = if ($boundary -ceq 'attention_output') { 1 } else { 0 }
            if ($env:FAKE_PARITY_BAD_LAYER_BOUNDARY -eq '1' -and $boundary -ceq 'block_input') { $mismatch = 9 }
            $lines.Add((@{
                type = 'layer_boundary'; layer = $badLayer; layer_type = $badLayerType; boundary = $boundary
                rows = 8; row_bytes = 20480
                first_mismatch_prediction = if ($mismatch -gt 0) { 176 } else { -1 }
                mismatch_count = $mismatch; scalar_window_hash = '3333333333333333'
                batch_window_hash = if ($mismatch) { '4444444444444444' } else { '3333333333333333' }
                finite = $true; exact_row_comparison = $true; repeat_stable = $true
            } | ConvertTo-Json -Compress))
        }
        if (-not $fullAttentionInternalRefine) {
            $lines.Add('{"type":"recurrent_state_boundary","layer":0,"boundary":"state_predelta","scalar_steps":8,"batch_prefix_states":1,"state_bytes":65536,"prefix_exact_match":true,"scalar_evolution_hash":"5555555555555555","batch_prefix_hash":"6666666666666666","finite":true,"repeat_stable":true}')
        }
        $lines.Add('{"type":"observer_invariance","stage":"refine","scalar_top2_match_unobserved":true,"batch_top2_match_unobserved":true,"scalar_partial_state_digest_match_unobserved":true,"batch_partial_state_digest_match_unobserved":true,"repeat_stable":true}')
        if ($internalRefine) {
            $internalSpecs = @(
                @('attn_norm',1,5120,1), @('linear_attn_qkv_mixed',1,10240,1), @('z',1,6144,1),
                @('beta',2,1,48), @('beta_sigmoid',2,1,48), @('alpha',1,48,1),
                @('gate',1,48,1), @('conv_output_silu',1,10240,1),
                @('q_conv_predelta',2,128,16), @('k_conv_predelta',2,128,16),
                @('v_conv_predelta',2,128,48), @('gdn_core_output',2,128,48),
                @('final_output',1,6144,1), @('linear_attn_out',1,5120,1)
            )
            for ($index = 0; $index -lt $internalSpecs.Count; ++$index) {
                $spec = $internalSpecs[$index]; $rank = [int] $spec[1]
                $scalarShape = if ($rank -eq 1) { @([int]$spec[2],1,1,1) } else { @([int]$spec[2],[int]$spec[3],1,1) }
                $batchShape = if ($rank -eq 1) { @([int]$spec[2],8,1,1) } else { @([int]$spec[2],[int]$spec[3],8,1) }
                $mismatch = if ($index -ge 1) { 8 } else { 0 }
                $lines.Add((@{
                    type='recurrent_internal_boundary'; layer=0; boundary=[string]$spec[0]
                    source_sequence_index=$index; rows=8
                    row_bytes=([int64]$spec[2]*[int64]$spec[3]*4)
                    scalar_shape=$scalarShape; batch_shape=$batchShape
                    first_mismatch_prediction=if($mismatch){176}else{-1}; mismatch_count=$mismatch
                    scalar_window_hash='1010101010101010'
                    batch_window_hash=if($mismatch){'2020202020202020'}else{'1010101010101010'}
                    finite=$true; exact_token_slice_comparison=$true; repeat_stable=$true
                } | ConvertTo-Json -Compress))
            }
            foreach ($cache in @('conv_states','state_predelta')) {
                $shape = if ($cache -ceq 'conv_states') { @(30720,1,1,1) } else { @(128,128,48,1) }
                $bytes = if ($cache -ceq 'conv_states') { 122880 } else { 3145728 }
                $lines.Add((@{
                    type='recurrent_internal_cache'; layer=0; boundary=$cache
                    scope='cache_input_prefix_and_scalar_evolution'; scalar_steps=8; batch_prefix_states=1
                    state_bytes=$bytes; prefix_exact_match=$true; scalar_shape=$shape; batch_shape=$shape
                    scalar_evolution_hash='3030303030303030'; batch_prefix_hash='4040404040404040'
                    finite=$true; repeat_stable=$true
                } | ConvertTo-Json -Compress))
            }
            $lines.Add('{"type":"observer_invariance","stage":"recurrent_internal","scalar_top2_match_unobserved":true,"batch_top2_match_unobserved":true,"scalar_partial_state_digest_match_unobserved":true,"batch_partial_state_digest_match_unobserved":true,"repeat_stable":true}')
            $batchFusedRoutes = if ($env:FAKE_PARITY_BAD_INTERNAL -eq '1') { 95 } else { 96 }
            $lines.Add((@{
                type='recurrent_internal_summary'; layer=0; rows=8; boundaries=14; cache_boundaries=2
                first_bad_observed_boundary='linear_attn_qkv_mixed'; first_bad_prediction=176
                boundary_order_semantic_not_causal=$true; scalar_gdn=0; scalar_gdn_fused_cache=768
                batch_gdn=0; batch_gdn_fused_cache=$batchFusedRoutes; one_context_at_a_time=$true
                content_free=$true; observer_top2_and_state_digest_invariant=$true
            } | ConvertTo-Json -Compress))
        }
        if ($fullAttentionInternalRefine) {
            $fullSpecs = @(
                @('attn_norm',1,5120,1), @('qg_projection',1,12288,1),
                @('q_pre_norm',2,256,24), @('q_post_norm',2,256,24),
                @('k_projection',1,1024,1), @('k_post_norm',2,256,4),
                @('v_projection',1,1024,1), @('gate_pre_sigmoid',1,6144,1),
                @('q_post_norm_rope',2,256,24), @('k_post_norm_rope',2,256,4),
                @('v_reshaped',2,256,4), @('kv_fa_output_pregate',1,6144,1),
                @('attn_gated',1,6144,1), @('output_projection',1,5120,1)
            )
            for ($index = 0; $index -lt $fullSpecs.Count; ++$index) {
                $spec = $fullSpecs[$index]; $rank = [int] $spec[1]
                $scalarShape = if ($rank -eq 1) { @([int]$spec[2],1,1,1) } else { @([int]$spec[2],[int]$spec[3],1,1) }
                $batchShape = if ($rank -eq 1) { @([int]$spec[2],8,1,1) } else { @([int]$spec[2],[int]$spec[3],8,1) }
                $mismatch = if ($index -ge 1) { 8 } else { 0 }
                $rowBytes = [int64]$spec[2]*[int64]$spec[3]*4
                if ($env:FAKE_PARITY_BAD_FULL_BOUNDARY -eq '1' -and $index -eq 1) { $rowBytes -= 4 }
                $lines.Add((@{
                    type='full_attention_internal_boundary'; layer=3; boundary=[string]$spec[0]
                    source_sequence_index=$index; rows=8; row_bytes=$rowBytes
                    scalar_shape=$scalarShape; batch_shape=$batchShape
                    first_mismatch_prediction=if($mismatch){176}else{-1}; mismatch_count=$mismatch
                    scalar_window_hash='5050505050505050'
                    batch_window_hash=if($mismatch){'6060606060606060'}else{'5050505050505050'}
                    finite=$true; exact_token_slice_comparison=$true; repeat_stable=$true
                } | ConvertTo-Json -Compress))
            }
            $lines.Add((@{
                type='observer_invariance'; stage='full_attention_internal'
                scalar_top2_match_unobserved=$true; batch_top2_match_unobserved=$true
                scalar_partial_state_digest_match_unobserved=$true; batch_partial_state_digest_match_unobserved=$true
                scalar_full_state_digest_match_unobserved=$true; batch_full_state_digest_match_unobserved=$true
                route_snapshot_matches_unobserved=($env:FAKE_PARITY_BAD_FULL_OBSERVER -ne '1')
                repeat_stable=$true
            } | ConvertTo-Json -Compress))
            $badBatchMma = if ($env:FAKE_PARITY_BAD_FULL_INTERNAL -eq '1') { 15 } else { 16 }
            $lines.Add((@{
                type='full_attention_internal_summary'; layer=3; rows=8; boundaries=14
                first_bad_observed_boundary='qg_projection'; first_bad_prediction=176
                boundary_order_semantic_not_causal=$true
                scalar_fattn_vec=128; scalar_fattn_mma_f16=0; scalar_fattn_tile=0
                batch_fattn_vec=0; batch_fattn_mma_f16=$badBatchMma; batch_fattn_tile=0
                scalar_sigmoid=0; scalar_sigmoid_mul=128; batch_sigmoid=0; batch_sigmoid_mul=16
                scalar_set_rows=256; batch_set_rows=32
                scalar_rope_view_set_rows=0; batch_rope_view_set_rows=0
                route_snapshot_matches_unobserved=$true; full_state_digest_matches_unobserved=$true
                one_context_at_a_time=($env:FAKE_PARITY_BAD_FULL_CLAIM -ne '1'); content_free=$true
            } | ConvertTo-Json -Compress))
        }
        $lines.Add('{"type":"recurrent_partial_state","scope":"recurrent_only","prefix_hash":"7777777777777777","prefix_bytes":100,"scalar_evolution_hash":"8888888888888888","scalar_final_hash":"9999999999999999","batch_final_hash":"aaaaaaaaaaaaaaaa","final_bytes":100,"prefix_match":true,"final_match":false,"repeat_stable":true}')
        $lines.Add((@{type='layer_boundary_summary';window_start=176;window_width=8;layers=64;recurrent_layers=48;full_attention_layers=16;recurrent_layer_map_hash='a01852462c981ca5';recurrent_layer_map_source='qwen35.attention.recurrent_layers';layer_type_map_attested=$true;first_bad_layer=$badLayer;first_bad_prediction=176;refine_executed=$true;first_bad_boundary='attention_output';first_boundary_mismatch_prediction=176;one_context_at_a_time=$true;content_free=$true;observer_top2_and_state_digest_invariant=$true} | ConvertTo-Json -Compress))
    } else {
    $fakeWidths = if ($fattnVecEnabled) { @(1,2,3,4,5,6,7,8) } else { @(1,2,4,8) }
    foreach ($width in $fakeWidths) {
        foreach ($prediction in 176..199) {
            $lines.Add((@{ type = 'width'; width = $width; prediction_index = $prediction } | ConvertTo-Json -Compress))
        }
        $widthStateHash = if($env:FAKE_PARITY_BAD_FATTN_FULL_STATE -eq '1' -and $width -eq 5){'aaaaaaaaaaaaaaaa'}else{'2222222222222222'}
        $widthStateBytes = if($env:FAKE_PARITY_BAD_FATTN_FULL_STATE_BYTES -eq '1' -and $width -eq 5){201}else{200}
        $lines.Add((@{ type = 'width_state'; width = $width; consumed_token_index=198
            prefix_state_hash='1111111111111111';prefix_state_bytes=100
            state_hash=$widthStateHash;state_bytes=$widthStateBytes;repeat_stable=$true
            state_scope=if($fattnVecEnabled){'full'}else{'partial'} } | ConvertTo-Json -Compress))
        if ($fattnVecEnabled) {
            $selected = if ($width -ge 3) { 16 } else { 0 }
            if ($env:FAKE_PARITY_BAD_FATTN_ROUTE -eq '1' -and $width -eq 8) { $selected = 0 }
            $generalVec = 16
            if ($env:FAKE_PARITY_BAD_FATTN_DEFAULT_ROUTE -eq '1' -and $width -eq 2) { $generalVec = 0 }
            $hintSeen = if ($width -ge 3) { 16 } else { 0 }
            if ($env:FAKE_PARITY_BAD_FATTN_HINT -eq '1' -and $width -eq 3) { $hintSeen = 0 }
            $lines.Add((@{
                type='fattn_graph_hint_route'; width=$width; hint_seen=$hintSeen
                hint_expected=($width -ge 3); first_single_batch_probe=$true; content_free=$true
            } | ConvertTo-Json -Compress))
            $nkv = 1024
            $predicateFailMask = if ($width -ge 3) {'0000000000000000'} else {'0000000000000044'}
            if ($env:FAKE_PARITY_BAD_FATTN_PREDICATE -eq '1' -and $width -eq 3) {$predicateFailMask='0000000000001000'}
            $lines.Add((@{
                type='fattn_candidate_predicate';width=$width;fail_mask=$predicateFailMask
                device_cc=1200;compiled_arch=1200;src4_null=$true;hint=if($width -ge 3){8}else{0}
                op=if($env:FAKE_PARITY_BAD_FATTN_PREDICATE_FIELDS -eq '1' -and $width -eq 3){41}else{42}
                q_type=0;k_type=8;v_type=8;mask_type=1;dst_type=0;prec=10
                scale_bits=1031798784;max_bias_bits=0;softcap_bits=0
                mask_contiguous=$true;dst_contiguous=$true
                q_ne=@(256,$width,24,1);q_nb=@(4,24576,1024,(24576*$width))
                k_ne=@(256,$nkv,4,1);k_nb=@(34,1088,272,(1088*2048))
                v_ne=@(256,$nkv,4,1);v_nb=@(34,1088,272,(1088*2048))
                mask_ne=@($nkv,$width,1,1);mask_nb=@(2,(2*$nkv),(2*$nkv*$width),(2*$nkv*$width))
                dst_ne=@(256,24,$width,1);dst_nb=@(4,1024,24576,(24576*$width))
                evaluated=$true;content_free=$true
            } | ConvertTo-Json -Compress -Depth 5))
            $lines.Add((@{
                type='fattn_candidate_route'; width=$width; candidate_vec=$selected
                fattn_vec=if($width -ge 3){$selected}else{$generalVec}; fattn_mma_f16=0; fattn_tile=0
                candidate_selected=($width -ge 3); first_single_batch_probe=$true
                shape_guard_attested=$true; content_free=$true
            } | ConvertTo-Json -Compress))
        }
    }
    $fakeSummaryWidths = if ($fattnVecEnabled) { @(2,3,4,5,6,7,8) } else { @(2,4,8) }
    foreach ($width in $fakeSummaryWidths) {
        $firstDifference = if ($env:FAKE_PARITY_BAD_FATTN_EXACT -eq '1' -and $width -eq 5) { 181 } else { -1 }
        $lines.Add((@{ type = 'width_summary'; width = $width
            first_top1_difference=$firstDifference; first_top2_difference=$firstDifference
            first_top2_value_difference=$firstDifference; prefix_state_match_scalar=$true
            final_state_match_scalar=$true; state_scope=if($fattnVecEnabled){'full'}else{'partial'} } | ConvertTo-Json -Compress))
    }
    foreach ($prediction in 184..191) {
        $lines.Add((@{ type = 'rollback_batch'; width = 8; prediction_index = $prediction } | ConvertTo-Json -Compress))
    }
    $lines.Add((@{type='rollback_batch_summary';width=8;repeat_stable=$true
        row_top2_ids_match_scalar=$true;row_top2_values_match_scalar=$true;final_state_match_scalar=$true
        final_state_hash='3333333333333333';oracle_final_state_hash='3333333333333333'
        final_state_bytes=300;oracle_final_state_bytes=300
        intermediate_snapshot_generation_vs_selection_separable=$false
        state_scope=if($fattnVecEnabled){'full'}else{'partial'}} | ConvertTo-Json -Compress))
    foreach ($rollback in 1..7) {
        $committed = 190 - $rollback
        $selectedHash = if($env:FAKE_PARITY_BAD_FATTN_ROLLBACK_STATE -eq '1' -and $rollback -eq 2){'bbbbbbbbbbbbbbbb'}else{'4444444444444444'}
        $lines.Add((@{ type = 'rollback'; width = 8; rollback = $rollback; repeat_stable = $true
            committed_token_index=$committed;continuation_token_index=($committed+1)
            continuation_prediction_index=($committed+2);continued_top1=17;oracle_top1=17
            continued_margin=0.25;oracle_margin=0.25
            selected_state_hash=$selectedHash;oracle_selected_state_hash='4444444444444444'
            continued_state_hash='5555555555555555';oracle_continued_state_hash='5555555555555555'
            state_bytes=400;oracle_selected_state_bytes=400;oracle_continued_state_bytes=400
            selected_state_match=$true;continued_state_match=$true;continued_top2_match=$true
            continued_top2_values_match=$true;snapshot_generation_vs_selection_separable=$false
            state_scope=if($fattnVecEnabled){'full'}else{'partial'} } | ConvertTo-Json -Compress))
    }
    }
    $lines.Add('{"type":"complete","operational_success":true,"exact_parity_gate_bypassed":false}')
    [System.IO.File]::WriteAllLines($StdoutPath, $lines)
}
[System.IO.File]::WriteAllText($StderrPath, '')
if ($env:FAKE_PARITY_MUTATE_RUNTIME -eq '1') {
    [System.IO.File]::AppendAllText($Executable, 'changed during guarded execution')
}
if ($env:FAKE_PARITY_DELETE_GUARD -eq '1') {
    Remove-Item -LiteralPath $PSCommandPath -Force
}
[pscustomobject]@{
    Status = 'completed'; ExitCode = 0; PeakUsedVramMiB = 1; MinimumFreeVramMiB = 9999
    MinimumFreeRamMiB = 99999; GpuIndex = 0
    GpuUuid = if ($env:FAKE_PARITY_WRONG_GPU -eq '1') { 'GPU-OTHER' } else { 'GPU-TEST' }
    RuntimeSeconds = 0.01
    Lease = (Join-Path $TaskWorkingDirectory 'nonexistent-test-lease.lock')
}
'@
$fakeGuardPath = Join-Path $scripts 'Invoke-ExclusiveGpuTask.ps1'
[System.IO.File]::WriteAllText($fakeGuardPath, $fakeGuard)
$fakeGuardSha256 = Get-Sha256 $fakeGuardPath

$executable = Join-Path $testRoot 'test-qwen38-recurrent-parity.exe'
$model = Join-Path $testRoot 'target.gguf'
$llamaDll = Join-Path $testRoot 'llama.dll'
$commonDll = Join-Path $testRoot 'llama-common.dll'
$server = Join-Path $testRoot 'llama-server.exe'
[System.IO.File]::WriteAllText($executable, 'fake diagnostic executable')
[System.IO.File]::WriteAllText($model, 'fake target model')
[System.IO.File]::WriteAllText($llamaDll, 'fake llama runtime')
[System.IO.File]::WriteAllText($commonDll, 'fake common runtime')
[System.IO.File]::WriteAllText($server, 'fake reference server')

$tokens = @(0..255)
$fixtureIdentity = ('1' * 64)
$fixtureTokenTrace = ('2' * 64)
$waves = @()
for ($i = 0; $i -lt 7; ++$i) {
    $waves += [ordered]@{
        wave = $i; parallel = 1
        clients = @([ordered]@{
            seed = 20260819; generated_tokens = 256; stop_type = 'limit';
            truncated = $false; token_sha256 = $fixtureTokenTrace; tokens = $tokens
        })
        metric_delta = [ordered]@{
            prompt_tokens = 736; cached_prompt_tokens = 0; tokens = 256;
            draft_tokens = 0; draft_steps = 0
        }
    }
}
$wavesPath = Join-Path $arm 'waves.json'
Write-Json $wavesPath ([ordered]@{
    schema = 'ai-loader-qwen38-dspark-qualification-arm/v1'; waves = $waves
})

$arguments = @(
    'fake-llama-server.exe', '--model', $model, '--ctx-size', '2048', '--parallel', '1',
    '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--flash-attn', 'on',
    '--batch-size', '2048', '--ubatch-size', '128', '--n-gpu-layers', 'all',
    '--device', 'CUDA0', '--split-mode', 'none', '--fit', 'off', '--cache-ram', '0',
    '--ctx-checkpoints', '0', '--reasoning', 'off', '--no-cache-idle-slots',
    '--no-cache-prompt', '--no-webui', '--metrics', '--kv-unified'
)
$commandPath = Join-Path $arm 'command.json'
Write-Json $commandPath ([ordered]@{
    schema = 'ai-loader-qwen38-dspark-qualification-arm/v1'; arm = 'target-only';
    tier = 'P1'; dynamic_rs = $false; arguments = $arguments
})
$receiptPath = Join-Path $arm 'receipt.json'
$summaryPath = Join-Path $arm 'summary.json'
[System.IO.File]::WriteAllText($summaryPath, 'receipt-bound summary')
Write-Json $receiptPath ([ordered]@{
    schema = 'ai-loader-qwen38-dspark-qualification-arm/v1'; arm = 'target-only';
    tier = 'P1'; dynamic_rs = $false
    artifacts = [ordered]@{
        command = [ordered]@{ path = 'command.json'; sha256 = Get-Sha256 $commandPath }
        waves = [ordered]@{ path = 'waves.json'; sha256 = Get-Sha256 $wavesPath }
        summary = [ordered]@{ path = 'summary.json'; sha256 = Get-Sha256 $summaryPath }
    }
})
$manifestPath = Join-Path (Split-Path (Split-Path $arm -Parent) -Parent) 'manifest.json'
Write-Json $manifestPath ([ordered]@{
    schema = 'ai-loader-qwen38-dspark-qualification-manifest/v1'
    identity = $fixtureIdentity
    components = [ordered]@{
        schema = 'ai-loader-qwen38-dspark-qualification-identity/v1'; tier = 'P1';
        server_sha256 = Get-Sha256 $server
        target_model_sha256 = Get-Sha256 $model
        gpu = [ordered]@{ uuid = 'GPU-TEST' }
        server_environment = [ordered]@{ cuda_visible_devices = 'GPU-TEST' }
        arm_arguments = @([ordered]@{ name = 'target-only'; arguments = $arguments })
        runtime_bundle = @(
            [ordered]@{ name = 'llama.dll'; sha256 = Get-Sha256 $llamaDll },
            [ordered]@{ name = 'llama-common.dll'; sha256 = Get-Sha256 $commonDll }
        )
    }
})

$qualificationRoot = Split-Path (Split-Path $arm -Parent) -Parent
$fixtureRuntimeBundleSha = ('3' * 64)
$fixtureProjectionBundleSha = Get-StringSha256 ((@(
    "llama-common.dll:$(Get-Sha256 $commonDll)",
    "llama.dll:$(Get-Sha256 $llamaDll)"
) | Sort-Object) -join "`n")
$validationPath = Join-Path $qualificationRoot 'validation.json'
Write-Json $validationPath ([ordered]@{
    schema = 'ai-loader-qwen38-target-trace-validation/v1'; valid = $true
    identity = $fixtureIdentity; qualification_identity = $fixtureIdentity
    manifest_sha256 = Get-Sha256 $manifestPath; model_sha256 = Get-Sha256 $model
    waves_sha256 = Get-Sha256 $wavesPath; token_trace_sha256 = $fixtureTokenTrace
    command_sha256 = Get-Sha256 $commandPath; receipt_sha256 = Get-Sha256 $receiptPath
    runtime_bundle_sha256 = $fixtureRuntimeBundleSha; runtime_bundle_count = 2
})
$traceGuardReceiptPath = Join-Path $qualificationRoot 'guard-receipt.json'
Write-Json $traceGuardReceiptPath ([ordered]@{
    schema = 'ai-loader-qwen38-target-trace-guard-receipt/v1'; stage = 'completed'
    guard_status = 'completed'; trace_validation = 'passed'; guard_started = $true; error = $null
    guard_sha256 = Get-Sha256 (Join-Path $scripts 'Invoke-ExclusiveGpuTask.ps1')
    server_sha256 = Get-Sha256 $server; model_sha256 = Get-Sha256 $model
    manifest_sha256 = Get-Sha256 $manifestPath; validation_sha256 = Get-Sha256 $validationPath
    guard = [ordered]@{ Status = 'completed'; ExitCode = 0; GpuIndex = 0; GpuUuid = 'GPU-TEST' }
    postflight = [ordered]@{ relevant_processes = 0; port_18136_free = $true; recursive_gpu_leases = 0 }
})

# The production wrapper has immutable approved-run constants. This isolated
# fixture substitutes internally consistent anchors in its private copied script.
$wrapperText = Get-Content -LiteralPath $wrapper -Raw
$wrapperText = $wrapperText.Replace(
    'aab1f6698569cd26ef92cf4b6d99ac996397af06cc3c169e1ecd5ad3fe23f639', $fixtureIdentity)
$wrapperText = $wrapperText.Replace(
    '67da6391d2240a058562973def2949721fd9c336308d996f5e3b0de73f2405c0', (Get-Sha256 $manifestPath))
$wrapperText = $wrapperText.Replace(
    'fe38d4bc2702e2a5129235927f550e0269e482c121df3aeda90fbc91c0fc0cbb', (Get-Sha256 $validationPath))
$wrapperText = $wrapperText.Replace(
    '667728f1bcd38da34830f3d416924c446ec72d9f607e6f9779b1e626b0e74492', (Get-Sha256 $traceGuardReceiptPath))
$wrapperText = $wrapperText.Replace(
    '46e0ec4bdab3907346fbb486b1e4f78b49a66b245374aa0d3d628a8d099c0336', (Get-Sha256 $model))
$wrapperText = $wrapperText.Replace(
    '1e5f7b997354cebfded587fdbfc6339737c63fc025760a7f75b224dbada5f348', (Get-Sha256 $wavesPath))
$wrapperText = $wrapperText.Replace(
    '80d896aed5f33f8fb90964caeadaf1f0accb72f464c4acfe9bae642f2b10200f', $fixtureTokenTrace)
$wrapperText = $wrapperText.Replace(
    '640418e80973413b3f7b95334a8ecf0aa639100e05339d27b5c4ec9ac0704284', (Get-Sha256 $commandPath))
$wrapperText = $wrapperText.Replace(
    'c9c1fd02566478c8eb53a75c72ba5ac671ccd1111057d8c6891c75503569a9b9', (Get-Sha256 $receiptPath))
$wrapperText = $wrapperText.Replace(
    '429fec7fbc3ed32cb498433d6da760a4888fb286ebd09c4b4b32540aa4bd4c1c', $fixtureRuntimeBundleSha)
$wrapperText = $wrapperText.Replace('$TrustedRuntimeBundleCount = 10', '$TrustedRuntimeBundleCount = 2')
$wrapperText = $wrapperText.Replace(
    'bc639a16a17da220d6ca83b7956f7db8a86e6c714ada254a246544abe98ee3f2', (Get-Sha256 $executable))
$wrapperText = $wrapperText.Replace(
    '8dd01510919d8af3c00abc3fa9cd4df5db8adb4da7f9add5d7704fc21152d007', (Get-Sha256 $executable))
$wrapperText = $wrapperText.Replace(
    '3957122672ce2def719a58b321e70334ec048e0b4d9f435c11b1b5d2d61f5b9d', (Get-Sha256 $server))
$wrapperText = $wrapperText.Replace(
    '05ba7aa482de89760a7208cf96b706c9ce0e6dcb4dd486ee58209e7ad0552241', $fixtureProjectionBundleSha)
$projectionRuntime = @"
`$TrustedProjectionRuntime = [ordered]@{
    'llama.dll' = '$(Get-Sha256 $llamaDll)'
    'llama-common.dll' = '$(Get-Sha256 $commonDll)'
}
"@
$wrapperText = [regex]::Replace(
    $wrapperText,
    '(?ms)^\$TrustedProjectionRuntime = \[ordered\]@\{.*?^\}',
    $projectionRuntime.TrimEnd())
$wrapperText = $wrapperText.Replace(
    '3120e9fe7c247250a440389d0450c71741718b391d435a270fd2f16d33dc1463',
    (Get-Sha256 (Join-Path $scripts 'Invoke-ExclusiveGpuTask.ps1')))
[System.IO.File]::WriteAllText($wrapper, $wrapperText)
$priorSpecType = $env:LLAMA_ARG_SPEC_TYPE
$priorLlamaTrace = $env:LLAMA_TRACE
$priorGgmlGraphs = $env:GGML_CUDA_DISABLE_GRAPHS
$priorCudaBlocking = $env:CUDA_LAUNCH_BLOCKING
$priorProjectionOptIn = $env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS
$priorBf16RecurrentOptIn = $env:LLAMA_QWEN35_BF16_ROW_INVARIANT_RECURRENT_PROJECTIONS
$priorFattnVecOptIn = $env:LLAMA_QWEN35_FATTN_D256_Q8_GQA6_VEC_QCOLS3_8
$priorLayerBoundary = $env:LLAMACPP_QWEN38_PARITY_LAYER_BOUNDARY_BISECTION
$priorInternalRefine = $env:LLAMACPP_QWEN38_PARITY_RECURRENT_INTERNAL_REFINE
$priorFullAttentionInternalRefine = $env:LLAMACPP_QWEN38_PARITY_FULL_ATTENTION_INTERNAL_REFINE

try {
    $env:LLAMA_ARG_SPEC_TYPE = 'draft-mtp'
    $env:LLAMA_TRACE = '1'
    $env:GGML_CUDA_DISABLE_GRAPHS = '1'
    $env:CUDA_LAUNCH_BLOCKING = '1'
    $env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS = '1'
    $env:LLAMA_QWEN35_BF16_ROW_INVARIANT_RECURRENT_PROJECTIONS = '1'
    $env:LLAMA_QWEN35_FATTN_D256_Q8_GQA6_VEC_QCOLS3_8 = '1'
    $env:LLAMACPP_QWEN38_PARITY_LAYER_BOUNDARY_BISECTION = '1'
    $env:LLAMACPP_QWEN38_PARITY_RECURRENT_INTERNAL_REFINE = '1'
    $env:LLAMACPP_QWEN38_PARITY_FULL_ATTENTION_INTERNAL_REFINE = '1'
    $output = Join-Path $testRoot 'diagnostic.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath -OutputPath $output
    Assert-True $? 'valid fixture should pass through the fake exclusive guard'
    Assert-True (Test-Path -LiteralPath $output) 'wrapper should preserve diagnostic JSONL'
    $guardReceipt = Get-Content -LiteralPath "$output.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($guardReceipt.schema -ceq 'ai-loader-qwen38-parity-guard-receipt/v1') `
        'wrapper must persist the guard receipt schema'
    Assert-True ([int] $guardReceipt.postflight.relevant_processes -eq 0) `
        'guard receipt must prove zero relevant postflight processes'
    Assert-True ([bool] $guardReceipt.postflight.port_18136_free) `
        'guard receipt must prove port 18136 is free'
    Assert-True ([int] $guardReceipt.postflight.recursive_gpu_leases -eq 0) `
        'guard receipt must prove all recursive GPU leases are absent'
    Assert-True ($guardReceipt.guard_status -ceq 'completed' -and
            $guardReceipt.diagnostic_validation -ceq 'passed') `
        'guard receipt must distinguish guard completion from diagnostic validation'
    Assert-True ($env:LLAMA_ARG_SPEC_TYPE -eq 'draft-mtp') 'wrapper must restore the caller LLAMA_ARG_* environment'
    Assert-True ($env:LLAMA_TRACE -eq '1' -and $env:GGML_CUDA_DISABLE_GRAPHS -eq '1' -and
            $env:CUDA_LAUNCH_BLOCKING -eq '1') 'wrapper must restore the caller runtime environment'
    Assert-True ($env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS -eq '1') `
        'default wrapper must scrub the projection opt-in inside the guard and restore it afterward'
    Assert-True ($env:LLAMA_QWEN35_BF16_ROW_INVARIANT_RECURRENT_PROJECTIONS -eq '1') `
        'default wrapper must scrub the BF16 recurrent opt-in inside the guard and restore it afterward'
    Assert-True ($env:LLAMA_QWEN35_FATTN_D256_Q8_GQA6_VEC_QCOLS3_8 -eq '1') `
        'default wrapper must scrub the full-attention opt-in inside the guard and restore it afterward'
    Assert-True ($env:LLAMACPP_QWEN38_PARITY_LAYER_BOUNDARY_BISECTION -eq '1') `
        'default wrapper must scrub the layer-boundary marker inside the guard and restore it afterward'
    Assert-True ($env:LLAMACPP_QWEN38_PARITY_RECURRENT_INTERNAL_REFINE -eq '1') `
        'default wrapper must scrub the internal-refine marker inside the guard and restore it afterward'
    Assert-True ($env:LLAMACPP_QWEN38_PARITY_FULL_ATTENTION_INTERNAL_REFINE -eq '1') `
        'default wrapper must scrub the full-attention internal marker inside the guard and restore it afterward'

    $projectionOutput = Join-Path $testRoot 'projection-diagnostic.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
        -OutputPath $projectionOutput -Qwen35Nvfp4FfnProjectionDiagnostic
    Assert-True $? 'valid projection fixture should pass through the fake exclusive guard'
    $projectionGuardReceipt = Get-Content -LiteralPath "$projectionOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($projectionGuardReceipt.guard_status -ceq 'completed' -and
            $projectionGuardReceipt.diagnostic_validation -ceq 'passed' -and
            $projectionGuardReceipt.diagnostic_candidate -ceq 'QWEN35_NVFP4_FFN_PROJECTIONS_V1') `
        'projection wrapper must bind, validate, and persist the opt-in candidate identity'
    Assert-True (-not (Test-Path Env:LLAMACPP_QWEN38_PARITY_DIAGNOSTIC_CANDIDATE) -and
            $env:LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS -eq '1') `
        'projection wrapper must remove its marker and restore the caller projection environment afterward'

    $bf16Output = Join-Path $testRoot 'bf16-recurrent.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
        -OutputPath $bf16Output -Qwen35Bf16RecurrentProjectionDiagnostic
    $bf16GuardReceipt = Get-Content -LiteralPath "$bf16Output.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($bf16GuardReceipt.guard_status -ceq 'completed' -and
            $bf16GuardReceipt.diagnostic_validation -ceq 'passed' -and
            $bf16GuardReceipt.diagnostic_candidate -ceq 'QWEN35_BF16_RECURRENT_PROJECTIONS_V1' -and
            [bool] $bf16GuardReceipt.qwen35_nvfp4_head_hint_compiled -and
            -not [bool] $bf16GuardReceipt.qwen35_nvfp4_ffn_projections -and
            [bool] $bf16GuardReceipt.qwen35_bf16_recurrent_projections) `
        'BF16 recurrent wrapper must bind and persist exact candidate plus compiled head baseline identity'
    Assert-True (-not (Test-Path Env:LLAMACPP_QWEN38_PARITY_DIAGNOSTIC_CANDIDATE) -and
            $env:LLAMA_QWEN35_BF16_ROW_INVARIANT_RECURRENT_PROJECTIONS -eq '1') `
        'BF16 recurrent wrapper must remove its marker and restore the caller opt-in afterward'

    $fattnOutput = Join-Path $testRoot 'fattn-vec-v3.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
        -OutputPath $fattnOutput `
        -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
        -Qwen35FullAttentionVecDiagnostic
    Assert-True $? 'explicit full-attention V3 fixture should pass through the fake exclusive guard'
    $fattnReceipt = Get-Content -LiteralPath "$fattnOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($fattnReceipt.guard_status -ceq 'completed' -and
            $fattnReceipt.diagnostic_validation -ceq 'passed' -and
            $fattnReceipt.diagnostic_candidate -ceq `
                'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PLUS_FATTN_VEC_V3' -and
            [bool] $fattnReceipt.qwen35_nvfp4_ffn_projections -and
            [bool] $fattnReceipt.qwen35_bf16_recurrent_projections -and
            [bool] $fattnReceipt.qwen35_fattn_d256_q8_gqa6_vec_qcols3_8) `
        'full-attention V3 wrapper must attest all three explicit candidate switches'

    $singleFattnRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'fattn-missing-prerequisites.jsonl') `
            -Qwen35FullAttentionVecDiagnostic
    } catch { $singleFattnRejected = $_.Exception.Message -like '*requires both existing candidate switches*' }
    Assert-True $singleFattnRejected 'full-attention selector must reject an implicit candidate combination before guard'

    $env:FAKE_PARITY_BAD_FATTN_ROUTE = '1'
    $badFattnOutput = Join-Path $testRoot 'fattn-bad-route.jsonl'
    $badFattnRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $badFattnOutput `
            -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
            -Qwen35FullAttentionVecDiagnostic
    } catch { $badFattnRejected = $_.Exception.Message -like '*Invalid full-attention candidate route proof*' }
    Assert-True $badFattnRejected 'wrapper must reject a missing width-8 candidate VEC route'
    $badFattnReceipt = Get-Content -LiteralPath "$badFattnOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($badFattnReceipt.guard_status -ceq 'completed' -and
            $badFattnReceipt.diagnostic_validation -ceq 'failed') `
        'bad full-attention route evidence must retain guard completion and fail validation'
    Remove-Item Env:FAKE_PARITY_BAD_FATTN_ROUTE

    $env:FAKE_PARITY_BAD_FATTN_DEFAULT_ROUTE = '1'
    $badDefaultRouteOutput = Join-Path $testRoot 'fattn-bad-default-route.jsonl'
    $badDefaultRouteRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $badDefaultRouteOutput `
            -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
            -Qwen35FullAttentionVecDiagnostic
    } catch { $badDefaultRouteRejected = $_.Exception.Message -like '*Invalid full-attention candidate route proof*' }
    Assert-True $badDefaultRouteRejected 'wrapper must reject a missing width-2 default VEC route'
    $badDefaultRouteReceipt = Get-Content -LiteralPath "$badDefaultRouteOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($badDefaultRouteReceipt.guard_status -ceq 'completed' -and
            $badDefaultRouteReceipt.diagnostic_validation -ceq 'failed') `
        'bad default-route evidence must retain guard completion and fail validation'
    Remove-Item Env:FAKE_PARITY_BAD_FATTN_DEFAULT_ROUTE

    $env:FAKE_PARITY_BAD_FATTN_HINT = '1'
    $badHintOutput = Join-Path $testRoot 'fattn-bad-graph-hint.jsonl'
    $badHintRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $badHintOutput `
            -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
            -Qwen35FullAttentionVecDiagnostic
    } catch { $badHintRejected = $_.Exception.Message -like '*Invalid full-attention graph-hint proof*' }
    Assert-True $badHintRejected 'wrapper must reject a missing width-3 graph hint separately from selector counters'
    $badHintReceipt = Get-Content -LiteralPath "$badHintOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($badHintReceipt.guard_status -ceq 'completed' -and
            $badHintReceipt.diagnostic_validation -ceq 'failed') `
        'bad graph-hint evidence must retain guard completion and fail validation'
    Remove-Item Env:FAKE_PARITY_BAD_FATTN_HINT

    $env:FAKE_PARITY_BAD_FATTN_PREDICATE = '1'
    $badPredicateOutput = Join-Path $testRoot 'fattn-bad-cuda-predicate.jsonl'
    $badPredicateRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $badPredicateOutput `
            -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
            -Qwen35FullAttentionVecDiagnostic
    } catch { $badPredicateRejected = $_.Exception.Message -like '*Invalid full-attention CUDA predicate observation*' }
    Assert-True $badPredicateRejected 'wrapper must reject a nonzero width-3 CUDA predicate fail mask'
    $badPredicateReceipt = Get-Content -LiteralPath "$badPredicateOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($badPredicateReceipt.guard_status -ceq 'completed' -and
            $badPredicateReceipt.diagnostic_validation -ceq 'failed') `
        'bad CUDA predicate evidence must retain guard completion and fail validation'
    Remove-Item Env:FAKE_PARITY_BAD_FATTN_PREDICATE

    $env:FAKE_PARITY_BAD_FATTN_PREDICATE_FIELDS = '1'
    $badPredicateFieldOutput = Join-Path $testRoot 'fattn-bad-cuda-predicate-fields.jsonl'
    $badPredicateFieldRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $badPredicateFieldOutput `
            -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
            -Qwen35FullAttentionVecDiagnostic
    } catch { $badPredicateFieldRejected = $_.Exception.Message -like '*Invalid full-attention CUDA predicate observation*' }
    Assert-True $badPredicateFieldRejected 'wrapper must recompute the CUDA predicate contract instead of trusting its fail mask'
    $badPredicateFieldReceipt = Get-Content -LiteralPath "$badPredicateFieldOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($badPredicateFieldReceipt.guard_status -ceq 'completed' -and
            $badPredicateFieldReceipt.diagnostic_validation -ceq 'failed') `
        'inconsistent CUDA predicate fields must retain guard completion and fail validation'
    Remove-Item Env:FAKE_PARITY_BAD_FATTN_PREDICATE_FIELDS

    $env:FAKE_PARITY_BAD_FATTN_EXACT = '1'
    $badFattnExactOutput = Join-Path $testRoot 'fattn-bad-exact.jsonl'
    $badFattnExactRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $badFattnExactOutput `
            -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
            -Qwen35FullAttentionVecDiagnostic
    } catch { $badFattnExactRejected = $_.Exception.Message -like '*failed exact logits or full-state width parity*' }
    Assert-True $badFattnExactRejected 'wrapper must reject a V3 width/logit mismatch'
    $badFattnExactReceipt = Get-Content -LiteralPath "$badFattnExactOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($badFattnExactReceipt.guard_status -ceq 'completed' -and
            $badFattnExactReceipt.diagnostic_validation -ceq 'failed') `
        'bad V3 exact-parity evidence must retain guard completion and fail validation'
    Remove-Item Env:FAKE_PARITY_BAD_FATTN_EXACT

    foreach ($badStateCase in @(
            @{ Env='FAKE_PARITY_BAD_FATTN_FULL_STATE'; Name='fattn-bad-full-state'; Pattern='*invalid or non-matching full width-state record*' },
            @{ Env='FAKE_PARITY_BAD_FATTN_FULL_STATE_BYTES'; Name='fattn-bad-full-state-bytes'; Pattern='*invalid or non-matching full width-state record*' },
            @{ Env='FAKE_PARITY_BAD_FATTN_ROLLBACK_STATE'; Name='fattn-bad-rollback-state'; Pattern='*rollback schema is invalid*' })) {
        Set-Item -LiteralPath ("Env:" + $badStateCase.Env) -Value '1'
        $badStateOutput = Join-Path $testRoot ($badStateCase.Name + '.jsonl')
        $badStateRejected = $false
        try {
            & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
                -OutputPath $badStateOutput `
                -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
                -Qwen35FullAttentionVecDiagnostic
        } catch { $badStateRejected = $_.Exception.Message -like $badStateCase.Pattern }
        Assert-True $badStateRejected ("wrapper must reject " + $badStateCase.Name)
        $badStateReceipt = Get-Content -LiteralPath "$badStateOutput.guard-receipt.json" -Raw | ConvertFrom-Json
        Assert-True ($badStateReceipt.guard_status -ceq 'completed' -and
                $badStateReceipt.diagnostic_validation -ceq 'failed') `
            'bad V3 state evidence must retain guard completion and fail validation'
        Remove-Item -LiteralPath ("Env:" + $badStateCase.Env)
    }

    $layerOutput = Join-Path $testRoot 'layer-boundary.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
        -OutputPath $layerOutput -PredictionCount 8 `
        -Qwen35Nvfp4FfnProjectionDiagnostic -LayerBoundaryBisection
    Assert-True $? 'valid layer-boundary fixture should pass through the fake exclusive guard'
    $layerGuardReceipt = Get-Content -LiteralPath "$layerOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($layerGuardReceipt.guard_status -ceq 'completed' -and
            $layerGuardReceipt.diagnostic_validation -ceq 'passed' -and
            [bool] $layerGuardReceipt.layer_boundary_bisection) `
        'layer-boundary wrapper must bind, validate, and persist the diagnostic mode'
    Assert-True ($env:LLAMACPP_QWEN38_PARITY_LAYER_BOUNDARY_BISECTION -eq '1') `
        'layer-boundary wrapper must restore the caller marker after the guarded child'

    $internalOutput = Join-Path $testRoot 'recurrent-internal.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
        -OutputPath $internalOutput -PredictionCount 8 `
        -Qwen35Nvfp4FfnProjectionDiagnostic -LayerBoundaryBisection -RecurrentLayerInternalRefine
    Assert-True $? 'valid recurrent-internal fixture should pass through the fake exclusive guard'
    $internalReceipt = Get-Content -LiteralPath "$internalOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($internalReceipt.guard_status -ceq 'completed' -and
            $internalReceipt.diagnostic_validation -ceq 'passed' -and
            [bool] $internalReceipt.recurrent_internal_refine) `
        'recurrent-internal wrapper must bind, validate, and persist the diagnostic mode'
    Assert-True ($env:LLAMACPP_QWEN38_PARITY_RECURRENT_INTERNAL_REFINE -eq '1') `
        'recurrent-internal wrapper must restore the caller marker after the guarded child'

    $bf16LayerOutput = Join-Path $testRoot 'bf16-layer-boundary.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
        -OutputPath $bf16LayerOutput -PredictionCount 8 `
        -Qwen35Bf16RecurrentProjectionDiagnostic -LayerBoundaryBisection
    Assert-True $? 'valid BF16 layer-boundary fixture should pass through the fake exclusive guard'
    $bf16LayerReceipt = Get-Content -LiteralPath "$bf16LayerOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($bf16LayerReceipt.guard_status -ceq 'completed' -and
            $bf16LayerReceipt.diagnostic_validation -ceq 'passed' -and
            $bf16LayerReceipt.diagnostic_candidate -ceq 'QWEN35_BF16_RECURRENT_PROJECTIONS_V1' -and
            -not [bool] $bf16LayerReceipt.qwen35_nvfp4_ffn_projections -and
            [bool] $bf16LayerReceipt.qwen35_bf16_recurrent_projections -and
            [bool] $bf16LayerReceipt.layer_boundary_bisection) `
        'BF16 layer wrapper must bind exactly the BF16 candidate and preserve layer evidence'

    $bf16InternalOutput = Join-Path $testRoot 'bf16-recurrent-internal.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
        -OutputPath $bf16InternalOutput -PredictionCount 8 `
        -Qwen35Bf16RecurrentProjectionDiagnostic -LayerBoundaryBisection -RecurrentLayerInternalRefine
    Assert-True $? 'valid BF16 recurrent-internal fixture should pass through the fake exclusive guard'
    $bf16InternalReceipt = Get-Content -LiteralPath "$bf16InternalOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($bf16InternalReceipt.guard_status -ceq 'completed' -and
            $bf16InternalReceipt.diagnostic_validation -ceq 'passed' -and
            $bf16InternalReceipt.diagnostic_candidate -ceq 'QWEN35_BF16_RECURRENT_PROJECTIONS_V1' -and
            -not [bool] $bf16InternalReceipt.qwen35_nvfp4_ffn_projections -and
            [bool] $bf16InternalReceipt.qwen35_bf16_recurrent_projections -and
            [bool] $bf16InternalReceipt.recurrent_internal_refine) `
        'BF16 internal wrapper must preserve exact candidate, observer, and route contracts'

    $combinedInternalOutput = Join-Path $testRoot 'combined-recurrent-internal.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
        -OutputPath $combinedInternalOutput -PredictionCount 8 `
        -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
        -LayerBoundaryBisection -RecurrentLayerInternalRefine
    Assert-True $? 'explicit combined V2 recurrent-internal fixture should pass through the fake exclusive guard'
    $combinedInternalReceipt = Get-Content -LiteralPath "$combinedInternalOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($combinedInternalReceipt.guard_status -ceq 'completed' -and
            $combinedInternalReceipt.diagnostic_validation -ceq 'passed' -and
            $combinedInternalReceipt.diagnostic_candidate -ceq `
                'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PROJECTIONS_V2' -and
            [bool] $combinedInternalReceipt.qwen35_nvfp4_ffn_projections -and
            [bool] $combinedInternalReceipt.qwen35_bf16_recurrent_projections -and
            [bool] $combinedInternalReceipt.layer_boundary_bisection -and
            [bool] $combinedInternalReceipt.recurrent_internal_refine) `
        'combined layer/internal wrapper must attest both explicit flags and the V2 candidate identity'

    $combinedFullInternalOutput = Join-Path $testRoot 'combined-full-attention-internal.jsonl'
    & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
        -OutputPath $combinedFullInternalOutput -PredictionCount 8 `
        -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
        -LayerBoundaryBisection -FullAttentionInternalRefine
    Assert-True $? 'explicit combined V2 full-attention fixture should pass through the fake exclusive guard'
    $combinedFullReceipt = Get-Content -LiteralPath "$combinedFullInternalOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($combinedFullReceipt.guard_status -ceq 'completed' -and
            $combinedFullReceipt.diagnostic_validation -ceq 'passed' -and
            $combinedFullReceipt.diagnostic_candidate -ceq 'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PROJECTIONS_V2' -and
            [bool] $combinedFullReceipt.qwen35_nvfp4_ffn_projections -and
            [bool] $combinedFullReceipt.qwen35_bf16_recurrent_projections -and
            [bool] $combinedFullReceipt.layer_boundary_bisection -and
            [bool] $combinedFullReceipt.full_attention_internal_refine) `
        'combined full-attention wrapper must attest both candidates and the internal mode'
    Assert-True ($env:LLAMACPP_QWEN38_PARITY_FULL_ATTENTION_INTERNAL_REFINE -eq '1') `
        'full-attention wrapper must restore the caller marker after the guarded child'

    $mutualInternalRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'mutual-internal.jsonl') -PredictionCount 8 `
            -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
            -LayerBoundaryBisection `
            -RecurrentLayerInternalRefine -FullAttentionInternalRefine
    } catch { $mutualInternalRejected = $_.Exception.Message -like '*mutually exclusive*' }
    Assert-True $mutualInternalRejected 'wrapper must reject combined recurrent and full-attention internal modes'

    $singleCandidateFullRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'single-candidate-full.jsonl') -PredictionCount 8 `
            -Qwen35Bf16RecurrentProjectionDiagnostic -LayerBoundaryBisection -FullAttentionInternalRefine
    } catch { $singleCandidateFullRejected = $_.Exception.Message -like '*combined V2 identity*' }
    Assert-True $singleCandidateFullRejected `
        'full-attention internal wrapper must reject a single candidate before the guard'

    $wrongFullWindowOutput = Join-Path $testRoot 'wrong-full-window.jsonl'
    $wrongFullWindowRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $wrongFullWindowOutput -PredictionStart 175 -PredictionCount 8 `
            -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
            -LayerBoundaryBisection -FullAttentionInternalRefine
    } catch { $wrongFullWindowRejected = $_.Exception.Message -like '*pinned to PredictionStart=176*' }
    Assert-True ($wrongFullWindowRejected -and -not (Test-Path -LiteralPath $wrongFullWindowOutput)) `
        'full-attention internal wrapper must reject another window before invoking the guard'

    $env:FAKE_PARITY_BAD_FULL_INTERNAL = '1'
    $badFullOutput = Join-Path $testRoot 'bad-full-attention-internal.jsonl'
    $badFullRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $badFullOutput -PredictionCount 8 `
            -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
            -LayerBoundaryBisection -FullAttentionInternalRefine
    } catch { $badFullRejected = $_.Exception.Message -like '*Full-attention summary*' }
    Assert-True $badFullRejected 'wrapper must reject inconsistent full-attention route evidence'
    $badFullReceipt = Get-Content -LiteralPath "$badFullOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($badFullReceipt.guard_status -ceq 'completed' -and
            $badFullReceipt.diagnostic_validation -ceq 'failed') `
        'invalid full-attention evidence must retain guard completion and fail validation'
    Remove-Item Env:FAKE_PARITY_BAD_FULL_INTERNAL

    $fullNegativeCases = @(
        @{ Env='FAKE_PARITY_BAD_FULL_BOUNDARY'; Slug='boundary'; Error='*row byte count*' },
        @{ Env='FAKE_PARITY_BAD_FULL_OBSERVER'; Slug='observer'; Error='*Observer*' },
        @{ Env='FAKE_PARITY_BAD_FULL_CLAIM'; Slug='claim'; Error='*Full-attention summary*' }
    )
    foreach ($case in $fullNegativeCases) {
        [System.Environment]::SetEnvironmentVariable([string] $case.Env, '1', 'Process')
        $caseOutput = Join-Path $testRoot ("bad-full-" + [string] $case.Slug + '.jsonl')
        $caseRejected = $false
        try {
            & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
                -OutputPath $caseOutput -PredictionCount 8 `
                -Qwen35Nvfp4FfnProjectionDiagnostic -Qwen35Bf16RecurrentProjectionDiagnostic `
                -LayerBoundaryBisection -FullAttentionInternalRefine
        } catch { $caseRejected = $_.Exception.Message -like [string] $case.Error }
        Assert-True $caseRejected "wrapper must reject the full-attention $([string] $case.Slug) mutation"
        $caseReceipt = Get-Content -LiteralPath "$caseOutput.guard-receipt.json" -Raw | ConvertFrom-Json
        Assert-True ($caseReceipt.guard_status -ceq 'completed' -and
                $caseReceipt.diagnostic_validation -ceq 'failed') `
            "full-attention $([string] $case.Slug) mutation must retain guard completion and fail validation"
        [System.Environment]::SetEnvironmentVariable([string] $case.Env, $null, 'Process')
    }

    $missingLayerCandidateRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'missing-layer-candidate.jsonl') -PredictionCount 8 `
            -LayerBoundaryBisection
    } catch {
        $missingLayerCandidateRejected = $_.Exception.Message -like `
            '*LayerBoundaryBisection requires at least one explicit reviewed Qwen35 candidate*'
    }
    Assert-True $missingLayerCandidateRejected `
        'layer-boundary wrapper must reject a run without any explicit reviewed candidate before the guard'

    $env:FAKE_PARITY_BAD_INTERNAL = '1'
    $badInternalOutput = Join-Path $testRoot 'bad-recurrent-internal.jsonl'
    $badInternalRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $badInternalOutput -PredictionCount 8 `
            -Qwen35Nvfp4FfnProjectionDiagnostic -LayerBoundaryBisection -RecurrentLayerInternalRefine
    } catch { $badInternalRejected = $_.Exception.Message -like '*Recurrent-internal summary*' }
    Assert-True $badInternalRejected 'wrapper must reject inconsistent recurrent-internal fused-route evidence'
    $badInternalReceipt = Get-Content -LiteralPath "$badInternalOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($badInternalReceipt.guard_status -ceq 'completed' -and
            $badInternalReceipt.diagnostic_validation -ceq 'failed') `
        'invalid recurrent-internal evidence must retain guard completion and fail validation'
    Remove-Item Env:FAKE_PARITY_BAD_INTERNAL

    $env:FAKE_PARITY_BAD_LAYER_BOUNDARY = '1'
    $badLayerOutput = Join-Path $testRoot 'bad-layer-boundary.jsonl'
    $badLayerRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $badLayerOutput -PredictionCount 8 `
            -Qwen35Nvfp4FfnProjectionDiagnostic -LayerBoundaryBisection
    } catch { $badLayerRejected = $_.Exception.Message -like '*Invalid first-bad-layer boundary evidence*' }
    Assert-True $badLayerRejected 'wrapper must reject inconsistent layer-boundary mismatch evidence'
    $badLayerReceipt = Get-Content -LiteralPath "$badLayerOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($badLayerReceipt.guard_status -ceq 'completed' -and
            $badLayerReceipt.diagnostic_validation -ceq 'failed') `
        'invalid layer evidence must retain guard completion and fail validation'
    Remove-Item Env:FAKE_PARITY_BAD_LAYER_BOUNDARY

    $env:FAKE_PARITY_WRONG_GPU = '1'
    $wrongGpuOutput = Join-Path $testRoot 'wrong-gpu.jsonl'
    $wrongGpuRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $wrongGpuOutput
    } catch { $wrongGpuRejected = $_.Exception.Message -like '*GPU identity does not match*' }
    Assert-True $wrongGpuRejected 'wrapper must reject a diagnostic run on a different GPU UUID'
    $wrongGpuReceipt = Get-Content -LiteralPath "$wrongGpuOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($wrongGpuReceipt.guard_status -ceq 'completed' -and
            $wrongGpuReceipt.diagnostic_validation -ceq 'not_run') `
        'wrong-GPU receipt must preserve guard completion while rejecting diagnostic identity'
    Remove-Item Env:FAKE_PARITY_WRONG_GPU

    $env:FAKE_PARITY_MUTATE_RUNTIME = '1'
    $runtimeMutationOutput = Join-Path $testRoot 'runtime-mutation.jsonl'
    $runtimeMutationRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $runtimeMutationOutput
    } catch { $runtimeMutationRejected = $_.Exception.Message -like '*changed at the guarded launch boundary*' }
    Assert-True $runtimeMutationRejected 'wrapper must reject runtime mutation during guarded execution'
    $runtimeMutationReceipt = Get-Content -LiteralPath "$runtimeMutationOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($runtimeMutationReceipt.guard_status -ceq 'failed' -and
            $runtimeMutationReceipt.diagnostic_validation -ceq 'not_run') `
        'runtime mutation must persist a fail-closed guard receipt'
    [System.IO.File]::WriteAllText($executable, 'fake diagnostic executable')
    Remove-Item Env:FAKE_PARITY_MUTATE_RUNTIME

    $env:FAKE_PARITY_DELETE_GUARD = '1'
    $guardDeletionOutput = Join-Path $testRoot 'guard-deletion.jsonl'
    $guardDeletionRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $guardDeletionOutput
    } catch { $guardDeletionRejected = $_.Exception.Message -like '*changed at the guarded launch boundary*' }
    Assert-True $guardDeletionRejected 'wrapper must reject guard deletion during guarded execution'
    $guardDeletionReceipt = Get-Content -LiteralPath "$guardDeletionOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($guardDeletionReceipt.guard_status -ceq 'failed' -and
            $guardDeletionReceipt.diagnostic_validation -ceq 'not_run' -and
            $guardDeletionReceipt.guard_sha256 -ceq $fakeGuardSha256) `
        'guard deletion must persist a fail-closed receipt using the accepted hash'
    [System.IO.File]::WriteAllText($fakeGuardPath, $fakeGuard)
    Remove-Item Env:FAKE_PARITY_DELETE_GUARD

    $validationOriginal = Get-Content -LiteralPath $validationPath -Raw
    [IO.File]::AppendAllText($validationPath, 'changed')
    $validationMismatchRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'validation-mismatch.jsonl')
    } catch { $validationMismatchRejected = $_.Exception.Message -like '*Validation or trace guard receipt SHA-256*' }
    Assert-True $validationMismatchRejected 'wrapper must reject a changed fresh-trace validation artifact'
    [IO.File]::WriteAllText($validationPath, $validationOriginal)

    $traceGuardOriginal = Get-Content -LiteralPath $traceGuardReceiptPath -Raw
    [IO.File]::AppendAllText($traceGuardReceiptPath, 'changed')
    $traceGuardMismatchRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'trace-guard-mismatch.jsonl')
    } catch { $traceGuardMismatchRejected = $_.Exception.Message -like '*Validation or trace guard receipt SHA-256*' }
    Assert-True $traceGuardMismatchRejected 'wrapper must reject a changed fresh-trace guard receipt'
    [IO.File]::WriteAllText($traceGuardReceiptPath, $traceGuardOriginal)

    $oldQualificationRoot = Join-Path $testRoot 'qualification\p1-final-v5'
    New-Item -ItemType Directory -Path (Split-Path -Parent $oldQualificationRoot) -Force | Out-Null
    Copy-Item -LiteralPath $qualificationRoot -Destination $oldQualificationRoot -Recurse
    $oldWavesPath = Join-Path $oldQualificationRoot 'arms\target-only\waves.json'
    $oldPathRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $oldWavesPath `
            -OutputPath (Join-Path $testRoot 'old-path.jsonl')
    } catch { $oldPathRejected = $_.Exception.Message -like '*qualification-row-invariant-20260820*' }
    Assert-True $oldPathRejected 'wrapper must reject the stale qualification\p1-final-v5 path contract'

    $collisionRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath -OutputPath $model -Force
    } catch { $collisionRejected = $_.Exception.Message -like '*collides*' }
    Assert-True $collisionRejected 'Force must not allow output/model collision'
    Assert-True ((Get-Content -LiteralPath $model -Raw) -eq 'fake target model') 'collision rejection must preserve model'

    $receiptCollisionRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $summaryPath -Force
    } catch { $receiptCollisionRejected = $_.Exception.Message -like '*receipt-bound*' }
    Assert-True $receiptCollisionRejected 'Force must not allow output/receipt-artifact collision'
    Assert-True ((Get-Content -LiteralPath $summaryPath -Raw) -eq 'receipt-bound summary') `
        'receipt collision rejection must preserve evidence'

    [System.IO.File]::AppendAllText($executable, 'changed')
    $executableMismatchRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'executable-mismatch.jsonl')
    } catch { $executableMismatchRejected = $_.Exception.Message -like '*reviewed CUDA build*' }
    Assert-True $executableMismatchRejected 'wrapper must reject an unreviewed diagnostic executable'
    [System.IO.File]::WriteAllText($executable, 'fake diagnostic executable')

    [System.IO.File]::AppendAllText($commonDll, 'changed')
    $bundleMismatchRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'bundle-mismatch.jsonl')
    } catch { $bundleMismatchRejected = $_.Exception.Message -like '*not identical*' }
    Assert-True $bundleMismatchRejected 'wrapper must reject a changed adjacent DLL before the guard'
    [System.IO.File]::WriteAllText($commonDll, 'fake common runtime')

    $unexpectedDll = Join-Path $testRoot 'unexpected.dll'
    [System.IO.File]::WriteAllText($unexpectedDll, 'unexpected runtime')
    $unexpectedDllRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'unexpected-dll.jsonl')
    } catch { $unexpectedDllRejected = $_.Exception.Message -like '*DLL set*' }
    Assert-True $unexpectedDllRejected 'wrapper must reject an unexpected adjacent DLL before the guard'
    Remove-Item -LiteralPath $unexpectedDll

    $env:FAKE_PARITY_GUARD_FAILURE = '1'
    $failureOutput = Join-Path $testRoot 'guard-failure.jsonl'
    $guardFailureRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath $failureOutput
    } catch { $guardFailureRejected = $_.Exception.Message -like '*synthetic guarded child failure*' }
    Assert-True $guardFailureRejected 'wrapper must propagate a guarded child failure'
    $failureReceipt = Get-Content -LiteralPath "$failureOutput.guard-receipt.json" -Raw | ConvertFrom-Json
    Assert-True ($failureReceipt.guard_status -ceq 'failed' -and
            $failureReceipt.diagnostic_validation -ceq 'not_run') `
        'guard failure must persist a status-bearing failure receipt'
    Assert-True ([int] $failureReceipt.postflight.relevant_processes -eq 0 -and
            [int] $failureReceipt.postflight.recursive_gpu_leases -eq 0 -and
            [bool] $failureReceipt.postflight.port_18136_free) `
        'guard failure receipt must persist postflight proof'
    Remove-Item Env:FAKE_PARITY_GUARD_FAILURE

    $env:FAKE_PARITY_MALFORMED = '1'
    $malformedRejected = $false
    try {
        & $wrapper -Executable $executable -ModelPath $model -TargetOnlyWavesPath $wavesPath `
            -OutputPath (Join-Path $testRoot 'malformed.jsonl')
    } catch { $malformedRejected = $_.Exception.Message -like '*not valid JSON*' }
    Assert-True $malformedRejected 'wrapper must reject malformed diagnostic JSONL'
} finally {
    Restore-ProcessEnvironment 'LLAMA_ARG_SPEC_TYPE' $priorSpecType
    Restore-ProcessEnvironment 'LLAMA_TRACE' $priorLlamaTrace
    Restore-ProcessEnvironment 'GGML_CUDA_DISABLE_GRAPHS' $priorGgmlGraphs
    Restore-ProcessEnvironment 'CUDA_LAUNCH_BLOCKING' $priorCudaBlocking
    Restore-ProcessEnvironment 'LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS' $priorProjectionOptIn
    Restore-ProcessEnvironment 'LLAMA_QWEN35_BF16_ROW_INVARIANT_RECURRENT_PROJECTIONS' $priorBf16RecurrentOptIn
    Restore-ProcessEnvironment 'LLAMA_QWEN35_FATTN_D256_Q8_GQA6_VEC_QCOLS3_8' $priorFattnVecOptIn
    Restore-ProcessEnvironment 'LLAMACPP_QWEN38_PARITY_LAYER_BOUNDARY_BISECTION' $priorLayerBoundary
    Restore-ProcessEnvironment 'LLAMACPP_QWEN38_PARITY_RECURRENT_INTERNAL_REFINE' $priorInternalRefine
    Restore-ProcessEnvironment 'LLAMACPP_QWEN38_PARITY_FULL_ATTENTION_INTERNAL_REFINE' $priorFullAttentionInternalRefine
    if ([string]::IsNullOrEmpty([string] $priorSpecType)) {
        Assert-True (-not (Test-Path -LiteralPath Env:LLAMA_ARG_SPEC_TYPE)) `
            'wrapper test cleanup must not leave an empty LLAMA_ARG_SPEC_TYPE entry'
    }
    Remove-Item Env:FAKE_PARITY_MALFORMED -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_PARITY_GUARD_FAILURE -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_PARITY_WRONG_GPU -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_PARITY_MUTATE_RUNTIME -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_PARITY_DELETE_GUARD -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_PARITY_BAD_LAYER_BOUNDARY -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_PARITY_BAD_FULL_INTERNAL -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_PARITY_BAD_FULL_BOUNDARY -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_PARITY_BAD_FULL_OBSERVER -ErrorAction SilentlyContinue
    Remove-Item Env:FAKE_PARITY_BAD_FULL_CLAIM -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
}
