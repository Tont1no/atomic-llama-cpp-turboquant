# Qwen3.8 recurrent parity diagnostic

This is a guarded, model-specific locator for a deterministic difference
between scalar target decoding and speculative target verification. It does not
replace the production qualification's exact-token parity gate and never marks
an arm qualified.

The executable evaluates the same target-only reference trace in contexts with
teacher-forced widths 1, 2, 4, and 8. It records the top two target logits,
their margin, and a 64-bit FNV-1a digest of the serialized partial recurrent
state. It then evaluates one width-8 batch repeatedly, rolls it back by depths
1 through 7, and compares the selected and continued state plus the next top-2
result with a fresh scalar oracle.

This is a controlled target-width and recurrent-rollback locator. It does not
replay the complete DSpark draft proposal, accept/reject history, or server
slot scheduler. A clean locator result does not prove end-to-end DSpark parity;
the production qualification must still prove the complete raw token stream.

Only the wrapper should launch the executable. It binds the input to the exact
reviewed row-invariant `qualification-row-invariant-20260820\p1-final-v5` target-only
manifest, validation, trace guard receipt, trusted qualification identity,
command, arm receipt, wave/trace hashes, target model SHA-256, server
executable, and every adjacent DLL. The adjacent DLL name set and every hash
must match exactly.
A missing, extra, renamed, or changed DLL stops before the GPU lease. It then delegates the
child to `Invoke-ExclusiveGpuTask.ps1`,
which owns the machine-wide GPU lease, RAM/VRAM limits, timeout, and Windows Job
Object cleanup. The binary also requires the guard's lease identity rather than
treating its model-specific environment sentinel as a safety boundary. It
creates and destroys one context at a time and never loads a draft model.
Before invoking the guard, the wrapper temporarily removes every inherited
`LLAMA_*`, `LLAMACPP_QWEN38_PARITY_*`, `GGML_*`, and `CUDA_*` variable and
restores it afterward. The guard
then binds `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` to the selected GPU.
Runtime environment settings cannot silently alter the pinned diagnostic
command.
The diagnostic also uses the no-system-config parser entry point, so global or
user `llama.cpp/config.ini` files are ignored without changing parser behavior
for any other executable.

Every width and rollback cell is executed twice and must reproduce identical
top-2 bit patterns and compact state checksums. Any repeat instability is an
operational failure, not a parity result. The JSONL `config` record includes
the executable, model, waves, manifest, command, and receipt hashes used by the
wrapper.

Before any width comparison, two fresh scalar passes validate all 256 greedy
predictions against the approved trace. Each pass accepts all 736 prompt tokens
and all 256 generated tokens into the production CPU sampler. The exact five
EOG IDs are suppressed for `ignore_eos=true`. A mismatch reports token IDs,
raw top-2 logits, raw margin, sampler selection, and raw top-1 EOG status. It
never emits token text or prompt content. This prevents teacher forcing from
hiding an earlier model or runtime mismatch. Every fresh context attaches the
same common threadpools before the BOS+EOS startup warmup, memory clear,
synchronize, and performance reset used by the server. Guard overrides may
only be stricter than the defaults:
at least 18 GiB free RAM and 4096 MiB free VRAM, and at most 28672 MiB used
VRAM.

The wrapper writes `<OutputPath>.guard-receipt.json`. It binds the output and
stderr hashes, guard memory measurements, runtime identity, zero relevant
processes, an absent guard lease, and a free port 18136 postflight check.

## Reviewed current launch

The fresh P1 target-only trace under `qualification-row-invariant-20260820` completed
under one exclusive guard and passed the fail-closed evidence validator. Its
manifest identity, validation, guard receipt, command, arm receipt, waves,
token trace and ten-file runtime bundle are now immutable wrapper anchors. The
older `qualification\p1-final-v5` trace remains rejected by the path contract.

```powershell
$source = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-performance-final'
$build  = Join-Path $source 'build-dspark-rs-integrated-cuda132-fresh'
$waves  = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\benchmarks\qwen38-dspark-production-qualification-20260819\qualification-row-invariant-20260820\p1-final-v5\arms\target-only\waves.json'
$model  = 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf'

& (Join-Path $source 'scripts\Invoke-Qwen38ParityDiagnostic.ps1') `
  -Executable (Join-Path $build 'bin\test-qwen38-recurrent-parity.exe') `
  -ModelPath $model `
  -TargetOnlyWavesPath $waves `
  -OutputPath (Join-Path $source 'parity-diagnostic-row-invariant-20260820.jsonl')
```

Do not add `-Force` for the first run. It exists only for an intentional retry
after the prior output and receipt have been reviewed.

The opt-in all-NVFP4 projection diagnostic uses the separately pinned
`build-final-cuda132-ninja` executable/server/runtime bundle and adds
`-Qwen35Nvfp4FfnProjectionDiagnostic`. The wrapper then binds both the candidate
identity and `LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS=1` only inside the
exclusive-guard child. The ordinary launch remains unchanged and cannot select
the candidate accidentally.

The narrower BF16 recurrent-projection candidate adds
`-Qwen35Bf16RecurrentProjectionDiagnostic`. It enables only the tagged Qwen35
recurrent alpha/beta BF16 `[5120,48]` matmuls for M2 through M8; M1, untagged
graphs, LoRA graphs, and every other shape retain the established route. The
current compiled baseline also contains the already-reviewed unconditional
NVFP4 LM-head hint, which is attested separately from this BF16 opt-in. The
wrapper scrubs ambient `LLAMA_*` values, binds the BF16 opt-in only in the
guarded child, and records both flags and the exact candidate identity.

Exact next guarded model-parity command (fresh output, no `-Force`):

```powershell
$source = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-performance-final'
$waves  = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\benchmarks\qwen38-dspark-production-qualification-20260819\qualification-row-invariant-20260820\p1-final-v5\arms\target-only\waves.json'
$model  = 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf'

& (Join-Path $source 'scripts\Invoke-Qwen38ParityDiagnostic.ps1') `
  -Executable (Join-Path $source 'build-final-cuda132-ninja\bin\test-qwen38-recurrent-parity.exe') `
  -ModelPath $model `
  -TargetOnlyWavesPath $waves `
  -OutputPath (Join-Path $source 'parity-bf16-recurrent-projections-20260820.jsonl') `
  -PredictionStart 176 -PredictionCount 24 -RollbackStart 184 `
  -MinFreeRamGiB 20 -MinFreeVramMiB 5000 -MaxUsedVramMiB 27000 `
  -MaxRuntimeSeconds 1800 `
  -Qwen35Bf16RecurrentProjectionDiagnostic
```

## Layer-boundary bisection

`-LayerBoundaryBisection` is a bounded diagnostic-only refinement of an
explicit reviewed Qwen35 candidate. NVFP4-only, BF16-only, and the explicitly
requested combination are supported. Passing both candidate switches binds the
existing `QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PROJECTIONS_V2` identity and
attests both opt-ins; the wrapper never adds a candidate automatically. The
mode also requires exactly one eight-token window and does not change the
production server outside the selected diagnostic child.

The harness first runs callback-free scalar and width-8 controls. A coarse pass
then reads only the 64 canonical `l_out` block outputs and locates the first
block with an exact F32 row-byte difference. A second pass observes only that
block's exact input, attention/GDN branch output, attention residual, FFN
output, and block output. A recurrent block also records the finite
`state_predelta` input. Scalar and width-8 contexts never coexist. Every traced
arm is repeated from the same fully restored prefix, and its top-2 bits plus
recurrent partial-state digest must match the callback-free control.

The pinned model SHA binds the reviewed 64-layer layout: 48 recurrent blocks
and 16 full-attention blocks. A 2-D `attn_output` is the full-attention branch;
the recurrent branch is the 2-D `linear_attn_out`. The unrelated 4-D GDN core
also named `attn_output` is explicitly ignored by layer type and feature width,
even when its scalar token/sequence axes are singleton. Comparable activation
boundaries must remain `[n_embd, token_rows, 1, 1]`; axes are never remapped.
Activation comparisons use the
complete row bytes; the JSONL exposes only content-free hashes and mismatch
indices. The loaded model's actual recurrent-layer metadata source and resulting
map are attested against the pinned 48/16 map. `recurrent_partial_state` is scoped as recurrent-only;
full-attention KV is excluded from that compact comparison. Traced runtime is
not performance evidence.

```powershell
$source = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-performance-final'
$waves  = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\benchmarks\qwen38-dspark-production-qualification-20260819\qualification-row-invariant-20260820\p1-final-v5\arms\target-only\waves.json'
$model  = 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf'

& (Join-Path $source 'scripts\Invoke-Qwen38ParityDiagnostic.ps1') `
  -Executable (Join-Path $source 'build-final-cuda132-ninja\bin\test-qwen38-recurrent-parity.exe') `
  -ModelPath $model `
  -TargetOnlyWavesPath $waves `
  -OutputPath (Join-Path $source 'parity-layer-boundary-retry-20260820.jsonl') `
  -PredictionStart 176 `
  -PredictionCount 8 `
  -MinFreeRamGiB 20 `
  -MinFreeVramMiB 5000 `
  -MaxUsedVramMiB 27000 `
  -Qwen35Nvfp4FfnProjectionDiagnostic `
  -LayerBoundaryBisection
```

Do not use `-Force` for the first run.

### Recurrent layer-0 internal refinement

After the reviewed layer pass has located the first difference at recurrent
layer 0's `attention_output`, add `-RecurrentLayerInternalRefine`. This mode is
deliberately pinned to that exact finding and the same prediction window. It
runs an additional sequential scalar/width-8 pair from the approved prefix; the two
contexts never coexist.

The internal trace requires exactly 14 finite activation boundaries with the
reviewed Qwen3.8 feature dimensions: `attn_norm`,
`linear_attn_qkv_mixed`, `z`, `beta`, `beta_sigmoid`, `alpha`,
`gate`, `conv_output_silu`,
`q_conv_predelta`, `k_conv_predelta`, `v_conv_predelta`,
`gdn_core_output`, `final_output`, and `linear_attn_out`. The emitted
`gdn_core_output` is the source callback named `attn_output-0`, not the
post-projection `linear_attn_out-0`. Scalar rows and the
width-8 rows are compared as exact F32 byte slices on their real token axis;
axes are never flattened or remapped. The reported ordering is the reviewed
semantic source order, not a claim that parallel branches execute causally in
that order.

The intermediate `a_softplus` and `conv_output_raw` callbacks are intentionally
not selected: ending scheduler graph views there would disable the production
CUDA `SOFTPLUS -> MUL` and `SSM_CONV -> SILU` fusions. Their fused outputs
`gate` and `conv_output_silu` are captured instead.

`conv_states` and `state_predelta` are recorded separately as exact cache-input
prefix evidence plus scalar evolution hashes. They are not mislabeled as
post-write cache snapshots. Callback-free controls must retain the same top-2
logit IDs/bits and compact recurrent-state digest; this is not a full-logit or
raw-state byte-equality claim. The trace must also prove that callback
segmentation kept the ordinary GDN route at zero and used the fused-cache GDN
route exactly 768 times for the repeated scalar arm and 96 times for the
repeated width-8 arm. This diagnostic timing is not performance evidence.

```powershell
$source = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-performance-final'
$waves  = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\benchmarks\qwen38-dspark-production-qualification-20260819\qualification-row-invariant-20260820\p1-final-v5\arms\target-only\waves.json'
$model  = 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf'

& (Join-Path $source 'scripts\Invoke-Qwen38ParityDiagnostic.ps1') `
  -Executable (Join-Path $source 'build-final-cuda132-ninja\bin\test-qwen38-recurrent-parity.exe') `
  -ModelPath $model `
  -TargetOnlyWavesPath $waves `
  -OutputPath (Join-Path $source 'parity-combined-recurrent-layer0-internal-20260820.jsonl') `
  -PredictionStart 176 -PredictionCount 8 -RollbackStart 184 `
  -MinFreeRamGiB 20 `
  -MinFreeVramMiB 5000 `
  -MaxUsedVramMiB 27000 `
  -MaxRuntimeSeconds 1800 `
  -Qwen35Nvfp4FfnProjectionDiagnostic `
  -Qwen35Bf16RecurrentProjectionDiagnostic `
  -LayerBoundaryBisection `
  -RecurrentLayerInternalRefine
```

Do not use `-Force` for the first internal-refinement run.

### Full-attention layer-3 internal refinement

After the combined candidate locates the first difference at full-attention
layer 3's `attention_output`, use `-FullAttentionInternalRefine`. The switch is
mutually exclusive with `-RecurrentLayerInternalRefine` and is fail-closed to
exactly layer 3, prediction 176, and the reviewed combined V2 candidate. It
does not auto-enable either candidate.

The trace records exactly 14 content-free activation boundaries using the
pinned model dimensions D=256, Hq=24, Hkv=4: `attn_norm`, the contiguous joint
`qg_projection`, offline-derived `q_pre_norm` and `gate_pre_sigmoid`,
`q_post_norm`, raw `k_projection`, `k_post_norm`, raw `v_projection`,
post-RoPE Q/K, reshaped V, `kv_fa_output_pregate`, `attn_gated`, and the output
projection. Raw K/V callbacks have unique source names, so the diagnostic never
guesses between pre- and post-transform tensors. `gate_sigmoid` is deliberately
not selected because doing so would split the production SIGMOID-to-MUL
fusion. Q/G extraction respects the real per-head [Q256,G256] interleave; no
noncontiguous Q view is flattened.

Scalar and width-8 contexts remain strictly sequential. Each observed arm is
replayed from the complete prefix and must match its callback-free control in
top-2 bits, recurrent-state digest, complete-sequence-state digest, and the
entire CUDA route snapshot. The pinned route evidence requires scalar VEC
flash-attention 128 times, width-8 MMA-F16 16 times, no TILE route, fused
SIGMOID-MUL 128/16 times, Q8 K/V fallback SET_ROWS 256/32 times, and no
ROPE-VIEW-SET_ROWS fusion for this IMROPE model. These are correctness and
observer-invariance facts, not performance measurements.

```powershell
$source = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-performance-final'
$waves  = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\benchmarks\qwen38-dspark-production-qualification-20260819\qualification-row-invariant-20260820\p1-final-v5\arms\target-only\waves.json'
$model  = 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf'

& (Join-Path $source 'scripts\Invoke-Qwen38ParityDiagnostic.ps1') `
  -Executable (Join-Path $source 'build-final-cuda132-ninja\bin\test-qwen38-recurrent-parity.exe') `
  -ModelPath $model `
  -TargetOnlyWavesPath $waves `
  -OutputPath (Join-Path $source 'parity-combined-full-attention-layer3-internal-20260820.jsonl') `
  -PredictionStart 176 -PredictionCount 8 -RollbackStart 184 `
  -MinFreeRamGiB 20 -MinFreeVramMiB 5000 -MaxUsedVramMiB 27000 `
  -MaxRuntimeSeconds 1800 `
  -Qwen35Nvfp4FfnProjectionDiagnostic `
  -Qwen35Bf16RecurrentProjectionDiagnostic `
  -LayerBoundaryBisection `
  -FullAttentionInternalRefine
```

Do not use `-Force` for the first full-attention refinement run. If this trace
confirms equality through projections/gating but first diverges at the
flash-attention output, the next separately reviewed candidate is the exact
D256, Q8-K/Q8-V tuple with query columns 3..8 routed to the existing VEC
kernel using `cols_per_block=2`. Width 2 is already exact and must remain on
its existing path; this locator does not implement that candidate.

### Full-attention D256/Q8/GQA6 VEC candidate

The direct V3 candidate is the smallest follow-up to the layer-3 result. It
keeps the existing NVFP4 FFN/head and BF16 recurrent-projection candidates and
adds one explicit, default-off graph hint for Qwen3.5 main-stack full-attention
ops. CUDA accepts the hint only on the reviewed Blackwell SM120 contract:
D=256, 24 query heads, 4 Q8_0 K/V heads (GQA6), a causal F16 mask, F32 output,
the exact Q/K/V/mask/output strides, scale 0.0625, F32 precision, no sinks, and
query columns 3..8. M1, M2, M9+, MTP, other architectures, missing hints, wrong
types/layouts, and non-Blackwell devices retain the established selector.

The selected path is the existing VEC implementation with
`cols_per_block=2`; no new attention kernel is introduced. The JSONL contains
one single-batch `fattn_graph_hint_route`, one `fattn_candidate_predicate`,
and one `fattn_candidate_route` record for every width 1..8. The first record
proves independently that the graph tag reached CUDA. The predicate record
contains only integers: its fail mask plus CC/compiled architecture,
op/hint/types, Q/K/V/mask/output shapes and strides, source-4 presence,
contiguity, parameter float bits, and precision. It is flushed before any
route failure. The wrapper recomputes the complete predicate contract from
those integers instead of trusting the reported mask: M1/M2 must report
`0000000000000044` (only the absent graph-hint bit 2 and unsupported-width bit
6), while M3..M8 must report `0000000000000000`. The M1 record is captured on
the prefix producer's first scalar decode immediately
after the differently shaped prompt; M2..8 restore that completed prefix into
fresh one-context-at-a-time probes. Widths 1/2 must show zero hint/candidate hits and exactly 16
general VEC hits. Every width 3..8 must show exactly 16 hint, candidate, and general
VEC hits (one per reviewed full-attention layer), with MMA-F16 and TILE both
zero. This prevents prefix graph replay from hiding the unchanged M1/M2 route.
The subsequent full numerical pass repeats all
eight widths independently and compares full sequence-state digests. These
counters are correctness evidence, not a performance claim. Expected risk is higher latency
for width 3..8 full-attention blocks because VEC replaces the faster MMA-F16
path; promotion requires complete width/state/rollback parity before any
throughput measurement.

```powershell
$source = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-performance-final'
$waves  = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\benchmarks\qwen38-dspark-production-qualification-20260819\qualification-row-invariant-20260820\p1-final-v5\arms\target-only\waves.json'
$model  = 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf'

& (Join-Path $source 'scripts\Invoke-Qwen38ParityDiagnostic.ps1') `
  -Executable (Join-Path $source 'build-final-cuda132-ninja\bin\test-qwen38-recurrent-parity.exe') `
  -ModelPath $model `
  -TargetOnlyWavesPath $waves `
  -OutputPath (Join-Path $source 'parity-combined-fattn-vec-v3-20260820.jsonl') `
  -PredictionStart 176 -PredictionCount 24 -RollbackStart 184 `
  -MinFreeRamGiB 20 -MinFreeVramMiB 5000 -MaxUsedVramMiB 27000 `
  -MaxRuntimeSeconds 1800 `
  -Qwen35Nvfp4FfnProjectionDiagnostic `
  -Qwen35Bf16RecurrentProjectionDiagnostic `
  -Qwen35FullAttentionVecDiagnostic
```

Do not use `-Force`. Promote only if width 1..8 token IDs and logit bits,
all full final states, the rollback batch, and rollback depths 1..7 are exact
and all eight graph-hint, predicate, and selector record sets pass.

Interpretation is deliberately fail-closed:

- A width-8 top-1 difference before any rollback locates the problem in the
  target's width-dependent compute path. It does not identify which operator is
  responsible. FP8 projection, NVFP4 projection/fused SwiGLU, and fused GDN must
  then be isolated with operator-level route-marker tests.
- A rollback/oracle difference alone cannot separate width-dependent snapshot
  generation from snapshot selection/replay. The record explicitly marks this
  attribution as inseparable. Snapshot selection becomes a specific suspect
  only after an operator-level or per-plane comparison proves that the exact
  width-8 batch generated scalar-identical intermediate snapshots.
- If both width and rollback records differ, the rollback comparison is
  inconclusive because it inherits width-dependent numerical state. Establish
  the first differing tensor before changing rollback code.
- Matching FNV and byte count are strong compact evidence, not a cryptographic
  proof, that the serialized states agree. A mismatch is evidence of different
  serialized state, not by itself evidence of an incorrect state.

The executable returns success when the diagnostic completed, regardless of
whether values differ. Operational failures return nonzero. The final JSONL
record always contains `"exact_parity_gate_bypassed":false`.
Exceptions emit a flushed JSON record on stderr with `type`, `stage`, and an
escaped actionable `message`, including failures before model loading.
