# Qwen3.8 row-invariance locator

## What the guarded model locator proved

`parity-diagnostic-fresh-current-20260820.jsonl` completed operationally and is
repeat-stable. Prediction indices are zero-based output-token indices, so index
182 is the 183rd generated token.

- At index 176, all widths still select token 3242, but top-2 values already
  differ. Scalar margin is 0.870368958; W2 is 0.825881958, W4 is 0.911373138,
  and W8 is 0.786785126.
- At index 182, scalar selects 6391 over 4277 with margin 0.405575752. W2 and
  W4 still select 6391, but their runner-up changes to 2064 (margins
  0.00448036194 and 0.137525558). W8 flips to 2064 over token 1 with margin
  0.0703268051.
- Width final-state hashes after consumed output index 198 are all distinct:
  W1 `ce72c8cf578b9aa4`, W2 `0ce6c450574ab39e`, W4 `15d899a88f58aef2`, and
  W8 `d1a48ace6bac8a87`.
- The W8 rollback batch covers prediction indices 184 through 191. Its only
  raw top-1 failure is index 191: token 3108 wins over expected token 5224 by
  0.0516967773.
- Rollbacks 1 through 7 all mismatch selected and continued state. Only
  rollback 1 changes continued top-1: after committing output index 189 and
  continuing with output index 190, prediction index 191 selects 3108 instead
  of oracle 5224 (margin 0.0133991241 versus 0.166227341).

These facts localize a width-dependent numerical difference before rollback.
They do not identify FP8, NVFP4, GDN, or rollback selection as the faulty
operator. The rollback artifact explicitly remains generation-versus-selection
inseparable. Repeat stability excludes observed run-to-run drift; it does not
prove numerical correctness.

## Operator matrix

`test-qwen38-row-invariance` compares the exact F32 bit patterns from eight
independent M1 evaluations with one M8 evaluation. It uses deterministic,
cancellation-sensitive inputs, eight replays, and alternates M1-first/M8-first
execution order.

Quick mode has six cases:

1. FP8 E4M3 `K5120/N1024`.
2. Raw NVFP4 `K5120/N128`.
3. Raw NVFP4 `K17408/N128`.
4. Qwen-style NVFP4 gate/up/scales/SwiGLU `K5120/N128` (M1 fused, M8 unfused).
5. GDN one-token K1 versus K8 snapshot slot 0 through the production-shaped
   strided cache fusion.
6. GDN eight sequential K1 calls versus one eight-token K8 call through the
   same cache fusion, including newest-first snapshot mapping.

Full mode adds the other four production FP8 shapes: `5120x6144`, `6144x5120`,
`5120x10240`, and `5120x12288`.

Every case fails closed unless its CUDA route marker advances. Raw NVFP4 must
prove distinct M1 and M8 MMVQ routes; the FFN case must prove M1 fused FFN and
M8 unfused MMVQ. GDN must prove the fused-cache route, not merely its contiguous
output tail. Route instrumentation is compile-gated and absent from ordinary
production dispatch unless `GGML_CUDA_DIAGNOSTIC_ROUTES=ON` was configured.
Passing this locator only clears the listed operator/shape fixtures. It does not
by itself clear end-to-end Qwen3.8 or DSpark parity.

## Quick6 result and targeted candidate

The guarded Quick6 run stopped on the first raw NVFP4 case, after the FP8 case
passed bitwise. For raw NVFP4 `K5120/N128`, replay 0 output 1 was
`0x44632808` (`908.625488`) for eight scalar M1 evaluations and `0x44632806`
(`908.625366`) for M8. This proves an M-dependent NVFP4 reduction difference at
the production LM-head shape before rollback. Because the run stopped at the
first failure, it says nothing about the remaining four Quick6 cases.

The candidate is intentionally narrow:

- Only the dense Qwen3.5/Qwen3.8 NVFP4 LM-head graph node receives the explicit
  `GGML_HINT_MUL_MAT_ROW_INVARIANT` hint.
- LoRA-composed heads, non-contiguous layouts, unsupported fusion operands,
  M1, M greater than 16, and other model/head nodes keep their existing paths.
- Tagged M2 through M16 reuse the exact current M1 NVFP4/Q8_1 kernel
  specialization as concurrent independent grid channels: four warps, one
  output row per block, the same reduction, and the existing fused scalar
  output-scale epilogue.

`--head-candidate` checks every accepted width M2 through M16 against joined M1
rows, and then exercises M8 with distinct, duplicate, and permuted rows across
eight alternating-order graph replays. It requires exact F32 bits, the fused
output scale, a positive candidate route marker, the unchanged M1 route marker,
and a negative untagged-M8 control. This is correctness/route evidence only; it
does not establish performance or end-to-end model parity.

The executable's CPU-only matrix validation is safe to run without the GPU
guard:

```powershell
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --quick
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --full
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --head-candidate
```

Do not run `--cuda` directly. After the reviewed executable, guard, adjacent
DLLs, and resolved `cudart64_13.dll` hashes match the wrapper anchors, use:

```powershell
& .\scripts\Invoke-Qwen38RowInvariance.ps1 `
  -Executable .\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe `
  -OutputDirectory .\row-invariance-head-candidate-20260820 `
  -Mode head-candidate -GpuIndex 0 `
  -MinFreeRamGiB 20 -MinFreeVramMiB 12000 `
  -MaxUsedVramMiB 20000 -MaxRuntimeSeconds 1800
```

The wrapper scrubs inherited `LLAMA_*`, `GGML_*`, and `CUDA_*` variables,
binds the reviewed runtime set, launches once through the exclusive GPU guard,
and persists a content-free JSONL trace plus a status-bearing receipt and
postflight proof even on failure.

## First guarded head-candidate result

The first guarded head-candidate attempt stopped after 1.9 seconds. Its
untagged negative control correctly observed a zero candidate-route count, and
the first width comparison completed bitwise before the old aggregate route
assertion failed. The receipt and postflight proof reported zero remaining
relevant processes and leases.

That attempt did not serialize the individual route counters, so the artifact
alone cannot name the missing counter. Static dispatch analysis plus its CUDA
graph warmup/reset log identifies the stale assertion: route markers count host
dispatches, while the test reset them between newly allocated graphs. CUDA
graphs are keyed by the first node address; allocator reuse can replay the
unchanged scalar M1 graph without another host dispatch, leaving the reset M1
counter at zero even though the already captured M1 kernel ran. Batch width
changes still force a new tagged dispatch.

The corrected diagnostic keeps the scalar graph and every M2 through M16 graph
alive simultaneously, resets route counters once, requires the unchanged M1
route cumulatively, and requires a positive tagged-scaled counter delta for
each width. Before any route assertion it emits a `width_route` JSON record
containing the individual observed counts and explicit expected minima. It also
checks that the constructed graph retains the hint and the exact scalar-scale
fusion shape. No production CUDA dispatch was changed for this correction.

## Opt-in Qwen3.5 projection candidate

The LM-head-only model run changed a small set of width-8 logit bits, proving
that the tagged head path was selected, but it did not change any width/rollback
summary or recurrent-state hash. The next diagnostic therefore covers all
NVFP4 dense projections in the Qwen3.5 graph: fused gate+up+SwiGLU, down, and
the existing LM head. The target model has 64 dense layers, so its 193 NVFP4
tensors are exactly `64 * (gate + up + down) + output`; attention projections
are FP8 E4M3.

This wider candidate remains diagnostic-only. It is enabled only when
`LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS=1` is bound by the reviewed
parity wrapper. Default execution, M1, non-NVFP4 weights, LoRA-composed
matmuls, unsupported layouts/fusions, and widths outside M2 through M16 keep
their existing dispatch. `--projection-candidate --validate-matrix` is the
CPU-only contract check; guarded CUDA execution additionally requires both
production shapes (`5120 -> 17408` fused gate/up and `17408 -> 5120` down),
negative untagged controls, all widths M2 through M16, three row fixtures, and
eight alternating-order graph replays to be bitwise equal to joined M1 rows.

The row-invariant path intentionally reuses the M1 reduction independently per
token. It can be slower than the ordinary batched M2-M16 route, particularly
because it applies to 192 FFN projections per decode step. No performance gain
or regression is claimed until the end-to-end guarded run is measured.

## Selective FP8 and GDN locators

`--fp8` selects only the five production FP8 E4M3 shapes. Each shape compares
joined M1 rows with every batch width M2 through M8, requires finite results and
exact F32 bits, performs eight alternating-order replays, and proves that the
native FP8 CUDA route advanced for every width. No NVFP4 or GDN case can stop or
invalidate this artifact.

`--gdn` selects only the two existing production-shaped fused-cache semantic
cases: K1 versus K8 snapshot slot zero, and eight sequential K1 updates versus
one K8 update with newest-first strided-cache placement. Both require finite,
bitwise-stable results across eight alternating replays and the mandatory fused
cache route. No FP8 or NVFP4 case participates.

CPU-only matrix checks:

```powershell
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --fp8
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --gdn
```

The guarded runs must be separate jobs with separate evidence directories:

```powershell
& .\scripts\Invoke-Qwen38RowInvariance.ps1 `
  -Executable .\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe `
  -OutputDirectory .\row-invariance-fp8-only-20260820 `
  -Mode fp8 -GpuIndex 0 `
  -MinFreeRamGiB 20 -MinFreeVramMiB 12000 `
  -MaxUsedVramMiB 20000 -MaxRuntimeSeconds 1800

& .\scripts\Invoke-Qwen38RowInvariance.ps1 `
  -Executable .\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe `
  -OutputDirectory .\row-invariance-gdn-only-20260820 `
  -Mode gdn -GpuIndex 0 `
  -MinFreeRamGiB 20 -MinFreeVramMiB 12000 `
  -MaxUsedVramMiB 20000 -MaxRuntimeSeconds 1800
```

Do not add `-Force` to the first run. The wrapper pins the executable, guard,
adjacent DLLs, and resolved CUDA runtime, scrubs inherited runtime variables,
and writes mode-specific JSONL, stderr, guard receipt, and postflight proof.

## Selective recurrent layer-0 operator locators

These modes isolate production-route components without enabling model changes:

- `bf16-projections`: the exact `K5120/N48` beta and alpha projections. M1
  must select BF16 MMVF with the original F32 activation; M2 through M8 must
  select cuBLAS, whose activation conversion to BF16 is deliberately exercised
  by a non-BF16-grid fixture. Route proof is emitted before bitwise comparison.
- `bf16-candidate`: the opt-in replacement for those two tagged recurrent
  projections only. It preserves ordinary MMVF for M1 and forces the same MMVF
  reduction for M2 through M8. Beta includes the production reshape/sigmoid;
  alpha includes reshape/add/softplus/negative-scale/reshape. Untagged M8 must
  remain on cuBLAS, and a tagged non-contiguous M8 view must also fall back to
  cuBLAS with both candidate counters at zero. The test compares the raw matmul
  F32 bits and the transformed output bits separately. The shared graph-hint
  contract rejects LoRA residuals before a hint reaches the backend. The
  synchronized microbenchmark is operator-local evidence, not an end-to-end
  performance qualification.
- `rms`: weighted RMSNorm on `[5120,M]`, with only the fused
  `RMS_NORM -> MUL(weight)` route accepted.
- `ssm-conv`: fused `SSM_CONV(d_conv=4,channels=10240) -> SILU`, using a
  deterministic nonzero three-row history. It compares eight sequential scalar
  outputs and evolved windows with one M8 output and all eight newest-first
  rollback cache slots.
- `l2`: q and k L2 normalization on production-strided views
  `[128,16,M,1]` into `[10240,M]`, including the real k byte offset.
- `gated-norm`: the post-GDN `[128,48,M,1]` weighted RMSNorm and gated
  SILU/multiply graph, requiring both production fusions and zero fallbacks.

All modes use eight alternating-order replays, exact F32 bit comparison, exact
route schemas, and reject every unrelated diagnostic route. CPU-only contract
checks never initialize CUDA:

```powershell
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --bf16-projections
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --bf16-candidate
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --rms
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --ssm-conv
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --l2
.\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe --validate-matrix --gated-norm
```

Run each CUDA locator as its own exclusive guarded job. Start with the stronger
BF16 suspect; do not add `-Force` to a fresh output directory:

```powershell
& .\scripts\Invoke-Qwen38RowInvariance.ps1 `
  -Executable .\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe `
  -OutputDirectory .\row-invariance-bf16-candidate-20260820 `
  -Mode bf16-candidate -GpuIndex 0 `
  -MinFreeRamGiB 20 -MinFreeVramMiB 12000 `
  -MaxUsedVramMiB 20000 -MaxRuntimeSeconds 1800

& .\scripts\Invoke-Qwen38RowInvariance.ps1 `
  -Executable .\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe `
  -OutputDirectory .\row-invariance-bf16-projections-20260820 `
  -Mode bf16-projections -GpuIndex 0 `
  -MinFreeRamGiB 20 -MinFreeVramMiB 12000 `
  -MaxUsedVramMiB 20000 -MaxRuntimeSeconds 1800

& .\scripts\Invoke-Qwen38RowInvariance.ps1 `
  -Executable .\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe `
  -OutputDirectory .\row-invariance-rms-only-20260820 `
  -Mode rms -GpuIndex 0 -MinFreeRamGiB 20 -MinFreeVramMiB 12000 -MaxUsedVramMiB 20000 -MaxRuntimeSeconds 1800

& .\scripts\Invoke-Qwen38RowInvariance.ps1 `
  -Executable .\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe `
  -OutputDirectory .\row-invariance-ssm-conv-only-20260820 `
  -Mode ssm-conv -GpuIndex 0 -MinFreeRamGiB 20 -MinFreeVramMiB 12000 -MaxUsedVramMiB 20000 -MaxRuntimeSeconds 1800

& .\scripts\Invoke-Qwen38RowInvariance.ps1 `
  -Executable .\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe `
  -OutputDirectory .\row-invariance-l2-only-20260820 `
  -Mode l2 -GpuIndex 0 -MinFreeRamGiB 20 -MinFreeVramMiB 12000 -MaxUsedVramMiB 20000 -MaxRuntimeSeconds 1800

& .\scripts\Invoke-Qwen38RowInvariance.ps1 `
  -Executable .\build-final-cuda132-ninja\bin\test-qwen38-row-invariance.exe `
  -OutputDirectory .\row-invariance-gated-norm-only-20260820 `
  -Mode gated-norm -GpuIndex 0 -MinFreeRamGiB 20 -MinFreeVramMiB 12000 -MaxUsedVramMiB 20000 -MaxRuntimeSeconds 1800
```

These are correctness locators, not performance benchmarks. A passing result
only clears the exact named graph, shape, layout, and route. A BF16 failure is
expected to be especially informative because it proves the M1-MMVF versus
batched-cuBLAS conversion boundary before SSM convolution.
