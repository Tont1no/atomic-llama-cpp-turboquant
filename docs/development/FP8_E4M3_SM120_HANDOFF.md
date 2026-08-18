# Native ModelOpt FP8 E4M3 SM120 handoff

This document freezes the exact artifacts, build contract, runtime arguments, and
integration order for the Qwen3.8-27B native-FP8 target and its DSpark draft.
The production path keeps the selected ModelOpt E4M3 weights as FP8 and executes
them with cuBLASLt W8A8; it does not dequantize the weight for every matmul.

## Source and artifact identity

Implementation worktree:

```text
C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-fp8-e4m3
branch: codex/fp8-e4m3-runtime
base tag b10448: ad1de39e0708e3ced9c71bb3c82d93a2c046a73f
native FP8 implementation: 620079c4e591da3c6acfd405072fa9bc62556143
official DSpark schema and FP8 LM-head sidecar fix: a593261bc82f3934b269da61fe36fb26fa6dba41
```

Target source checkpoint:

```text
C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-HF
```

Native-FP8 target GGUF:

```text
C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf
size: 21016372896 bytes
SHA256: 46E0EC4BDAB3907346FBB486B1E4F78B49A66B245374AA0D3D628A8D099C0336
preserved FP8 aggregate raw-byte SHA256: 0ad15647c24f6d8d2d8593b1673a04ef69df4b7855c3d2d6c9243d76aaa523d1
layout: 208 E4M3 weights, 416 scalar scale sidecars, 193 NVFP4 tensors
```

The aggregate hash is the validator's source-to-GGUF hash over the 208 preserved
FP8 weight payloads. It is not a substitute for the full-file SHA256.

DSpark draft GGUF:

```text
C:\Users\pasca\Documents\GitHub\Ai-Loader\models\draft\Qwen3.8-27B-DSpark-Q8_0.gguf
size: 1455376576 bytes
SHA256: B007A76B2CE57C1A3ECF36046543AEBF60A763861624647D653F16D336C781D2
```

## Conversion contract

Convert the target while preserving only the supported ModelOpt scalar-scale
E4M3 projection roles:

```powershell
$Source = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-fp8-e4m3'
$TargetHF = 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-HF'
$TargetGGUF = 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf'

Set-Location -LiteralPath $Source
python convert_hf_to_gguf.py $TargetHF `
    --outfile $TargetGGUF `
    --outtype bf16 `
    --keep-fp8-e4m3

python tests/test-modelopt-fp8-gguf.py `
    --model-dir $TargetHF `
    --gguf $TargetGGUF `
    --expected-count 208
```

`--keep-fp8-e4m3` and `--fp8-as-q8` are mutually exclusive. Unsupported
architectures, tensor roles, shapes, or missing/non-scalar sidecars must fail
conversion rather than silently producing a nominal native-FP8 model.

The official RadixArk checkpoint uses the architecture name
`DSparkDraftModel`. Commit `a593261bc` adds that alias to the Qwen DSpark
converter. Convert a local checkout of the draft and then quantize it:

```powershell
$DraftHF = '<local path to RadixArk/Qwen3.8-27B-DSpark>'
$DraftBF16 = '<output path>\Qwen3.8-27B-DSpark-BF16.gguf'
$DraftQ8 = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\models\draft\Qwen3.8-27B-DSpark-Q8_0.gguf'
$Quantize = "$Source\build-fp8-sm120-cuda132\bin\Release\llama-quantize.exe"

python convert_hf_to_gguf.py $DraftHF `
    --target-model-dir $TargetHF `
    --outfile $DraftBF16 `
    --outtype bf16

& $Quantize $DraftBF16 $DraftQ8 Q8_0
```

Do not pass `--keep-fp8-e4m3` when converting the DSpark draft. The draft is a
separate Q8_0 model; the target remains the native-FP8/NVFP4 model.

## CUDA 13.2 build contract

The supported build uses the local CUDA 13.2.2 toolkit (nvcc 13.2.86,
cuBLAS 13.4.1.3) and compiles specifically for SM120. CUDA 13.1 is
prototype/compile-only and is rejected by the native runtime gate.

```powershell
$Cuda132Root = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\cuda-13.2\toolkit'
$Source = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-fp8-e4m3'
$Build = "$Source\build-fp8-sm120-cuda132"

$env:CUDA_PATH = $Cuda132Root
$env:CUDACXX = "$Cuda132Root\bin\nvcc.exe"
$env:PATH = "$Cuda132Root\bin;$Cuda132Root\bin\x64;$env:PATH"

cmake -S $Source -B $Build `
    -G 'Visual Studio 17 2022' -A x64 `
    -T "cuda=$Cuda132Root" `
    "-DCMAKE_CUDA_COMPILER=$Cuda132Root\bin\nvcc.exe" `
    "-DCUDAToolkit_ROOT=$Cuda132Root" `
    -DCMAKE_CUDA_ARCHITECTURES=120 `
    -DGGML_CUDA=ON `
    -DGGML_CUDA_GRAPHS=ON `
    -DLLAMA_BUILD_TESTS=ON `
    -DLLAMA_BUILD_SERVER=ON

cmake --build $Build --config Release --parallel `
    --target test-fp8-e4m3 llama-server llama-quantize
```

With the Visual Studio generator, `-T "cuda=$Cuda132Root"` is mandatory.
`CMAKE_CUDA_COMPILER` and `CUDAToolkit_ROOT` alone previously allowed a silent
fallback to the globally installed CUDA 13.1 toolkit.

Toolkit provenance is recorded at:

```text
C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\cuda-13.2\PROVENANCE.md
```

Ship the matching CUDA 13.2 DLL set beside the executable or keep both toolkit
binary directories at the front of the scoped runtime `PATH`:

| DLL | SHA256 |
| --- | --- |
| `cudart64_13.dll` | `47c6fc219d98fb4bd4e569ce56fb08795e28d89ec0ce1b44753ca0c072a6345c` |
| `cublas64_13.dll` | `91200c2ff57b8477e94254a4501030d75e1bb35a2de3a476e7ab66df897cb046` |
| `cublasLt64_13.dll` | `8f54d7b3e5173bf2659f45bad6ae789ffef8218ab6254f10c0cc5e3ec874c` |

The direct `cublasLt64_13.dll` dependency is intentional. The Windows release
workflow copies all three DLLs; an older or global `cublasLt64_13.dll` must not
win DLL resolution.

## Runtime argument sets

The following PowerShell arrays are the reviewed arguments. Production must keep
warmup enabled so the scale-specific FP8 plans are created before request-time
CUDA graph capture. `--no-warmup` is for bounded diagnostics only.

```powershell
$Server = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\llama-fp8-e4m3\build-fp8-sm120-cuda132\bin\Release\llama-server.exe'
$TargetGGUF = 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf'
$DraftGGUF = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\models\draft\Qwen3.8-27B-DSpark-Q8_0.gguf'

$Common = @(
    '--model', $TargetGGUF,
    '--ctx-size', '9216',
    '--parallel', '1',
    '--cache-type-k', 'q8_0',
    '--cache-type-v', 'q8_0',
    '--flash-attn', 'on',
    '--batch-size', '2048',
    '--ubatch-size', '512',
    '--n-gpu-layers', 'all',
    '--device', 'CUDA0',
    '--split-mode', 'none',
    '--fit', 'off',
    '--cache-ram', '0',
    '--ctx-checkpoints', '0',
    '--no-cache-idle-slots',
    '--no-cache-prompt',
    '--kv-unified',
    '--metrics',
    '--reasoning', 'off',
    '--warmup'
)

$TargetOnlyArgs = $Common

$DSparkArgs = $Common + @(
    '--spec-type', 'draft-dspark',
    '--spec-draft-model', $DraftGGUF,
    '--spec-draft-n-max', '7',
    '--spec-draft-n-min', '0',
    '--spec-draft-p-min', '0',
    '--spec-draft-type-k', 'q8_0',
    '--spec-draft-type-v', 'q8_0',
    '--spec-draft-ngl', 'all',
    '--spec-draft-device', 'CUDA0'
)
```

`--spec-draft-n-max 7` matches the official draft's trained block size. The
runtime also clamps the requested value to the model metadata. A non-zero
`--spec-draft-p-min P` is optional and enables confidence-based early stopping.

Do not execute `$Server` directly during shared development. Every GPU executable
must be launched through the reviewed v2 form of
`C:\Users\pasca\Documents\GitHub\Ai-Loader\scripts\Invoke-ExclusiveGpuTask.ps1`
with at least these limits:

```text
MinFreeRamGiB=16
MinFreeVramMiB=4096
MaxUsedVramMiB=28672
preflight relevant GPU/server processes=0
one server or test process at a time
explicit stop and postflight relevant processes=0
```

The existing guard revision is paused pending its race/cleanup fix. Do not run
either argument set until guard v2 has passed review and exclusive GPU ownership
has been explicitly reassigned.

## DSpark and FP8 static compatibility

The Qwen DSpark target-feature path is compatible with preserved target FP8
weights and their scalar sidecars:

- Target taps `[4, 16, 28, 40, 52]` become block-input slots
  `[5, 17, 29, 41, 53]`, matching outputs after those target layers.
- Each exported target feature is the contiguous F32 block input captured before
  the target layer's norm/matmul. The FP8 weight and scale sidecars are downstream
  ancestors and do not become part of the extracted feature representation.
- The device fastpath imports detached F32 leaves only after exact type, shape,
  device, and zero-target-ancestor checks. The host fallback preserves the same
  row/layer/hidden ordering.
- The Q8_0 DSpark draft has no dependency on the target's FP8 GGUF tensor enum.
- Borrowed target output projections now forward both `output_s` and
  `output_in_s`. This fixes generic preserved-FP8 LM heads and is harmless for
  the exact audited target, whose LM head is NVFP4.

The exact target/draft pair is therefore statically correct. This is not a claim
that DSpark throughput has been qualified: DSpark still requires its own guarded
RTX 5090 parity and performance run.

## Verification status

Post-hardening CPU/build-only checks:

- CUDA 13.2 Release build of `test-fp8-e4m3`, `llama-server`, and the changed
  `dflash.cpp` passed. The binaries were built but no GPU test/server was started.
- `python -m compileall -q conversion` passed.
- `get_model_class('DSparkDraftModel')` resolved to
  `conversion.qwen.DSparkModel`.
- `convert_hf_to_gguf.py --print-supported-models` lists
  `DFlashDraftModel`, `DSparkDraftModel`, and `Qwen3DSparkModel`.
- Independent static review found no remaining P0/P1 in target-only execution,
  DSpark feature extraction, sidecar routing, or the official draft schema.

The native target had an earlier guarded full-model load/generation success at
the FP8 implementation commit, but the DSpark pair has not been run after the
official-schema/LM-head follow-up. Do not treat this document as final N1/N4/N8
performance evidence.

## Integration into the shared b10448 performance branch

The existing shared branch is already the correct integration base:

```text
branch: codex/qwen-performance-integration
base: ad1de39e0 (b10448)
current inspected HEAD: 3424287930aa67a63bb3f97bbdfacce848628928
```

It already contains patch-equivalent versions of the five device-resident
DFlash commits, the FP8 plan, and the native FP8 implementation (`119760874` and
`3c150a0c8`). `git cherry` confirms that only `a593261bc` is new from this FP8
worktree. Therefore the safe integration is:

1. Update a dedicated worktree for `codex/qwen-performance-integration` and
   confirm its expected HEAD/clean state.
2. Cherry-pick `a593261bc82f3934b269da61fe36fb26fa6dba41`.
3. Cherry-pick the documentation-only handoff commit if the durable runbook is
   wanted on that branch.
4. Do not cherry-pick `780506a50` or `620079c4e` again; their patch-equivalent
   commits are already present.
5. Reconfigure from a fresh build directory using the exact CUDA 13.2 contract,
   build Release, run the Python converter/registry checks, and validate the
   exact 208/416 tensor counts and hashes before any GPU work.
6. After guard v2 review and exclusive assignment, qualify in this order:
   native FP8 micro-parity for audited shapes; target-only C1; DSpark C1; then
   paired N4/N8 runs. Record native-route counters, model placement, context,
   output tokens, total throughput, p50/p95, acceptance, and errors.

Do not merge `codex/qwen-final-benchmark-evidence` wholesale. Its inspected HEAD
`1d6c42eeef8c0d8ae3a451b4c3c1527144689135` diverges from the FP8 branch at the
much older `51eae8cfc`; selectively replay only its evidence/benchmark changes
after the b10448 functional and performance stack is stable.
