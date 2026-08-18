# ModelOpt FP8 E4M3 on SM120: implementation gate and plan

Base: `73c6d7f58b24f2f9aad72b77ad8be47a9c36c8eb` (b10448-derived DFlash device branch)

## Decision

Do not add a storage-only `GGML_TYPE_F8_E4M3` fallback and call it FP8 runtime support.
The current CUDA backend has no FP8 GEMM path. A new one-byte tensor type could be
dequantized to F16 before cuBLAS, but the current cuBLAS fallback allocates and converts
the full weight for every multiplication. For Qwen3.8-27B this would expand 6.72 GiB of
FP8 projection weights to 13.44 GiB repeatedly and would be slower than the existing
`--fp8-as-q8` conversion. That is not a safe performance implementation.

A correct implementation needs the GGUF type, scale semantics, and a native SM120
W8A8 GEMM to land together. Until that path is implemented and validated, keep
`--fp8-as-q8` as the explicit llama.cpp fallback.

## Checkpoint audit

Audited checkpoint:
`RadixArk/Qwen3.8-27B-NVFP4`, revision
`554ebba9b5f1b79dc11246341960360e6ef05ef4`.

The safetensors contain:

- 208 FP8 E4M3 weight tensors, 7,214,202,880 elements (6.71875 GiB raw).
- 208 scalar FP32 `weight_scale` tensors and 208 scalar FP32 `input_scale` tensors.
- 193 packed NVFP4 weight tensors whose block-scale tensors are also E4M3.
- 401 E4M3 tensors in total: 208 actual FP8 weights plus 193 NVFP4 block scales.
- All FP8 GEMM dimensions are multiples of 16. Shapes are:
  - 32 x `(1024, 5120)`
  - 64 x `(5120, 6144)`
  - 48 x `(6144, 5120)`
  - 48 x `(10240, 5120)`
  - 16 x `(12288, 5120)`

The FP8 layers are exactly the attention and linear-attention projections. The dense
MLPs and LM head are NVFP4. ModelOpt uses per-tensor static W8A8 scales here:

```
W = fp8_codes(W) * weight_scale
Xq = clamp(X / input_scale, -448, 448).to(E4M3)
Y = (Xq @ fp8_codes(W)) * input_scale * weight_scale
```

Preserving FP8 instead of converting these weights to Q8_0 saves only about 0.42 GiB
(`6.71875 GiB` versus approximately `7.13867 GiB`). The important gain is native FP8
Tensor Core compute, not storage alone.

## Current b10448 behavior

- GGUF/GGML ends at type 42 (`GGML_TYPE_Q2_0`); there is no FP8 weight type.
- The converter preserves packed ModelOpt NVFP4, including its E4M3 block scales and
  scalar output scale.
- ModelOpt FP8 weights are dequantized by `weight.float() * weight_scale`.
- With `--fp8-as-q8`, those dequantized FP8 weights are stored as Q8_0. Without the
  flag they become BF16/F16.
- CUDA includes `cuda_fp8.h` only to encode/decode NVFP4 scale bytes. There is no
  `CUDA_R_8F_E4M3`/cuBLASLt/FlashInfer FP8 matrix multiplication.
- The native Blackwell NVFP4 path accepts `GGML_TYPE_NVFP4` weights. Its activation
  operand starts as F32 and is dynamically quantized to FP4 for the native MMA path
  (or Q8_1 for the generic path). It does not accept FP8 weights or FP8 activations.

## Required implementation slices

### 1. Storage and conversion

Add `GGML_TYPE_F8_E4M3` as a scalar one-byte floating type in C and Python GGUF enums,
type traits, readers, endian handling, tensor-size validation, and inspection tools.
Do not encode FP8 as `I8` or `U8`; those types have different arithmetic semantics.

Add an opt-in converter mode such as `--keep-fp8-e4m3` that:

- writes the original E4M3 weight bits losslessly;
- maps `weight_scale` to the GGUF tensor's `.scale` sidecar;
- maps `input_scale` to `.input_scale`;
- preserves Qwen's row/column transforms on the raw byte tensor;
- keeps `k_scale` and `v_scale` separate for a later FP8-KV feature;
- rejects E5M2 and non-scalar/blockwise scale layouts rather than silently treating
  them as this format.

Add converter tests which reopen the GGUF and verify all 208 FP8 tensor types, shapes,
raw-byte hashes, and 416 scalar scales. Add dequantization parity tests against PyTorch.

### 2. Scaled-matmul graph contract

The existing `GGML_OP_MUL_MAT` has only weight and activation inputs. The FP8 kernel
also needs `weight_scale` and `input_scale`. The model currently loads these sidecars,
but input scales are not part of the multiplication graph.

Add an explicit four-input scaled matmul operation (or an equivalently explicit backend
contract) rather than looking up tensors by name inside CUDA. Its reference semantics
must be the formula above. Update Qwen3.5/Qwen3.8 attention, GDN projections, and the
generic dense helper to use it only when the weight type is FP8 E4M3. Existing F16,
Q8, and NVFP4 graphs must remain byte-for-byte unchanged.

Provide a CPU reference implementation for correctness and an opt-in diagnostic CUDA
dequantization fallback. The diagnostic fallback must not be the default production
path and must log that it expands FP8 to F16.

### 3. Native SM120 W8A8 CUDA path

Implement a cuBLASLt E4M3 x E4M3 GEMM with FP32 accumulation and F32/BF16 output.
The CUDA path needs:

- a static activation-quantization kernel using the checkpoint's scalar
  `input_scale`;
- fused application of the activation and weight scales in the GEMM epilogue where
  supported, otherwise one bounded output-scale kernel;
- cached cuBLASLt descriptors/algorithms/workspace keyed by M/N/K, output type, and
  transpose layout;
- no descriptor creation, host synchronization, or allocation during CUDA graph
  capture;
- a fail-closed capability check for SM120+ and the required CUDA/cuBLASLt version;
- handling for M = 1, 4, 8, 32 and large prefill M without hidden shape padding bugs.

The local CUDA 13.1 headers already expose `CUDA_R_8F_E4M3` and the cuBLASLt
`A_SCALE_POINTER`, `B_SCALE_POINTER`, and `D_SCALE_POINTER` attributes, so an external
FlashInfer dependency is not required for the first native prototype. Availability in
the headers is not sufficient by itself: algorithm support for the exact E4M3/F32 or
E4M3/BF16 combinations still needs a real SM120 qualification matrix.

GGML stores a linear weight logically as `[K, N]` (`ne[0] = K`, `ne[1] = N`) with
each output row contiguous. This is compatible with a column-major `[K, N]` view for
the cuBLASLt call, but the exact leading dimensions and transpose flags must be tested
against the CPU reference before using it in a model.

### 4. Qualification gates

Before enabling the converter mode by default for this checkpoint:

1. Unit-test E4M3 special values, saturation, scale application, raw GGUF round trips,
   and Qwen projection reorders.
2. Compare native CUDA output with BF16 reference for every checkpoint shape at
   M = 1, 4, 8, 32, 512, including CUDA graph replay.
3. Run end-to-end deterministic logits/tokens against the current dequantized model.
4. Benchmark the actual llama-server path with DSpark at 1, 4, and 8 users, reporting
   aggregate throughput, per-user throughput, TTFT p50/p95, acceptance, errors, VRAM,
   and the exact FP8/NVFP4 kernel counters.
5. Keep `--fp8-as-q8` available as the portable fallback and compare it directly.

## External implementation reference

SGLang's SM120 implementation uses static activation quantization followed by
FlashInfer `bmm_fp8(..., backend="cublas")`, passing scalar activation and weight scales
to the GEMM. Its SM120 switch was intentionally small only because SGLang already had
the FP8 tensor representation, quantizer, scale-aware linear abstraction, FlashInfer
wrapper, and tests. llama.cpp currently has none of those FP8-weight layers, so copying
the two-line SM120 dispatch change without the supporting contract would be incorrect.
