# Experimental DFlash K-cache fusion

This branch adds an SM120-only CUDA fusion for the Qwen DFlash K-cache
injection path.  It collapses the following established ggml graph into one
kernel:

`RMS_NORM -> MUL -> NEOX ROPE -> FWHT128 -> SET_ROWS(Q4_0/Q8_0)`

The fused kernel still consumes the allocator-owned I64 row-index tensor.  It
does not derive or retain host-side cache positions, so unified multi-sequence
and iSWA cache placement keep the existing semantics.  The cache Hadamard
rotation is included before block quantization.

The dispatch is deliberately narrow:

- NVIDIA SM120 only;
- head dimension and rotated dimension exactly 128;
- full NEOX RoPE with no frequency-factor tensor;
- F32 projection and norm weight;
- unified cache storage;
- Q4_0 or Q8_0 cache for the FWHT path.

The existing no-FWHT direct write also supports F16, and this branch adds the
same guarded direct write for BF16.  Every unsupported shape or device uses the
original graph.  Set `GGML_CUDA_DFLASH_K_FUSION=0` to force that fallback for
an A/B comparison.

DSpark is intentionally not enabled here.  Its DSV4 graph currently expresses
full-head normal RoPE as a zero-length NOPE split plus RoPE and concat.  A safe
follow-up should first remove that redundant split/concat only when
`n_embd_head_nope == 0`, then add and test a separate NORMAL-RoPE matcher.

Build-only qualification used for this branch:

```powershell
cmake -S . -B build-fused-sm120 -G "Visual Studio 17 2022" -A x64 `
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 `
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-fused-sm120 --config Release --target test-backend-ops -j 8
```

Before promotion, run the focused `RMS_NORM_MUL_ROPE` backend cases on SM120,
then an end-to-end DFlash parity and N1/N4/N8 throughput A/B with the fallback
environment variable above.  This branch intentionally performed no GPU run.
