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

## End-to-end qualification marker

The DFlash model graph names only its real K-cache writes
`dflash_k_cache_write_<layer>`.  The CUDA backend requires that model-side tag
before emitting either of these release-visible messages:

- `DFlash-K FWHT cache fusion active`: the new eight-node FWHT plus
  quantized-cache kernel was dispatched;
- `DFlash-K model fusion fallback`: the tagged cache write reached ordinary
  `SET_ROWS` because fusion was disabled or a runtime guard did not match.

The first occurrence of each path is logged once per CUDA backend context, and
context shutdown reports FWHT-fused and fallback host-dispatch totals.  The
older five-node no-FWHT fusion does not increment the success metric, so F16 or
a disabled cache rotation cannot falsely qualify the new path.  Synthetic
backend-op tests do not carry the tag and therefore cannot produce a false
end-to-end success marker.  CUDA graph capture counts as one host dispatch;
later replays remain fused but do not increment the host-side total.

## Raw-cache byte validation

`GGML_CUDA_DFLASH_K_VALIDATE=1` enables a diagnostic, fail-fast self-check for
the eight-node Q4_0/Q8_0 path.  For each matching dispatch it executes the
exact normal fallback sequence (fused RMS/MUL/ROPE, standalone FWHT, ordinary
`SET_ROWS`), copies only the indexed cache rows to the host, poisons every byte
of those rows with the inverse reference value, then executes the direct fused
writer and compares the raw quantized bytes.  A mismatch logs the global cache
row, byte offset, values, and hashes before aborting.  The poison prevents an
incomplete fused write from inheriting reference bytes and falsely passing.

The focused cases use seven deterministic, permuted global row IDs in a
37-row cache, so they cover sparse/wrapped multi-sequence-style placement
rather than only dense rows 0 through 6.  Validation synchronizes the CUDA
stream and is incompatible with graph capture, so it must never be used for a
performance result:

```powershell
$env:GGML_CUDA_DFLASH_K_VALIDATE = '1'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
& .\build-fused-sm120\bin\Release\test-backend-ops.exe `
  -b CUDA -o RMS_NORM_MUL_ROPE `
  -p 'cache_type=(q4_0|q8_0).*use_fwht=1,cache_rows=37'
Remove-Item Env:GGML_CUDA_DFLASH_K_VALIDATE
Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS
```

A `DFlash-K raw cache validation PASS` line is the byte-identity signal.  The
ordinary numerical backend result alone is not sufficient for this gate.

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
cmake --build build-fused-sm120 --config Release --target llama-server -j 8
```

Before promotion, run the focused `RMS_NORM_MUL_ROPE` backend cases on SM120,
then an end-to-end DFlash parity and N1/N4/N8 throughput A/B with the fallback
environment variable above.  This branch intentionally performed no GPU run.
