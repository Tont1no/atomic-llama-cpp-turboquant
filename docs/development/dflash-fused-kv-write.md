# Experimental DFlash K-cache fusion

This branch adds an SM120-only CUDA fusion for the Qwen DFlash K-cache
injection path.  It collapses the following established ggml graph into one
kernel:

`RMS_NORM -> MUL -> NEOX ROPE -> FWHT128 -> SET_ROWS(Q4_0/Q8_0)`

## Integration boundary

This opt-in branch is based on the stable Qwen performance integration at
`15f4cf24f`. It replays only the Fused-K CUDA runtime, dispatch marker,
raw-cache validator, deterministic harness, scheduler-reservation fix, and
their documentation. It does **not** include the earlier stacked K/V
projection commit (`31c9843a9`), its GGUF metadata, or its converter changes.
The Native-FP8/DSpark converter and runtime from the stable integration remain
unchanged.

The Fused-K source series was GPU-correctness qualified before this replay.
This integration replay itself is CPU-tested and CUDA-compiled only; it does
not start a server, load a model, or execute a CUDA test.

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
direct fused writer first, then executes the exact normal fallback sequence
(three-op fused, two-op RMS/MUL fused, or individual RMS/MUL/ROPE according to
the same runtime fusion priority, standalone FWHT, ordinary `SET_ROWS`) and compares only
the indexed raw quantized cache rows. It also snapshots the complete cache and
requires every unaddressed row to remain byte-identical after both direct
passes and the fallback. Direct-first ordering is required because
the graph allocator may reuse the dead RMS input allocation for a later FWHT
intermediate; fallback-first would then corrupt the input before the diagnostic
direct launch. The direct writer is repeated after poisoning its destination
with inverse bytes, and the fallback is run after another poison, so both paths
must deterministically overwrite every compared byte. A mismatch logs the
global cache row, byte offset, values, and hashes before aborting.

The focused cases use deterministic, permuted global row IDs in both a small
seven-token/37-row cache and a model-like 286-token/313-row cache. They cover
sparse/wrapped multi-sequence-style placement, transient source allocation and
allocator reuse rather than only dense rows. Validation synchronizes the CUDA
stream and is incompatible with graph capture, so it must never be used for a
performance result:

```powershell
$env:GGML_CUDA_DFLASH_K_VALIDATE = '1'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
& .\build-fused-sm120\bin\Release\test-backend-ops.exe `
  -b CUDA0 -o RMS_NORM_MUL_ROPE `
  -p 'cache_type=(q4_0|q8_0).*use_fwht=1,cache_rows=(37|313)'
Remove-Item Env:GGML_CUDA_DFLASH_K_VALIDATE
Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS
```

A `DFlash-K raw cache validation PASS` line is the byte-identity signal.  The
ordinary numerical backend result alone is not sufficient for this gate.

The model-like fixture also makes its FP32 inputs, positions, and untouched
cache rows deterministic. Its separate CPU-versus-CUDA semantic check uses a
quantization-local oracle instead of the default `1e-7` NMSE: dequantized
untouched rows must match exactly, and each written Q4_0/Q8_0 block may differ
by at most one FP16 scale ULP and one quantization level, with written-row NMSE
capped at `5e-6`. This admits only boundary amplification from the different
CPU and CUDA FP32 pipelines; it does not replace the exact CUDA
direct-versus-fallback byte comparison or the deterministic logits/KV harness.

`CUDA0` is intentional: `-b CUDA` does not match the numbered backend name and
can report success after running zero focused cases.

## Deterministic multi-sequence regression

`test-dflash-fusion-determinism` runs the fallback and fused implementations
sequentially in one process with independent CUDA contexts. It fixes both the
target and DFlash sampler seeds and directly constructs the same ragged N4 and
N8 prompt/verification batches, positions, and sequence IDs in each arm. The
test compares draft tokens, bitwise draft and target logits, accepted
prefix/final tokens, and serialized per-sequence draft KV state after both the
noise-block draft and verified-feature injection.

The kill switch controls only the new tagged eight-node FWHT writer; the older
five-node fusion remains identical in both arms. The harness also rejects a
false pass unless OFF observes the tagged fallback and ON observes both the
exact FWHT-fusion marker and raw-cache validation with no tagged fallback.
The OFF arm intentionally performs the first DFlash feature injection without
a warm-up decode, covering scheduler reservation before metadata-only external
target-feature aliases are attached.

Run it with the real target and DFlash GGUFs on SM120. The draft K cache must be
Q4_0 or Q8_0; run both commands when qualifying both formats:

```powershell
& .\build-fused-sm120\bin\Release\test-dflash-fusion-determinism.exe `
  -m <target.gguf> -md <dflash.gguf> --spec-type draft-dflash `
  -ngl 999 -ngld 999 -ctkd q4_0 -ctvd q4_0

& .\build-fused-sm120\bin\Release\test-dflash-fusion-determinism.exe `
  -m <target.gguf> -md <dflash.gguf> --spec-type draft-dflash `
  -ngl 999 -ngld 999 -ctkd q8_0 -ctvd q8_0
```

The final acceptance signal is `DFlash deterministic regression PASS` with
nonzero ON `active` and `raw` marker counts. This bounded fixture exercises
ragged and permuted unified-cache allocation, but does not force a complete
physical iSWA ring wrap; that remains a separate long-context qualification.

DSpark is intentionally not enabled here.  Its DSV4 graph currently expresses
full-head normal RoPE as a zero-length NOPE split plus RoPE and concat.  A safe
follow-up should first remove that redundant split/concat only when
`n_embd_head_nope == 0`, then add and test a separate NORMAL-RoPE matcher.

Build-only qualification used for this branch:

```powershell
$Cuda132Root = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\cuda-13.2\toolkit'
cmake -S . -B build-fused-sm120 -G "Visual Studio 17 2022" -A x64 `
  -T "cuda=$Cuda132Root" -DCUDAToolkit_ROOT="$Cuda132Root" `
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 `
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-fused-sm120 --config Release --target test-backend-ops -j 8
cmake --build build-fused-sm120 --config Release `
  --target test-dflash-fusion-determinism -j 8
```

Before promotion, run the focused `RMS_NORM_MUL_ROPE` backend cases and the
deterministic in-process regression on SM120, then run N1/N4/N8 throughput A/B.
Fresh-server output hashes are not a correctness oracle because request
admission and batching can vary between runs. This branch intentionally
performed no GPU run.
