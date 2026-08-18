# DFlash device-resident feature injection

This private-fork fast path removes the host staging loop from the steady-state
DFlash and DSpark generation path. Selected target layer inputs are imported as
preallocated graph leaves by the draft context. The draft graph performs:

1. feature concatenation;
2. the DFlash `fc` projection and encoder RMS normalization;
3. the existing per-layer K/V projection and cache writes.

The three steps are submitted as one draft decode graph. The target and draft
contexts are synchronized at the ownership boundaries, but no feature tensor is
copied through host memory on the guarded path.

## Guarded scope

The fast path is used only when all of the following are true:

- the target batch contains token IDs and the complete logical batch fits in a
  single retained target graph;
- the logical target and draft batches each fit in one ubatch;
- the draft uses unified KV and its actually prepared memory context contains
  the complete logical batch in exactly the retained target-ubatch order;
- target activations and draft KV metadata use the same retained ubatch order;
- every selected layer input is contiguous F32 with the expected shape;
- every source tensor and the DFlash fusion weight are on the same CUDA device;
- positions and sequence IDs match the target batch exactly.

If any check fails, `common/speculative.cpp` executes the previous encoder and
host injection path. Target feature extraction is deferred only for eligible
single-ubatch batches. If the device path is rejected, all retained target
tensors are materialized together before continuing the fallback, preserving
the established recurrent output reordering behavior.

## API and implementation

- `llama_set_embeddings_layer_inp_device()` opts a target context into deferred
  extraction for eligible batches.
- `llama_decode_dflash_features()` attempts the guarded device graph. `false`
  means unsupported and is not an error; `true` means the graph was attempted
  and its decode status is returned through `ret`.
- `llm_graph_params::external_layer_inputs` carries the foreign preallocated
  leaves and participates in graph reuse compatibility.
- `src/models/dflash.cpp` builds the external concat, fusion, normalization, and
  existing DFlash or DSV4 injection graph.

The raw target op tensors are never attached to the draft graph. Each source is
represented by a detached `GGML_OP_NONE` leaf with the same CUDA buffer, data
pointer, shape, and strides. Before allocation, the draft context walks all
reachable nodes and ancestors. It rejects the graph if a raw target tensor is
reachable or any detached source leaf is missing, and logs the graph node/leaf
counts with `target_ancestors=0` on the first validated graph.

Guard failures are reported once per low-cardinality reason to keep production
logs useful without per-token noise: `common_batch_guard`, `guard`,
`target_ready`, `row_set`, `device_backend`, `tensor_shape`, or
`draft_ubatch`. Shape and prepared-ubatch failures include bounded dimensions or
row indices needed to identify the violated guard.

Changed source files:

- `common/speculative.cpp`
- `src/llama-context.cpp`
- `src/llama-context.h`
- `src/llama-ext.h`
- `src/llama-graph.h`
- `src/models/dflash.cpp`

## Windows SM120 build

```powershell
cmake -S . -B build-sm120-vs17 -G "Visual Studio 17 2022" -A x64 `
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF
cmake --build build-sm120-vs17 --config Release --target llama-server -j 12
```

The generated server is
`build-sm120-vs17/bin/Release/llama-server.exe`.

## Remaining risks and next work

- This is compile-verified but requires GPU correctness and throughput A/B
  testing before deployment.
- Non-unified draft KV and the DSV4 DSpark path intentionally use the host
  fallback. Unified iSWA is supported only when its actually prepared memory
  context remains one complete, exactly ordered ubatch. If cache fragmentation
  makes iSWA fall back to a split/reordered layout, the attempt is rejected
  before cache application and graph execution, and the host path takes over.
- The imported graph-leaf behavior depends on the ggml scheduler preserving an
  already allocated tensor's buffer assignment. The guards validate allocation
  and device identity, while the GPU test must verify scheduler execution.
- Synchronizing the draft context before returning protects the target compute
  buffers but introduces a host-side barrier. A later implementation can replace
  it with cross-context CUDA events after the correctness baseline is proven.
- A prepared iSWA layout rejected by the actual-ubatch preflight currently
  increments the draft context's queued-token performance counter before the
  host fallback decodes the same rows. This affects context perf accounting for
  those rare fallback attempts, not output correctness or server request metrics.
- Concatenation and per-layer K/V projections remain separate CUDA operations.
  The next performance step is a stacked K/V materialization kernel similar to
  the fused SGLang implementation.
