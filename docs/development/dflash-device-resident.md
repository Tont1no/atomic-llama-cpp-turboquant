# DFlash device-resident feature injection

This private-fork fast path removes the host staging loop from the guarded
DFlash generation path. Selected target layer inputs are imported as
preallocated graph leaves by the draft context. DSV4 DSpark remains on the
host path, but batches active sequences into one encoder/injection pair. The
DFlash device graph performs:

1. feature concatenation;
2. the DFlash `fc` projection and encoder RMS normalization;
3. the existing per-layer K/V projection and cache writes.

The three steps are submitted as one draft decode graph. The target and draft
contexts are synchronized at the ownership boundaries, but no feature tensor is
copied through host memory on the guarded path.

## Guarded scope

The fast path is used only when all of the following are true:

- the target batch contains token IDs and the complete logical batch fits in
  the target ubatch limit; a single target UBatch stays zero-copy, while a
  hybrid/recurrent split is assembled in same-device staging;
- the logical target and draft batches each fit in one ubatch;
- the draft uses unified KV and its actually prepared memory context contains
  the complete logical batch in exactly the retained target-ubatch order;
- target activations and draft KV metadata use the same retained ubatch order;
- the complete target position vector and sequence ID match for every row;
- Qwen3.6's target position is the text-only IMROPE pattern `[p,p,p,0]`,
  which is validated before projecting `p` to the NEOX DFlash draft;
- every selected layer input is contiguous F32 with the expected shape;
- every source tensor and the DFlash fusion weight are on the same CUDA device;
- positions and sequence IDs match the target batch exactly.

If any check fails, `common/speculative.cpp` executes the previous encoder and
host injection path. Target host extraction is deferred only for eligible
bounded batches. If the device path is rejected, all retained target
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

Target ubatch metadata is captured directly inside the decode loop before the
memory context advances. Four-axis positions use the plane-major layout
`[p0 rows][p1 rows][p2 rows][p3 rows]`. This avoids stale positions when a
same-shaped graph is reused. The graph result's owned ubatch parameters are
also refreshed on reuse so diagnostics and guarded host materialization observe
the current decode. Multimodal target rows never enter the IMROPE-to-NEOX
projection. Separate plane-major scratch storage is used for embedding batches,
because the public batch allocator otherwise owns only one position per row.
The external injection call itself carries no token or embedding payload: its
batch contains only positions and sequence metadata, while the retained target
tensors are the feature payload. This avoids both the persistent
`n_batch * n_embd_enc` dummy allocation and the per-call
`n_tokens * n_embd_enc` host copy.

If Qwen's hybrid/recurrent memory splits an uneven multi-sequence verification
batch, each actual target UBatch is copied immediately into a target-owned CUDA
staging buffer at its logical row offset. Full IMROPE metadata is written in the
same order. The external graph imports only the populated prefix, while the
persistent capacity-tensor pointer remains stable for graph reuse. Allocation,
layout, backend, or copy preflight failures fall closed to the per-UBatch host
path; a partial stage is never published as device-ready. Uniform one-UBatch
decodes retain the original zero-copy path.

When device injection is unavailable, uniform verification batches can still
inject every active sequence in one host encoder/decode pair. The feature
encoder is row-independent and the cache decode receives the original position
and sequence ID for every row. Prompt, mixed-logit, and oversized batches keep
the established chunked fallback.

Guard failures are reported once per low-cardinality reason and batch-shape
bucket (`single`, `verify` up to 16 rows, or `prompt`) to keep production logs
useful without per-token noise: `common_batch_guard`, `guard`,
`target_ready`, `row_set`, `device_backend`, `tensor_shape`, or
`draft_ubatch`. Shape and prepared-ubatch failures include bounded dimensions or
row indices needed to identify the violated guard. The opted-in target also
reports one retention summary per shape bucket, including the actual ubatch
count and first four sizes, split point, deferred-extraction state, and retained
row/graph metadata. This keeps a prompt-prefill rejection from hiding the first
generation verification result.

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
