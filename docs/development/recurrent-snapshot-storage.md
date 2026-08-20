# Dynamic recurrent snapshot storage

## Shipped bounded layout

The adaptive Qwen hybrid and pure DSpark packed paths keep the configured rollback limit separate
from the physically resident rollback depth. The recurrent tensors retain the
existing plane-major addressing:

```
row = plane * n_seq_max + cell
```

Only `1 + max_active_depth` planes are resident once a shrink settles. A
cap-zero context therefore owns exactly the baseline recurrent state plane.
Logical execution depth changes immediately, while physical shrink requires
the same safe target in two distinct server scheduling epochs. Target or load
changes re-arm the dwell, so transient admission waves do not cause repeated
20-70 ms reallocations. Required growth remains immediate and N1 startup does
not wait. Storage changes only at the start of a logical decode batch:

1. synchronize the scheduler;
2. materialize a pending rollback plane into plane zero if the new allocation
   would remove it (shared tails retain the plane for one tick so normal
   copy-on-write detachment happens first);
3. invalidate graphs and the scheduler, which retain tensor pointers;
4. allocate the new tensors, copy retained planes, and release the old buffer;
5. reserve graphs using the new resident depth.

Snapshots for sequences absent from a batch keep the allocation from
shrinking below their last valid depth. Malformed metadata fails closed to the
configured maximum. Non-Qwen and non-dynamic contexts retain the original
fixed allocation. A pure DSpark chain enables the dynamic layout for fixed
packed proposals, SPS planning, SPS shadow planning, and SPS recorder
force-caps when explicitly enabled with `--spec-draft-dynamic-rs`. SPS and
adaptive planning remain mutually exclusive at server
startup. DFlash retains its adaptive-only opt-in, and mixed speculative chains
retain fixed storage. Pure DSpark defaults to a fixed startup allocation until
resize headroom is qualified.

The server's adaptive load estimate includes every assigned non-idle slot and
the inference-slot demand still queued for admission. Prompt, waiting-child,
and newly started slots therefore contribute before they reach `GENERATING`.
The load estimate selects the speculative cap and identifies scheduling
epochs, but it never lowers the batch-derived correctness depth.

The server also supplies an explicit per-row `llama_batch::rs_depth` contract.
Ordinary prompt, prefill, multimodal prompt, and target-only rows request depth
zero. Every row in a speculative verification span requests the number of
draft tokens in that span, clamped to the configured limit. Mixed batches use
the maximum explicit verification depth, so prompt length can no longer be
mistaken for rollback demand. A missing or invalid depth array remains a
fail-closed request for the configured maximum for external callers.
Because the experimental pointer extends the public by-value batch structure,
the fork's library and all callers must be rebuilt together.

Resize count, cumulative synchronized resize time,
resident/configured/required/pending depth, and stable-epoch count are available through
`llama_get_recurrent_resize_stats()` and the `/metrics` recurrent snapshot
series.

This layout exactly serves the homogeneous server shapes used by DFlash
(all active slots at cap 0, 1, 2, or 3). A mixed-depth batch still allocates
`n_seq_max * (1 + max(depth_i))` rows.
It does **not** allocate `sum_i(1 + depth_i)` rows: inactive configured slots
are part of every resident plane. Size production tiers accordingly. For
example, a wider P8 tier should use a lower configured
`--spec-draft-n-max` than a P4 tier when both must fit the same recurrent-state
budget. The metric pair `recurrent_snapshot_resident_depth` and
`recurrent_snapshot_configured_depth` makes that distinction observable.
For the measured Qwen layout whose recurrent state costs 149.625 MiB per
configured sequence and resident plane, `P4/nmax=3` and `P8/nmax=1` each cap
the rectangular allocation at about 2394 MiB.

Growth currently allocates and clears the complete new buffer before copying
and releasing the old buffer. Qualification must therefore reserve transient
headroom for `old buffer + new buffer`; a cap-zero P8 process growing to cap
one temporarily retains its 1197 MiB base while allocating the 2394 MiB depth-
one buffer. If that headroom is not proven, keep the default fixed allocation
(or pass `--no-spec-draft-dynamic-rs`).

## Exact ragged layout follow-up

To reach `sum_i(1 + depth_i)` for mixed-depth slots without adding graph-time
reallocation, split the cache into:

- permanent base tensors `r_l/s_l` with `n_seq_max` rows;
- snapshot sidecars `r_rs_l/s_rs_l` with `sum_i depth_i` rows;
- a dense `(seq_id -> offset, capacity)` table.

At the same synchronized between-tick boundary, repack only the sidecars and
materialize pending rollback rows into the permanent base. Qwen DeltaNet then
writes snapshot zero to the base and writes the remaining snapshots with
`ggml_set_rows` using host-provided sidecar row indices. CUDA supports this
scatter operation. Graph reuse must include the sidecar generation or refresh
the index input on every reuse.

Shared sequences need either reference-counted sidecar entries or a strict
fallback to the rectangular layout. State save/load must serialize the logical
current row after any pending rollback is materialized. This second layout is
intentionally separate because it changes graph topology and the state-copy
contract; it is not required for the homogeneous eight-slot cap-zero target.
