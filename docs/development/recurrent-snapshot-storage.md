# Dynamic recurrent snapshot storage

## Shipped bounded layout

The adaptive Qwen hybrid path keeps the configured rollback limit separate
from the physically resident rollback depth. The recurrent tensors retain the
existing plane-major addressing:

```
row = plane * n_seq_max + cell
```

Only `1 + max_active_depth` planes are resident. A cap-zero context therefore
owns exactly the baseline recurrent state plane. Storage changes only at the
start of a logical decode batch:

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
fixed allocation.

This layout exactly serves the homogeneous server shapes used by DFlash
(all active slots at cap 0, 1, 2, or 3). A mixed-depth batch still allocates
`n_seq_max * (1 + max(depth_i))` rows.

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
