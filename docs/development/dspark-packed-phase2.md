# DSpark packed-ragged proposal execution (phase 2)

## Scope

This slice removes the rectangular DSpark noise-row requirement when active
sequences have different proposal caps. It does not change DFlash.

The common speculative path resolves every source sequence to an exact width,
builds compact source-order rows, and refuses any plan that exceeds the trained
DSpark gamma or the draft context's configured row limit. A zero-width entry
remains target-only and contributes no draft row.

The model graph derives a validated `indptr` from the row sequence metadata.
Consecutive source-order sequences with the same width share the original
vectorized Markov head. Width transitions form separate groups whose outputs
are appended in source order. This preserves per-sequence Markov anchors,
confidence rows, sampling indices, and the server's existing acceptance,
checkpoint, rollback, and target-KV flow.

For the Qwen3.8 DSpark draft's unified iSWA cache, the preferred memory path
keeps the whole ragged source-order set in one compact ubatch, making the mixed
Markov graph reachable. If unified cache placement fails (or unified KV is
disabled), the safe fallback emits adjacent equal-width complete-run groups,
then individual complete runs. It never wave-splits a Markov chain. A failure
in any fallback group rolls back proposal KV rows for the entire logical batch.

## SPS and static behavior

SPS still observes confidence/survival only for rows that were actually
proposed, and its selected target-verification prefix remains bounded by that
survival vector. Invalid SPS plans keep the existing static verification
fallback. DSpark without SPS uses the same packed execution and fixed caps.

The SPS planner is intentionally not moved ahead of the DSpark pass: current
survival is produced by that pass, so using it to choose the same pass's row
count would be circular. This phase saves rows selected by pre-existing hard or
per-sequence caps; it does not claim that a post-draft SPS prefix retroactively
reduces already-computed noise rows.

## Correctness guards

- each row belongs to exactly one non-negative sequence;
- every sequence occupies one contiguous source-order run;
- every positive run width is in `[1, trained_gamma]`;
- the full packed transaction fits in the configured draft row limit;
- graph reuse includes the exact packed `indptr`, preventing same-total-row
  layouts with different Markov topology from sharing a graph;
- every prepared ubatch is checked against the logical source-order rows before
  it is applied; a later mismatch or compute failure rolls back all groups;
- equal-width static DSpark remains one vectorized Markov group.

## CUDA graph boundary

Exact packed shapes are reusable, but this slice does not pad to larger
total-row tiers. Padding real DSpark rows would write draft KV entries and alter
Markov/output semantics unless a dedicated non-writing dummy-row contract is
added across batching, attention, and memory. CUDA graph tiering is therefore a
follow-up optimization, not silently approximated here.

Scheduler/Fit reservation does include two valid DSpark worst-case probes: a
maximum equal-gamma group and a near-dense alternating-width ragged topology.
This covers the Markov vocabulary/confidence buffers and mixed-group concat
temporaries that the generic over-wide rectangular prompt probe must skip.

## Validation

`test-dspark-packed` and `test-batch-alloc` cover caps `0/1/2/4/7`, N1/N4/N8 mixed plans,
source-order grouping, gamma and ubatch bounds, non-contiguous sequences,
multi-owner rows, missing metadata, invalid unique-sequence counts, the unified
single-ubatch throughput split, and complete-run fallback splitting.

The implementation is validated compile-only on CPU and CUDA 13.2 / SM120.
No server, model, or GPU execution is part of this phase.
