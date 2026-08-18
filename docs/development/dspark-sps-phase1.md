# DSpark SPS prefix planning (phase 1)

This opt-in server scheduler uses a measured server-step (SPS) cost surface and
DSpark's per-position confidence head to select a global target-verification
prefix for every newly drafted sequence in a decode tick.

Phase 1 is deliberately limited: DSpark still executes its full batched noise
block. The planner trims `slot.spec_draft` before `handle_last_sampled_token()`
builds the target batch, so the selected prefix is a real target-verification
cap, but it is not packed-ragged DSpark execution.

## Enabling it

Static behavior is unchanged when no profile is supplied.

```text
llama-server ... --spec-type draft-dspark --spec-draft-n-max 7 \
  --spec-draft-sps-profile dspark-sps.json
```

Use `--spec-draft-sps-shadow` to compute and report plans without trimming the
target verification batch. A requested profile that cannot be opened, parsed,
or validated stops model loading. A valid profile that does not cover a runtime
operating point uses the existing static prefix for that tick and increments
the fallback counter.

On the performance-integration branch, SPS and `--spec-draft-adaptive` are
separate scheduling policies and cannot be enabled together; model loading
fails closed if both are requested.

The planner is restricted to exactly one enabled speculative implementation,
`draft-dspark`. This prevents confidence or state from another draft method
from being mistaken for DSpark survival.

## Profile schema

```json
{
  "schema_version": 2,
  "max_draft_tokens_per_slot": 7,
  "entries": [
    {
      "context_tokens": 8192,
      "active_slots": 4,
      "total_verify_rows": 16,
      "cost_us": 1275.5
    }
  ]
}
```

Definitions:

- `context_tokens`: maximum generation context length among slots in the tick.
- `active_slots`: number of compatible generating slots in the tick.
- `total_verify_rows`: total target input rows, including one mandatory anchor
  per active slot and fixed checkpoint-replay rows.
- `cost_us`: measured target server-step time for this point.

All fields are required and unknown fields are rejected. Schema v2 uses exact
`active_slots` lookup and permits each active count to have its own physically
reachable row axis; every active count must still contain a complete
context-by-row grid. The loader continues to accept legacy schema v1 profiles,
which use a shared row axis and ceiling lookup for active slots.

Coordinates and costs
must be positive, coordinates must be unique, `total_verify_rows` must be at
least `active_slots`, and costs must be monotonic over comparable load points.
Runtime lookup uses an unambiguous context/row ceiling and never extrapolates.

## Objective and fallback

For slot `i` and prefix length `l`, expected additional useful tokens are:

```text
U_i(l) = sum(survival_i[position], position < l)
```

The dynamic program finds the best allocation at each total prefix-row count.
The final allocation maximizes:

```text
(base useful tokens + sum U_i(prefix_i)) / SPS cost_us(total verify rows)
```

Zero is always an available target-only choice. Positive prefixes respect
`--spec-draft-n-min`, and no prefix exceeds the slot's server context cap or
the actual DSpark result length. Invalid/missing confidence, invalid runtime
dimensions, or missing profile coverage triggers a static per-tick fallback.

## Telemetry

Existing draft token counters now use the actual target-offered prefix. The
Prometheus endpoint exports:

- `spec_decode_num_draft_tokens_per_pos_total{position=...}` (legacy name,
  actual target-offered denominator)
- `spec_decode_sps_plan_ticks_total`
- `spec_decode_sps_fallback_ticks_total`
- `spec_decode_sps_shadow_ticks_total`
- `spec_decode_sps_static_verify_rows_total`
- `spec_decode_sps_planned_verify_rows_total`
- `spec_decode_sps_executed_verify_rows_total`

`static - planned` is the hypothetical planner saving; `static - executed` is
the actual saving after fallback and shadow-mode handling. A decode tick that
may mix compatible prompt rows uses the static fallback because phase-1 SPS
profiles describe decode/verify-only target batches.

Per-slot timing output reports acceptance with the offered-per-position
denominator and the selected SPS prefix histogram.

## Phase 2: real packed-ragged execution

Phase 2 must move the allocation boundary before DSpark graph construction.
The exact code points are:

1. `common_speculative_impl_draft_dflash::draft()` in
   `common/speculative.cpp`: replace the current rectangular `n_block_tokens`
   construction with per-sequence lengths and a packed row map. The backend
   sampling output and confidence rows must be gathered through that map.
2. `common_speculative_draft_params`: carry a pre-draft execution length that
   is distinct from the post-draft target verify length. Phase 1 only carries
   `n_max` plus the produced survival vector.
3. `server_context_impl::pre_decode()` in
   `tools/server/server-context.cpp`: planning currently happens after
   `common_speculative_draft()`. A packed implementation needs a prior-tick
   survival estimate or a cheap confidence-only stage so allocation can occur
   before the draft graph.
4. The DSpark model graph in `src/models/dflash.cpp`: accept the packed row
   layout and ensure Markov predecessor edges never cross sequence boundaries.
5. CUDA graph capture/bucketing: capture a bounded set of total packed-row
   shapes and retain a correct uncaptured fallback for unseen shapes.

Until all five points have token-parity and sequence-isolation tests, benchmark
claims must say "target verification caps", not "compact DSpark execution".
