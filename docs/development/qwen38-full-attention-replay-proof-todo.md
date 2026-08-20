# Qwen3.8 full-attention replay-proof TODO

The 2026-08-20 guarded full-attention internal trace failed because CUDA
diagnostic route counters measure host dispatch/capture, while CUDA graph
replays bypass those counters. Comparing the first and restored executions'
route arrays is therefore invalid.

If this trace is resumed, do not weaken its route proof. Use a separate prefix
producer context for width 1 and width 8, serialize the complete prefix, destroy
the producer, then restore that prefix into fresh callback-free and callback
target contexts. This keeps both target graphs cold at the same prefix while
preserving one-context-at-a-time operation.

For the pinned layer-3 window:

- scalar cold execution: row 0 direct, row 1 capture, rows 2..7 replay;
- width-8 cold execution: first call direct, second restored call capture;
- after warmup, require two restored executions with graph stats
  `direct=0`, `capture=0`, `replay>0` and all host route counters zero;
- compare the complete 28-entry cold route vectors only between equal cold
  callback-free/callback phases;
- retain exact top-2, partial-state, full-state, boundary bytes, and repeat
  equality.

Expected selected cold routes are scalar VEC=32, SIGMOID_MUL=32,
SET_ROWS=64, and width-8 MMA_F16=16, SIGMOID_MUL=16, SET_ROWS=32; all
opposite/fallback markers remain zero. The prefix producer is necessary because
the old in-place `prepare_prefix()` warms the callback-free scalar graph and
made its route snapshot incomparable with the callback graph.

This TODO intentionally leaves the currently reviewed full-attention trace
implementation unchanged. The next active experiment is the smaller direct,
default-off layer-3 flash-attention selector candidate.
