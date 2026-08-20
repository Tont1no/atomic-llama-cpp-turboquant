# Qwen3.8 DSpark production qualification

This runner qualifies one production tier at a time without ever overlapping
llama servers. `Invoke-Qwen38DsparkQualification.ps1` is the only supported
entry point. It validates the JSON plan on CPU, acquires the exclusive GPU
guard exactly once, and runs the target-only, static DSpark, and SPS servers
sequentially inside that one guarded child. P4 additionally includes a matched
static cap-3 arm before the existing best-product cap-2 arm, so SPS cap-3 has
both a causal matched comparison and a production-best comparison.

Dynamic recurrent snapshots are deliberately out of scope: every speculative
arm passes `--no-spec-draft-dynamic-rs` and `--no-spec-draft-adaptive`, and the
child environment sets `LLAMA_ARG_SPEC_DRAFT_DYNAMIC_RS=0`. The target-only arm
has no speculative CLI flags at all.

## Required preparation

Create a fresh SPS-v2 profile with the current final server before qualifying a
tier. The profile directory must also contain the profiler `manifest.json`.
The qualification runner rejects a profile whose manifest does not match the
current server, adjacent DLL bundle, target/draft model hashes, GPU identity,
parallel/context/batch settings, Q8 KV types, fixed-RS policy, or tier depth.

The checked qualification plan is:

`C:\Users\pasca\Documents\GitHub\Ai-Loader\benchmarks\qwen38-dspark-production-qualification-20260819\qualification-plan.json`

## Fresh P1 target-only trace for the parity diagnostic

`Invoke-Qwen38TargetTrace.ps1` is a separate, bounded evidence path. It reuses
the qualification runner's exact server arguments, warmup/measured waves and
native SSE contract, but selects only the non-speculative P1 target arm. It
does not accept or load a draft model or SPS profile and never starts a static
or SPS arm. One outer exclusive guard owns the one server lifecycle.

The output leaf is deliberately fixed to `p1-final-v5`, matching the parity
diagnostic's reviewed input layout. Use a fresh parent directory so the stale
historical `qualification\p1-final-v5` evidence is preserved:

```powershell
$root = 'C:\Users\pasca\Documents\GitHub\Ai-Loader'
$final = Join-Path $root '.codex-deploy\llama-performance-final'
$fresh = Join-Path $root 'benchmarks\qwen38-dspark-production-qualification-20260819\qualification-current-20260820\p1-final-v5'

& (Join-Path $final 'scripts\Invoke-Qwen38TargetTrace.ps1') `
  -PlanPath (Join-Path $root 'benchmarks\qwen38-dspark-production-qualification-20260819\qualification-plan.json') `
  -ServerPath (Join-Path $final 'build-dspark-rs-integrated-cuda132-fresh\bin\llama-server.exe') `
  -ModelPath 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf' `
  -OutputDir $fresh `
  -RuntimeLabel 'CUDA13.2-current-P1-target-only-parity-trace'
```

The runner produces `manifest.json`, `arms\target-only\command.json`,
`waves.json`, the arm `receipt.json`, `summary.json`, `validation.json`, and a
guard/postflight receipt. The manifest binds the current server, every adjacent
DLL, scrubbed `LLAMA_`/`GGML_`/`CUDA_` environment contract, model, runner,
wrapper and guard hashes. `validation.json` lists the candidate immutable
qualification identity and manifest/model/waves/token/command/receipt hashes.
Those candidates must be reviewed and copied into
`Invoke-Qwen38ParityDiagnostic.ps1`; the diagnostic intentionally continues to
reject new evidence until that explicit trust update is made.

### Row-invariant LM-head production bundle

The operator-passing NVFP4 LM-head candidate was rebuilt into the exact
target-trace server path on 2026-08-20 with the bundled CUDA 13.2 compiler,
SM120a and `GGML_CUDA_DIAGNOSTIC_ROUTES=OFF`:

- server SHA-256: `3050ffd92c700f754bc97c1b97d8b8f74c24c291e5f91516408dd4aa7c407dc6`
- canonical 10-DLL runtime bundle SHA-256:
  `429fec7fbc3ed32cb498433d6da760a4888fb286ebd09c4b4b32540aa4bd4c1c`
- production `ggml-cuda.dll` SHA-256:
  `839819a347ac36e29f287067b9d4673cb83a8425a19e6e42ccc4dbd372eb5876`

The production cache records the diagnostic option as `OFF`, its `build.ninja`
contains no diagnostic compile definition, and the linked CUDA DLL contains
neither diagnostic route lookup name. The separate guarded diagnostic build is
configured `ON` and retains both names, proving both sides of the compile-time
boundary.

The next fresh target-only trace must use a new parent because target traces
never overwrite or resume evidence:

```powershell
$root = 'C:\Users\pasca\Documents\GitHub\Ai-Loader'
$final = Join-Path $root '.codex-deploy\llama-performance-final'
$fresh = Join-Path $root 'benchmarks\qwen38-dspark-production-qualification-20260819\qualification-row-invariant-20260820\p1-final-v5'

& (Join-Path $final 'scripts\Invoke-Qwen38TargetTrace.ps1') `
  -PlanPath (Join-Path $root 'benchmarks\qwen38-dspark-production-qualification-20260819\qualification-plan.json') `
  -ServerPath (Join-Path $final 'build-dspark-rs-integrated-cuda132-fresh\bin\llama-server.exe') `
  -ModelPath 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf' `
  -OutputDir $fresh `
  -RuntimeLabel 'CUDA13.2-SM120-row-invariant-LM-head-P1-target-only'
```

This command is a target-only correctness trace, not a performance or DSpark
promotion run. Review its new manifest, validation and guard receipt before
updating any parity-diagnostic trust anchors.

Use a new output directory for every run. Partial runs are intentionally not
resumed: performance comparisons are valid only when every arm comes from the
same uninterrupted guarded session.

## Copyable invocation

```powershell
$root = 'C:\Users\pasca\Documents\GitHub\Ai-Loader'
$final = Join-Path $root '.codex-deploy\llama-performance-final'
$tier = 'P4' # P1, P4, or P8
$profile = Join-Path $root "benchmarks\qwen38-dspark-production-qualification-20260819\profiles\$($tier.ToLower())\profile.json"
$output = Join-Path $root "benchmarks\qwen38-dspark-production-qualification-20260819\qualification\$($tier.ToLower())-$(Get-Date -Format yyyyMMdd-HHmmss)"

& (Join-Path $final 'scripts\Invoke-Qwen38DsparkQualification.ps1') `
  -PlanPath (Join-Path $root 'benchmarks\qwen38-dspark-production-qualification-20260819\qualification-plan.json') `
  -Tier $tier `
  -ServerPath (Join-Path $final 'build-final-cuda132-vs17\bin\Release\llama-server.exe') `
  -ModelPath 'C:\Users\pasca\Desktop\AI-Loader-Main-Modelle-2026-08-16\Qwen3.8-27B-NVFP4-FP8-native.gguf' `
  -DraftModelPath (Join-Path $root 'models\draft\Qwen3.8-27B-DSpark-Q8_0.gguf') `
  -ProfilePath $profile `
  -OutputDir $output `
  -RuntimeLabel "CUDA13.2-SM120-Qwen38-$tier-fixed-RS"
```

Guard defaults are hard safety limits: GPU 0, at least 18 GiB free system RAM,
at least 5000 MiB free VRAM, at most 27000 MiB used VRAM, and 250 ms polling.
The wrapper allows stricter values but refuses weaker ones. It also refuses an
existing `llama-server`, a busy port, a mismatched plan path/hash, or a nonempty
output directory.

The 18 GiB RAM floor is deliberate: the final guarded profile runs retained
about 19.7–19.9 GiB free, so a 20 GiB admission floor rejected the verified
configuration before startup. Values below 18 GiB remain prohibited.

## Workload and evidence

Every arm uses the same native streaming `/completion` requests:

- numeric prompt `[1]` repeated 736 times;
- 256 output tokens, temperature 0, seed `20260819 + client_index`;
- `ignore_eos=true`, `cache_prompt=false`, `stream=true`,
  `return_tokens=true`;
- one excluded warmup wave and six measured barrier-synchronized waves.

The SSE client stores every raw token ID and requires exactly 256 partial-event
tokens, one terminal event, `stop_type=limit`, no truncation, 736 evaluated
prompt tokens, zero reused prompt tokens via `timings.cache_n`, and matching
timing counts. Native `tokens_cached` is retained diagnostically but is not a
prompt-cache-reuse counter: server source sets it to the slot's current context
length at final response time. The native `is_begin` marker flushes HTTP
headers but is converted to JSON null and never serialized as an SSE event;
TTFT nevertheless includes that header wait because timing begins before the
request is sent. Raw token arrays
and content must match for every client and wave within an arm and across all
arms. Hashes supplement the arrays; they never replace exact comparison.

An output-contract failure is written into `run-state.json` and guard stderr as
a content-free diagnostic. It lists every failed predicate plus observed raw
token count, terminal/timing keys, stop/truncation and numeric accounting fields;
it never includes generated token IDs or generated content.

Metrics are read only when `requests_processing` and `requests_deferred` are
zero. The runner checks exact generated/prompt token deltas, zero cached prompt
tokens, target-only speculative counters at zero, real DSpark work in DSpark
arms, SPS planning with zero fallback/shadow, planned rows equal executed rows
and not above static rows, and zero recurrent-snapshot resizes during measured
waves.

Primary throughput is aggregate goodput:

`parallel * 256 / common wave wall time`

Backend decode throughput is reported separately as the Prometheus counter
delta `tokens_predicted_total / tokens_predicted_seconds_total`. TTFT p50/p95,
per-request rate, acceptance, draft counters, SPS counters, and fixed snapshot
depth gauges are also retained.

Promotion requires exact parity, positive median paired SPS gain, the lower
bound of a deterministic 10,000-resample paired bootstrap 95% interval above
zero, and no more than 10% TTFT-p95 regression. P4 must beat both target-only
and the cap-2 best-product static arm; the cap-3 matched comparison is reported
separately. A non-promoted run can still complete successfully and reports
`status: not-promoted`.

Artifacts are written atomically under `manifest.json`, `run-state.json`,
`arms/<arm>/`, and `summary.json`. After the guard releases, the wrapper adds
`guard-receipt.json` and links its unique stdout/stderr paths. Any HTTP, SSE,
server, lease, Job Object, counter, parity, timeout, cleanup, or port-reuse
violation aborts the remaining arms.

The guard receipt includes peak used VRAM, minimum free VRAM, minimum free
system RAM, GPU identity, and guarded runtime. These outer samples are the
authoritative memory envelope for the complete sequential run.

## CPU-only tests

These tests never launch llama-server or query the GPU:

```powershell
python -m unittest tests.test_qwen38_dspark_qualification -v
pwsh -NoProfile -File tests/test_qwen38_target_trace_wrapper.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests/test_qwen38_target_trace_wrapper.ps1
pwsh -NoProfile -File tests/test_qwen38_dspark_qualification_wrapper.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests/test_qwen38_dspark_qualification_wrapper.ps1
```
