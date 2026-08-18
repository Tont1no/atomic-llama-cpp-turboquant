# DSpark SPS profiler orchestration

This directory contains the CPU-only planning layer and the implemented
server-side DSpark SPS recorder. The planning and validation commands do not
execute a model. The Python GPU runner remains intentionally disabled.

## Plan contract

Generate a deterministic plan:

```powershell
python .\scripts\dspark_sps_profile.py plan `
  --output .\benchmarks\dspark-sps-profile\plan.json `
  --active-slots 1,2,4,8 `
  --context-ceilings 2048,4096,8192 `
  --prefix-caps 0,1,2,3,7 `
  --max-draft-tokens-per-slot 7 `
  --repetitions 4 `
  --samples-per-cell 64 `
  --warmup-shape-runs 3 `
  --seed 20260818
```

Validate it independently before any GPU lease is acquired:

```powershell
python .\scripts\dspark_sps_profile.py validate `
  --plan .\benchmarks\dspark-sps-profile\plan.json
```

The JSON file has no timestamp and is byte-deterministic for the same inputs.
Unknown fields, duplicate JSON members, an invalid hash, odd repetitions, and
out-of-domain caps fail closed.

The schedule starts with a SHA-256 ordering of every profiling cell. For each
pair of repetitions, the first repetition uses a rotation of that order and
the second uses its exact reverse. Every cell therefore occurs exactly once in
every repetition and has complementary early and late positions in each pair.

`prefix_cap` is converted to an exact global target verification row count:

```text
total_verify_rows = active_slots * (1 + prefix_cap)
```

The anchor row is included. Row axes are separate for active 1, 2, 4, and 8.
For seven draft tokens per slot and caps 0, 1, 2, 3, and 7, they are:

```text
active 1: 1, 2, 3, 4, 8
active 2: 2, 4, 6, 8, 16
active 4: 4, 8, 12, 16, 32
active 8: 8, 16, 24, 32, 64
```

## Profile schema boundary

Legacy profile schema v1 requires one common Cartesian row axis. That is not a
valid representation of bounded per-slot DSpark depth: a common axis
containing 64 rows would also require the physically unreachable coordinate
`active_slots=1,total_verify_rows=64`.

The recorder writes schema v2 with these rules:

- `active_slots` is an exact-match lookup dimension.
- Each active-slot value has its own row axis.
- Context and row lookup use conservative ceilings inside that exact active
  level.
- Missing coverage selects the static scheduler.
- Monotonic validation applies to all comparable measured physical points.

Do not fill unreachable cells and present them as measurements. A temporary
synthetic envelope, if ever used for compatibility, must be identified as
synthetic in the audit sidecar and must not be confused with raw samples.

## Server recorder

The server exposes these record-only controls:

```text
--spec-draft-sps-record FILE
--spec-draft-sps-record-identity SHA256
--spec-draft-sps-force-verify-rows N
--spec-draft-sps-record-context-buckets N1,N2,...
--spec-draft-sps-record-active N1,N2,...
--spec-draft-sps-record-caps N1,N2,...
--spec-draft-sps-record-samples N
--spec-draft-sps-record-warmup N
```

The identity is a SHA-256 fingerprint over the target and draft model hashes,
server build, backend/CUDA and GPU identity, and relevant runtime arguments.
It is immutable across arms of one run. A mismatched identity makes resume
fail closed instead of mixing measurements from different systems.

The controls are fail-closed, DSpark-only, and incompatible with adaptive
speculation and an executing SPS profile. A retained sample must include raw
context tokens, exact active slots, exact total verification rows, per-slot
prefixes, elapsed microseconds, and execution kind. Mixed prompt/decode ticks,
sub-batches, retries, direct warmup, CUDA graph capture, and the configured
number of otherwise eligible stable executions are excluded. Prometheus and
the audit sidecar retain exclusion counters.

The intended guarded runner will execute one plan cell per server arm. It will
pass one context ceiling, one active count, one prefix cap, and the derived
forced row count to that arm. The arm can emit a complete one-coordinate v2
partial; the future runner must merge partials only after every scheduled cell
has completed. This is intentionally slower than mutating a live server and
avoids adding a runtime control endpoint to the production HTTP surface.

Use a numeric token array with the native completion endpoint to create an
exact prompt length. Use a request barrier for concurrent slots, temperature
zero, a deterministic request seed, ignored EOS, and enough output tokens to
collect the planned pure-decode sample count. A cell fails when its retained
samples do not all have the requested active-slot and row coordinates.

Use deterministic nearest-rank p95 followed by a monotone upper envelope for
the planner cost. The audit sidecar retains raw samples, p50, p95, maximum,
and exclusion counters. The primary profile contains the conservative envelope.

The timer begins immediately before target `llama_decode()` and ends after its
required synchronization. Queue wait, speculative post-processing, JSON I/O,
and metrics work are outside the measured interval. The primary profile is
written atomically only when its configured grid is complete. The audit
sidecar is atomic and resume-capable after every observation. An incomplete
sidecar cannot coexist with an old primary profile at the same output path.

The guarded runner will construct a server arm equivalent to this template;
do not launch it outside the exclusive GPU guard:

```text
llama-server ... --spec-type draft-dspark --spec-draft-n-max 7 \
  --spec-draft-sps-record record.json \
  --spec-draft-sps-record-identity <64-hex-runtime-fingerprint> \
  --spec-draft-sps-record-context-buckets 8192 \
  --spec-draft-sps-record-active 4 \
  --spec-draft-sps-record-caps 3 \
  --spec-draft-sps-force-verify-rows 16 \
  --spec-draft-sps-record-samples 64 \
  --spec-draft-sps-record-warmup 3
```

## Future guarded execution

`Invoke-DsparkSpsProfile.ps1` is the only intended execution entry point. It
validates the plan on CPU and then launches the whole matrix as one child of
`Invoke-ExclusiveGpuTask.ps1`. All server arms must stay sequential inside
that one child.

```powershell
.\scripts\Invoke-DsparkSpsProfile.ps1 `
  -PlanPath .\benchmarks\dspark-sps-profile\plan.json `
  -ServerPath .\build\bin\Release\llama-server.exe `
  -ModelPath C:\models\target.gguf `
  -DraftModelPath C:\models\draft.gguf `
  -OutputDir .\benchmarks\dspark-sps-profile\run `
  -GpuIndex 0 `
  -MinFreeRamGiB 16 `
  -MinFreeVramMiB 4096 `
  -MaxUsedVramMiB 28672
```

The Python `run` command currently exits before inspecting runtime paths or
starting a process. Remove that stop only with reviewed server-process,
request-barrier, artifact-validation, and cleanup lifecycle tests. The wrapper
already ensures the future full matrix is one guarded child, not one lease per
arm.

Expected future output layout:

```text
plan.json
manifest.json
runs/<ordinal>/command.json
runs/<ordinal>/server.stdout.log
runs/<ordinal>/server.stderr.log
runs/<ordinal>/client.json
runs/<ordinal>/metrics-before.txt
runs/<ordinal>/metrics-after.txt
runs/<ordinal>/record.json
runs/<ordinal>/record.samples.json
profile.json
profile.samples.json
validation.json
summary.json
guard.stdout.log
guard.stderr.log
```

The manifest must record the llama.cpp commit, server and model hashes, exact
non-secret arguments, CUDA and driver identity, GPU UUID, slot and context
configuration, output token contract, throughput statistics, and all errors.

## CPU tests

```powershell
python .\tests\test_dspark_sps_profile.py
```

These tests do not call `nvidia-smi`, load a model, or start a server.
