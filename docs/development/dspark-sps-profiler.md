# DSpark SPS profiler orchestration

This directory contains the CPU-only planning layer, the server-side DSpark
SPS recorder, and a guarded profiling runner. Planning and validation never
execute a model. GPU execution is accepted only through the PowerShell wrapper
and is not part of the CPU test suite.

## Plan contract

Generate a deterministic plan:

```powershell
python .\scripts\dspark_sps_profile.py plan `
  --output .\benchmarks\dspark-sps-profile\plan.json `
  --active-slots 1,2,3,4,5,6,7,8 `
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

The anchor row is included. Exact active-slot values are accepted for every
integer from 1 through 8, so ramp and drain states are profiled rather than
rounded to powers of two. Each active value has its own row axis. For seven
draft tokens per slot and caps 0, 1, 2, 3, and 7, representative axes are:

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

The guarded runner executes one plan cell per server arm. It passes one
context ceiling, one active count, one prefix cap, and the derived
forced row count to that arm. The arm emits a complete one-coordinate v2
partial; the runner merges partials only after every scheduled cell
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
On Windows the recorder publishes replacements with `ReplaceFileW`. The
runner never opens recorder JSON while a request wave is active: live progress
uses Prometheus counters only. It opens and strictly validates one stable
sidecar/profile generation after the workers and recorder-ready proof finish.
The recorder retries only Win32 errors 5, 32, 33, and 1175 for at most 500 ms
with short sleeps; other replacement errors and the single initial publish
fail immediately. Terminal logs include the retry count and Win32 error.

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

## Guarded execution

`Invoke-DsparkSpsProfile.ps1` is the only execution entry point. It
validates the plan on CPU and then launches the whole matrix as one child of
`Invoke-ExclusiveGpuTask.ps1`. The guard holds a readable but write-denied v2
lease containing a cryptographic nonce, owner PID, nonce-named Windows Job
Object, and selected GPU UUID.
Python requires the exact workspace lease path, matching child environment,
its direct parent as owner, matching `CUDA_VISIBLE_DEVICES`, a sharing-violation
write lock, and membership in that exact Job Object. The runner repeats the
proof before each arm and during each request wave. Copied, read-only,
sibling-locked, or unlocked lease JSON is rejected. All server arms stay
sequential inside that one child because the
exact forced-row cap is a startup-only option.

```powershell
.\scripts\Invoke-DsparkSpsProfile.ps1 `
  -PlanPath .\benchmarks\dspark-sps-profile\plan.json `
  -ServerPath .\build\bin\Release\llama-server.exe `
  -ModelPath C:\models\target.gguf `
  -DraftModelPath C:\models\draft.gguf `
  -OutputDir .\benchmarks\dspark-sps-profile\run `
  -RuntimeLabel 'CUDA 13.0 / SM120 optimized llama.cpp' `
  -DynamicRs $false `
  -GpuIndex 0 `
  -MinFreeRamGiB 16 `
  -MinFreeVramMiB 4096 `
  -MaxUsedVramMiB 28672 `
  -ContextSize 65536 `
  -Parallel 8 `
  -JobTimeoutSeconds 18000 `
  -MaxRuntimeSeconds 21600
```

`MaxRuntimeSeconds` is the outer hard kill limit and must be zero or greater
than `JobTimeoutSeconds`. The inner job, arm, readiness, HTTP,
recorder-progress, and process-stop timeouts are independently configurable.
A run always passes an explicit recurrent-snapshot policy to llama-server:
the default `-DynamicRs $false` emits `--no-spec-draft-dynamic-rs`; an explicit
`-DynamicRs $true` opt-in emits `--spec-draft-dynamic-rs`. This boolean is part
of the runtime fingerprint, manifest identity, and resume compatibility gate.
A server is started in its own process group for every arm and its full tree
is stopped in a `finally` path. The next arm cannot start until the previous
port is closed.

The numeric prompt length is
`context_ceiling - output_tokens - prompt_safety_tokens`. The configured
per-slot context (`ContextSize / Parallel`) must cover the largest ceiling;
the safety tokens leave headroom below that ceiling. Requests in one wave
share a barrier and use numeric token arrays, deterministic seeds, temperature
zero, and ignored EOS. A completion is accepted only when it stops at the
requested limit, reports exactly the requested token count, and is not
truncated. `/health`, `/props`, and recorder metrics are polled and validated
before an arm can complete. The sidecar and profile are read only after the
request workers finish and metrics prove the recorder quota ready. Completion HTTP calls are bounded by
the nearest request, arm, and recorder-progress deadline. A peer or progress
failure immediately closes registered completion connections and the server
endpoint. Request workers are daemonized and joined only up to the bounded
process-stop timeout, so cleanup never waits for the longer HTTP timeout.

Do not call `python scripts/dspark_sps_profile.py run` directly. It fails
before runtime-path inspection unless the outer guard proof is present.

Output layout:

```text
manifest.json
run-state.json
runs/<ordinal>/command.json
runs/<ordinal>/server.stdout.log
runs/<ordinal>/server.stderr.log
runs/<ordinal>/client.json
runs/<ordinal>/props.json
runs/<ordinal>/metrics-before.txt
runs/<ordinal>/metrics-after.txt
runs/<ordinal>/record.json
runs/<ordinal>/record.samples.json
runs/<ordinal>/receipt.json
profile.json
profile.samples.json
validation.json
summary.json
guard.stdout.log
guard.stderr.log
```

The manifest records the llama.cpp commit, server and model hashes, sorted
hashes of adjacent `llama*.dll` and `ggml*.dll` runtime libraries, exact
immutable non-secret arguments, explicit runtime label, driver identity, GPU
UUID, slot/context configuration, and output-token contract. Each completed
arm receipt binds the command, `/props`, metrics, client, recorder profile,
and raw sidecar by SHA-256. Run status and errors live in `run-state.json`;
throughput and final hashes live in `summary.json` and `validation.json`.

Resume is enabled by default. A changed plan, executable/model hash,
GPU/runtime identity, or immutable argument is rejected. A completed ordinal
is skipped only after all bound artifacts are loaded, strictly validated, and
rehashed. An incomplete ordinal may reuse its atomic recorder sidecar, but its
server command must be equivalent to the original command artifact.

After every scheduled repetition has produced its own sidecar, all raw samples
are combined per coordinate. The runner calculates deterministic nearest-rank
p95, applies a conservative monotone upper envelope over comparable physical
coordinates, and atomically writes the strict Loader-v2 `profile.json`. Raw
samples and aggregation evidence remain separate in `profile.samples.json`;
unreachable Cartesian cells are never invented.

## CPU tests

```powershell
python .\tests\test_dspark_sps_profile.py
pwsh -NoLogo -NoProfile -File .\tests\test_dspark_sps_profile_wrapper.ps1
pwsh -NoLogo -NoProfile -File .\tests\test_exclusive_gpu_guard.ps1
```

These tests use subprocess, HTTP, `nvidia-smi`, guard, timeout, and cleanup
mocks/fakes. They do not access a real GPU, load a model, or start llama-server.
