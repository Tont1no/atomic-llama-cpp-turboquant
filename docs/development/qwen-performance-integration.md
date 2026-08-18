# Qwen performance integration branch

This branch is based on upstream tag `b10448`
(`ad1de39e0708e3ced9c71bb3c82d93a2c046a73f`). It combines the independent
Qwen DFlash and native ModelOpt FP8 work without importing benchmark artifacts,
model files, or the measured-regression experiments.

## Integrated provenance

| Area | Source tip | Integration status |
| --- | --- | --- |
| Device-resident DFlash feature injection | `73c6d7f58` | Guarded fast path; host path remains the automatic fallback |
| Load-adaptive DFlash/DSpark proposal length | `c7943bf4c` | Experimental, disabled by default |
| Active recurrent execution depth | `24b8479a9` | Used only by the adaptive DFlash-family path |
| Compact recurrent snapshot storage and per-row depth | `9af282e74` | Used only by the adaptive DFlash-family path; requires all callers to be rebuilt |
| Native ModelOpt E4M3 matmul | `620079c4e` | Experimental, model-conversion opt-in, SM120 and CUDA 13.2 build required |

The complete ancestry, including intermediate correctness and diagnostics
commits, was replayed in order. Native FP8 was then applied as its separate
two-commit series (`780506a50`, `620079c4e`) after the recurrent-state series.

The following work is intentionally not part of this branch:

- deferred accepted-prefix work (`fb30ba7d4`): not qualified;
- F16 recurrent cache (`25a89a6dc`): reduced memory but regressed measured
  DFlash throughput;
- fused K-cache injection (`9c97d783f`): experimental and kept on a separate
  opt-in branch;
- stacked K/V projection (`f689bfe44`): experimental and kept on a separate
  opt-in branch.

## Enable and disable controls

### DFlash

DFlash itself is selected explicitly with `--spec-type draft-dflash` and a
DFlash draft model. The device-resident injection is guarded internally. If
any batch-layout, metadata, device, or tensor-shape precondition is not met,
the existing host injection path is used automatically.

There is no separate switch for only the device-resident sub-path. The
production kill switch is to remove `draft-dflash` from `--spec-type` (or use
`--spec-type none`). This turns off DFlash, including the device-resident path,
without changing the target model.

### Adaptive proposal and compact recurrent snapshots

Adaptive scheduling is off by default. Enable it only for a DFlash/DSpark
server:

```text
--spec-draft-adaptive
--spec-draft-load-caps 7,3,2,1,1,1,1,0
```

Disable it with `--no-spec-draft-adaptive` or the environment value
`LLAMA_ARG_SPEC_DRAFT_ADAPTIVE=0`. Disabling adaptive scheduling also disables
dynamic recurrent execution depth and compact recurrent snapshot resizing;
the configured fixed rollback layout is retained.

The load-cap entries correspond to one, two, and subsequent active slots. Only
`0`, `1`, `2`, `3`, and `7` are valid caps. Acceptance-based EMA adaptation is
separate and remains disabled while `--spec-draft-acceptance-ema` is zero.

### Native ModelOpt E4M3

Native FP8 is selected when converting a compatible ModelOpt checkpoint:

```text
python convert_hf_to_gguf.py MODEL_DIR --keep-fp8-e4m3 --outfile MODEL-fp8.gguf
```

This is deliberately not a runtime toggle. An F8 GGUF fails closed when the
backend, CUDA version, SM architecture, tensor shape, or scale sidecars do not
meet the native-kernel contract. Keep a separately converted portable model as
the kill-switch/fallback:

```text
python convert_hf_to_gguf.py MODEL_DIR --fp8-as-q8 --outfile MODEL-q8.gguf
```

`--keep-fp8-e4m3` and `--fp8-as-q8` are mutually exclusive. Native GeForce FP8
requires an SM120 CUDA build made with CUDA Toolkit 13.2 or newer. It is not a
4090/Ada path.

## Build and smoke checks

CPU parser and state tests can be built without a GPU backend:

```powershell
cmake -S . -B build-integration-cpu-vs17 -G "Visual Studio 17 2022" -A x64 `
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-integration-cpu-vs17 --config Release `
  --target llama-cli test-arg-parser test-speculative-adaptive `
  test-recurrent-active-depth test-server-queue-load test-fp8-e4m3 -j 12
```

The local side-by-side CUDA 13.2 toolkit must be selected explicitly; otherwise
Visual Studio may silently use the system CUDA toolkit:

```powershell
$Cuda132Root = 'C:\Users\pasca\Documents\GitHub\Ai-Loader\.codex-deploy\cuda-13.2\toolkit'
cmake -S . -B build-integration-cuda132-vs17 -G "Visual Studio 17 2022" -A x64 `
  -T "cuda=$Cuda132Root" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 `
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-integration-cuda132-vs17 --config Release `
  --target llama-cli test-fp8-e4m3 -j 8
```

These commands compile CUDA code but do not start `llama-server`, load a model,
or execute a CUDA test. GPU correctness and throughput qualification remain a
separate, exclusive-run step.
