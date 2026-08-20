# Exclusive local GPU tasks

Use `scripts/Invoke-ExclusiveGpuTask.ps1` for every local llama.cpp server,
backend test, or model validation that can allocate substantial VRAM.

The wrapper provides one workspace-wide lease, refuses to start while a known
llama/test process is running, binds the child to one selected CUDA adapter,
monitors RAM and VRAM, terminates the child tree on a limit violation, and
owns the child tree through a nonce-named Windows Job Object. Cleanup never
selects a process by name, PID reuse, or start-time similarity.

Example for the RTX 5090:

```powershell
.\scripts\Invoke-ExclusiveGpuTask.ps1 `
  -Executable .\.codex-deploy\some-build\bin\Release\llama-server.exe `
  -TaskArguments @('-m', 'C:\models\model.gguf', '--device', 'CUDA0') `
  -GpuIndex 0 `
  -MinFreeRamGiB 16 `
  -MinFreeVramMiB 4096 `
  -MaxUsedVramMiB 28672 `
  -MaxRuntimeSeconds 21600 `
  -StdoutPath .\benchmarks\run\stdout.log `
  -StderrPath .\benchmarks\run\stderr.log
```

`MaxRuntimeSeconds` defaults to `0` for backward compatibility (disabled).
When positive, it is a hard wall-clock limit: the guard kills the entire child
tree and still removes the GPU lease. The lease is v2 JSON containing a random
256-bit nonce, owner PID, owner lifetime, Job Object name, and GPU UUID. It
stays open with read sharing but write denial. Guarded children receive the
exact lease path, nonce, owner PID, Job Object name, UUID, and matching
`CUDA_VISIBLE_DEVICES`. A nested runner requires an active sharing-violation
lock and proves that it is a member of that exact Job Object. A copied,
read-only, sibling-locked, or unlocked lease cannot authorize execution.

`GpuIndex` is the physical `nvidia-smi` index. The wrapper resolves that
adapter's UUID, then gives the child `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES=<GPU UUID>`. The selected adapter therefore appears to
the child as `CUDA0` even if CUDA and `nvidia-smi` enumerate adapters
differently.

The guard is deliberately fail-closed. Do not bypass it because a benchmark is
short. Run the CPU-only regression suite after changing the wrapper:

```powershell
.\tests\test_exclusive_gpu_guard.ps1
```

The memory monitor polls periodically and is therefore a best-effort emergency
limit, not a CUDA allocator quota. Choose thresholds with headroom. The default
5090 policy leaves at least 4 GiB VRAM and 16 GiB system RAM available. Only one
root GPU task is allowed; a benchmark orchestrator must run all of its arms
sequentially inside that one guarded task.
