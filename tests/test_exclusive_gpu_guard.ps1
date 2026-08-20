[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) {
        throw "ASSERTION FAILED: $Message"
    }
}

function Stop-TestProcess {
    param($Process)
    if (-not $Process -or $Process.HasExited) {
        return
    }
    Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    [void] $Process.WaitForExit(5000)
}

$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$sourceGuardPath = Join-Path $workspaceRoot 'scripts\Invoke-ExclusiveGpuTask.ps1'
$sourceDriverPath = Join-Path $workspaceRoot 'scripts\dspark_sps_profile.py'
$pwshPath = (Get-Command pwsh.exe -ErrorAction Stop).Source
$pythonPath = (Get-Command python.exe -ErrorAction Stop).Source
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ai-loader-gpu-guard-" + [guid]::NewGuid().ToString('N'))
$testWorkspace = Join-Path $testRoot 'workspace'
$testScripts = Join-Path $testWorkspace 'scripts'
$guardPath = Join-Path $testScripts 'Invoke-ExclusiveGpuTask.ps1'
$leasePath = Join-Path $testWorkspace '.codex-deploy\gpu-exclusive.lock'
$fakeBin = Join-Path $testRoot 'bin'
$fakeLog = Join-Path $testRoot 'nvidia-smi-args.log'
$fakeCounter = Join-Path $testRoot 'nvidia-smi-counter.txt'

New-Item -ItemType Directory -Path $fakeBin -Force | Out-Null
New-Item -ItemType Directory -Path $testScripts -Force | Out-Null
Copy-Item -LiteralPath $sourceGuardPath -Destination $guardPath
Copy-Item -LiteralPath $sourceDriverPath -Destination (Join-Path $testScripts 'dspark_sps_profile.py')

$fakeSmiPs1 = @'
$ErrorActionPreference = 'Stop'
Add-Content -LiteralPath $env:FAKE_SMI_LOG -Value ($args -join ' ')
$count = 0
if (Test-Path -LiteralPath $env:FAKE_SMI_COUNTER) {
    $count = [int](Get-Content -Raw -LiteralPath $env:FAKE_SMI_COUNTER)
}
$count++
[System.IO.File]::WriteAllText($env:FAKE_SMI_COUNTER, [string]$count)
if ($env:FAKE_SMI_MODE -eq 'over-limit' -and $count -ge 2) {
    'GPU-00000000-0000-0000-0000-000000000007, 32607, 30000, 2607'
} else {
    'GPU-00000000-0000-0000-0000-000000000007, 32607, 2048, 30559'
}
'@
[System.IO.File]::WriteAllText((Join-Path $fakeBin 'fake-nvidia-smi.ps1'), $fakeSmiPs1)
$fakeSmiCmd = "@echo off`r`npwsh.exe -NoLogo -NoProfile -File `"%~dp0fake-nvidia-smi.ps1`" %*`r`n"
[System.IO.File]::WriteAllText((Join-Path $fakeBin 'nvidia-smi.cmd'), $fakeSmiCmd)

$oldPath = $env:PATH
$oldFakeLog = $env:FAKE_SMI_LOG
$oldFakeCounter = $env:FAKE_SMI_COUNTER
$oldFakeMode = $env:FAKE_SMI_MODE
$first = $null
$renamedPing = $null
$foreignProcess = $null
$lineageRunner = $null
$crashGuard = $null

try {
    $env:PATH = "$fakeBin;$oldPath"
    $env:FAKE_SMI_LOG = $fakeLog
    $env:FAKE_SMI_COUNTER = $fakeCounter
    $env:FAKE_SMI_MODE = 'safe'

    # Explicit GPU selection and a post-start sample are required even for short tasks.
    [System.IO.File]::WriteAllText($fakeCounter, '0')
    $visibleGpuPath = Join-Path $testRoot 'visible-gpu.txt'
    $guardMarkerPath = Join-Path $testRoot 'guard-marker.txt'
    $short = & $guardPath -Executable $pwshPath -TaskArguments @(
        '-NoLogo', '-NoProfile', '-Command',
        "[IO.File]::WriteAllText('$($visibleGpuPath.Replace("'", "''"))',`$env:CUDA_VISIBLE_DEVICES); " +
        "[IO.File]::WriteAllText('$($guardMarkerPath.Replace("'", "''"))',`$env:AI_LOADER_EXCLUSIVE_GPU_GUARD + '|' + `$env:AI_LOADER_EXCLUSIVE_GPU_UUID); " +
        "Start-Sleep -Milliseconds 50"
    ) -GpuIndex 7 -MaxUsedVramMiB 8192 -MinFreeVramMiB 20000 -PollMilliseconds 100
    Assert-True ($short.Status -eq 'completed') 'short task did not complete'
    Assert-True ($short.GpuIndex -eq 7) 'result did not preserve GpuIndex'
    Assert-True ($short.GpuUuid -eq 'GPU-00000000-0000-0000-0000-000000000007') 'result did not preserve GPU UUID'
    Assert-True ((Get-Content -Raw -LiteralPath $visibleGpuPath) -eq $short.GpuUuid) 'child was not bound to the selected GPU UUID'
    Assert-True ((Get-Content -Raw -LiteralPath $guardMarkerPath) -eq "1|$($short.GpuUuid)") 'child did not receive the exclusive guard proof'
    Assert-True ($short.PeakUsedVramMiB -eq 2048) 'short task did not record a memory sample'
    Assert-True ([int](Get-Content -Raw -LiteralPath $fakeCounter) -ge 3) 'short task did not record post-start and final memory samples'
    Assert-True ((Get-Content -Raw -LiteralPath $fakeLog) -match '--id=7') 'nvidia-smi did not receive --id=7'

    # Exercise the real Python lease verifier while the guard owns the
    # read-share/write-deny lease handle.
    $handshakeScript = Join-Path $testRoot 'verify-guard-handshake.py'
    $handshakeResult = Join-Path $testRoot 'verified-guard-uuid.txt'
    $handshakeSource = @"
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('dspark_sps_profile', r'$((Join-Path $testScripts 'dspark_sps_profile.py').Replace("'", "''"))')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Path(r'$($handshakeResult.Replace("'", "''"))').write_text(module.verify_guard_lease(Path(r'$($testWorkspace.Replace("'", "''"))')), encoding='utf-8')
"@
    [IO.File]::WriteAllText($handshakeScript, $handshakeSource)
    [System.IO.File]::WriteAllText($fakeCounter, '0')
    $handshake = & $guardPath -Executable $pythonPath -TaskArguments @(
        $handshakeScript
    ) -TaskWorkingDirectory $testWorkspace -PollMilliseconds 100
    Assert-True ($handshake.Status -eq 'completed') 'real Python guard handshake did not complete'
    Assert-True ((Get-Content -Raw -LiteralPath $handshakeResult) -eq $handshake.GpuUuid) 'Python did not verify the active lease UUID'

    # A loser in the CreateNew race must not remove the winning runner's lease.
    [System.IO.File]::WriteAllText($fakeCounter, '0')
    $firstRunnerScript = @"
& '$($guardPath.Replace("'", "''"))' -Executable '$($pwshPath.Replace("'", "''"))' -TaskArguments @(
    '-NoLogo', '-NoProfile', '-Command', 'Start-Sleep -Seconds 3'
) -PollMilliseconds 100 | Out-Null
"@
    $firstRunnerEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($firstRunnerScript))
    $first = Start-Process -FilePath $pwshPath -PassThru -ArgumentList @(
        '-NoLogo', '-NoProfile', '-EncodedCommand', $firstRunnerEncoded
    )
    $deadline = [datetime]::UtcNow.AddSeconds(5)
    while (-not (Test-Path -LiteralPath $leasePath) -and [datetime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 25
    }
    Assert-True (Test-Path -LiteralPath $leasePath) 'first runner never acquired the lease'

    $secondFailed = $false
    try {
        & $guardPath -Executable $pwshPath -TaskArguments @(
            '-NoLogo', '-NoProfile', '-Command', 'exit 0'
        ) -PollMilliseconds 100 | Out-Null
    } catch {
        $secondFailed = $true
    }
    Assert-True $secondFailed 'second runner unexpectedly acquired the lease'
    Assert-True (Test-Path -LiteralPath $leasePath) 'losing runner removed the winning lease'
    Assert-True (-not $first.HasExited) 'winning runner was disturbed by the collision'
    [void] $first.WaitForExit(10000)
    Assert-True $first.HasExited 'winning runner did not exit'
    Assert-True (-not (Test-Path -LiteralPath $leasePath)) 'winning runner did not release its lease'

    # A reused PID is stale when it started after the recorded lease creation.
    $stale = @{
        OwnerPid = $PID
        CreatedAt = '2000-01-01T00:00:00.0000000Z'
        Executable = 'stale-test'
    } | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText($leasePath, $stale)
    [System.IO.File]::WriteAllText($fakeCounter, '0')
    $staleResult = & $guardPath -Executable $pwshPath -TaskArguments @(
        '-NoLogo', '-NoProfile', '-Command', 'exit 0'
    ) -PollMilliseconds 100
    Assert-True ($staleResult.Status -eq 'completed') 'stale lease was not reclaimed'
    Assert-True (-not (Test-Path -LiteralPath $leasePath)) 'reclaimed lease was not cleaned up'

    # The wrapper must kill an over-limit child and release its lease.
    $childPidPath = Join-Path $testRoot 'guarded-child.pid'
    [System.IO.File]::WriteAllText($fakeCounter, '0')
    $env:FAKE_SMI_MODE = 'over-limit'
    $limitFailed = $false
    $limitError = ''
    try {
        & $guardPath -Executable $pwshPath -TaskArguments @(
            '-NoLogo', '-NoProfile', '-Command',
            "[IO.File]::WriteAllText('$($childPidPath.Replace("'", "''"))',[string]`$PID); Start-Sleep -Seconds 30"
        ) -PollMilliseconds 100 | Out-Null
    } catch {
        $limitError = $_.Exception.Message
        $limitFailed = $_.Exception.Message -like '*terminated by the memory guard*'
    }
    Assert-True $limitFailed "over-limit task did not fail through the memory guard: $limitError"
    Assert-True (Test-Path -LiteralPath $childPidPath) 'guarded child did not publish its PID'
    $guardedPid = [int](Get-Content -Raw -LiteralPath $childPidPath)
    Start-Sleep -Milliseconds 100
    Assert-True (-not (Get-Process -Id $guardedPid -ErrorAction SilentlyContinue)) 'over-limit child remained alive'
    Assert-True (-not (Test-Path -LiteralPath $leasePath)) 'over-limit run left a lease behind'

    # A hard runtime limit must kill the child tree and release its lease.
    $runtimeChildPidPath = Join-Path $testRoot 'runtime-child.pid'
    [System.IO.File]::WriteAllText($fakeCounter, '0')
    $env:FAKE_SMI_MODE = 'safe'
    $runtimeFailed = $false
    try {
        & $guardPath -Executable $pwshPath -TaskArguments @(
            '-NoLogo', '-NoProfile', '-Command',
            "[IO.File]::WriteAllText('$($runtimeChildPidPath.Replace("'", "''"))',[string]`$PID); Start-Sleep -Seconds 30"
        ) -PollMilliseconds 100 -MaxRuntimeSeconds 1 | Out-Null
    } catch {
        $runtimeFailed = $_.Exception.Message -like '*terminated by the runtime guard*'
    }
    Assert-True $runtimeFailed 'hard runtime limit did not fail through the runtime guard'
    Assert-True (Test-Path -LiteralPath $runtimeChildPidPath) 'runtime-limited child did not publish its PID'
    $runtimeChildPid = [int](Get-Content -Raw -LiteralPath $runtimeChildPidPath)
    Start-Sleep -Milliseconds 100
    Assert-True (-not (Get-Process -Id $runtimeChildPid -ErrorAction SilentlyContinue)) 'runtime-limited child remained alive'
    Assert-True (-not (Test-Path -LiteralPath $leasePath)) 'runtime-limited run left a lease behind'

    # A named descendant that outlives the root process must be terminated before
    # redirected stdout is awaited.
    $env:FAKE_SMI_MODE = 'safe'
    [System.IO.File]::WriteAllText($fakeCounter, '0')
    $renamedPing = Join-Path $testRoot 'test-backend-ops.exe'
    Copy-Item -LiteralPath (Join-Path $env:SystemRoot 'System32\ping.exe') -Destination $renamedPing
    $descendantLog = Join-Path $testRoot 'descendant.stdout.log'
    $descendantFailed = $false
    try {
        & $guardPath -Executable $pwshPath -TaskArguments @(
            '-NoLogo', '-NoProfile', '-Command',
            "Start-Process -FilePath '$($renamedPing.Replace("'", "''"))' -ArgumentList @('127.0.0.1','-n','30'); Start-Sleep -Milliseconds 100"
        ) -PollMilliseconds 100 -StdoutPath $descendantLog | Out-Null
    } catch {
        $descendantFailed = $_.Exception.Message -like '*Job Object terminated them*'
    }
    Assert-True $descendantFailed 'outliving named descendant was not reported'
    Start-Sleep -Milliseconds 100
    $renamedLeftovers = @(Get-Process -Name 'test-backend-ops' -ErrorAction SilentlyContinue)
    Assert-True ($renamedLeftovers.Count -eq 0) 'outliving named descendant was not terminated'
    Assert-True (-not (Test-Path -LiteralPath $leasePath)) 'descendant cleanup left a lease behind'

    # A same-named process started later by another parent is not task-owned and
    # must survive cleanup of a proven descendant.
    [System.IO.File]::WriteAllText($fakeCounter, '0')
    $lineageReady = Join-Path $testRoot 'lineage-ready.txt'
    $ownedPidPath = Join-Path $testRoot 'owned-descendant.pid'
    $lineageError = Join-Path $testRoot 'lineage-error.txt'
    $ownedChildCommand = @"
`$owned = Start-Process -FilePath '$($renamedPing.Replace("'", "''"))' -ArgumentList @('127.0.0.1','-n','30') -PassThru
[IO.File]::WriteAllText('$($ownedPidPath.Replace("'", "''"))',[string]`$owned.Id)
[IO.File]::WriteAllText('$($lineageReady.Replace("'", "''"))','ready')
Start-Sleep -Seconds 1
"@
    $ownedChildEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($ownedChildCommand))
    $lineageGuardScript = @"
try {
    & '$($guardPath.Replace("'", "''"))' -Executable '$($pwshPath.Replace("'", "''"))' -TaskArguments @(
        '-NoLogo', '-NoProfile', '-EncodedCommand', '$ownedChildEncoded'
    ) -PollMilliseconds 100 | Out-Null
} catch {
    [IO.File]::WriteAllText('$($lineageError.Replace("'", "''"))', `$_.Exception.Message)
}
"@
    $lineageGuardEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($lineageGuardScript))
    $lineageRunner = Start-Process -FilePath $pwshPath -PassThru -ArgumentList @(
        '-NoLogo', '-NoProfile', '-EncodedCommand', $lineageGuardEncoded
    )
    $deadline = [datetime]::UtcNow.AddSeconds(5)
    while (-not (Test-Path -LiteralPath $lineageReady) -and [datetime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 25
    }
    Assert-True (Test-Path -LiteralPath $lineageReady) 'guarded lineage child did not become ready'
    $foreignProcess = Start-Process -FilePath $renamedPing -PassThru -ArgumentList @('127.0.0.1', '-n', '30')
    [void] $lineageRunner.WaitForExit(10000)
    Assert-True $lineageRunner.HasExited 'lineage guard test did not exit'
    Assert-True (Test-Path -LiteralPath $lineageError) 'owned descendant was not reported in lineage test'
    Assert-True ((Get-Content -Raw -LiteralPath $lineageError) -like '*Job Object terminated them*') 'lineage cleanup reported an unexpected error'
    $ownedPid = [int](Get-Content -Raw -LiteralPath $ownedPidPath)
    Assert-True (-not (Get-Process -Id $ownedPid -ErrorAction SilentlyContinue)) 'owned same-named descendant survived cleanup'
    Assert-True (-not $foreignProcess.HasExited) 'foreign same-named process was terminated by cleanup'
    Stop-TestProcess $foreignProcess

    # Closing the guard process must close the Job Object and kill its child
    # before a later guard can reclaim the stale lease.
    [System.IO.File]::WriteAllText($fakeCounter, '0')
    $crashChildPidPath = Join-Path $testRoot 'crash-child.pid'
    $crashScript = @"
& '$($guardPath.Replace("'", "''"))' -Executable '$($pwshPath.Replace("'", "''"))' -TaskArguments @(
    '-NoLogo', '-NoProfile', '-Command',
    "[IO.File]::WriteAllText('$($crashChildPidPath.Replace("'", "''"))',[string]`$PID); Start-Sleep -Seconds 30"
) -PollMilliseconds 100 | Out-Null
"@
    $crashEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($crashScript))
    $crashGuard = Start-Process -FilePath $pwshPath -PassThru -ArgumentList @(
        '-NoLogo', '-NoProfile', '-EncodedCommand', $crashEncoded
    )
    $deadline = [datetime]::UtcNow.AddSeconds(5)
    while (-not (Test-Path -LiteralPath $crashChildPidPath) -and [datetime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 25
    }
    Assert-True (Test-Path -LiteralPath $crashChildPidPath) 'crash test child did not start'
    $crashChildPid = [int](Get-Content -Raw -LiteralPath $crashChildPidPath)
    Stop-TestProcess $crashGuard
    $deadline = [datetime]::UtcNow.AddSeconds(5)
    while ((Get-Process -Id $crashChildPid -ErrorAction SilentlyContinue) -and [datetime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 25
    }
    Assert-True (-not (Get-Process -Id $crashChildPid -ErrorAction SilentlyContinue)) 'Job kill-on-close did not terminate the crash child'
    $reclaimed = & $guardPath -Executable $pwshPath -TaskArguments @(
        '-NoLogo', '-NoProfile', '-Command', 'Start-Sleep -Milliseconds 50'
    ) -PollMilliseconds 100
    Assert-True ($reclaimed.Status -eq 'completed') 'later guard did not reclaim the crashed owner lease'
    Assert-True (-not (Test-Path -LiteralPath $leasePath)) 'crash-recovery guard left a lease behind'

    'Exclusive GPU guard CPU-only tests: PASS'
} finally {
    if ($first -and -not $first.HasExited) {
        try {
            Stop-TestProcess $first
        } catch {
        }
    }
    if ($lineageRunner -and -not $lineageRunner.HasExited) {
        try {
            Stop-TestProcess $lineageRunner
        } catch {}
    }
    if ($foreignProcess -and -not $foreignProcess.HasExited) {
        try {
            Stop-TestProcess $foreignProcess
        } catch {}
    }
    if ($crashGuard -and -not $crashGuard.HasExited) {
        try {
            Stop-TestProcess $crashGuard
        } catch {}
    }
    if ($renamedPing) {
        foreach ($process in @(Get-Process -Name 'test-backend-ops' -ErrorAction SilentlyContinue)) {
            try {
                if ($process.Path -eq $renamedPing) {
                    Stop-TestProcess $process
                }
            } catch {
            }
        }
    }
    $env:PATH = $oldPath
    $env:FAKE_SMI_LOG = $oldFakeLog
    $env:FAKE_SMI_COUNTER = $oldFakeCounter
    $env:FAKE_SMI_MODE = $oldFakeMode
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
