[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Executable,

    [string[]] $TaskArguments = @(),

    [string] $TaskWorkingDirectory = (Get-Location).Path,

    [int] $MinFreeRamGiB = 16,

    [int] $MinFreeVramMiB = 4096,

    [int] $MaxUsedVramMiB = 28672,

    [int] $GpuIndex = 0,

    [int] $PollMilliseconds = 500,

    [string] $StdoutPath = '',

    [string] $StderrPath = ''
)

$ErrorActionPreference = 'Stop'

if ($MinFreeRamGiB -lt 1) {
    throw 'MinFreeRamGiB must be positive.'
}
if ($MinFreeVramMiB -lt 512) {
    throw 'MinFreeVramMiB must be at least 512 MiB.'
}
if ($MaxUsedVramMiB -lt 512) {
    throw 'MaxUsedVramMiB must be at least 512 MiB.'
}
if ($PollMilliseconds -lt 100) {
    throw 'PollMilliseconds must be at least 100 ms.'
}
if ($GpuIndex -lt 0) {
    throw 'GpuIndex must be non-negative.'
}

$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$leaseDir = Join-Path $workspaceRoot '.codex-deploy'
$leasePath = Join-Path $leaseDir 'gpu-exclusive.lock'
$relevantProcessNames = @(
    'llama-server',
    'llama-cli',
    'test-backend-ops',
    'test-dflash-fusion-determinism',
    'test-fp8-e4m3',
    'test-recurrent-state-rollback'
)

function Get-MemorySnapshot {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeRamMiB = [math]::Floor([double] $os.FreePhysicalMemory / 1024.0)

    $gpuLine = & nvidia-smi `
        "--id=$GpuIndex" `
        --query-gpu=uuid,memory.total,memory.used,memory.free `
        --format=csv,noheader,nounits 2>$null
    if (-not $gpuLine) {
        throw 'nvidia-smi did not return a GPU memory snapshot.'
    }

    $parts = @($gpuLine -split ',' | ForEach-Object { $_.Trim() })
    if ($parts.Count -ne 4) {
        throw "Unexpected nvidia-smi memory output: $gpuLine"
    }

    [pscustomobject]@{
        GpuUuid       = [string] $parts[0]
        FreeRamMiB  = $freeRamMiB
        TotalVramMiB = [int] $parts[1]
        UsedVramMiB  = [int] $parts[2]
        FreeVramMiB  = [int] $parts[3]
    }
}

function Get-RelevantProcesses {
    @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $relevantProcessNames -contains $_.ProcessName
    } | ForEach-Object {
        $processPath = $null
        $processStartTime = $null
        try { $processPath = $_.Path } catch {}
        try { $processStartTime = $_.StartTime } catch {}
        [pscustomobject]@{
            Id          = $_.Id
            ProcessName = $_.ProcessName
            Path        = $processPath
            StartTime   = $processStartTime
        }
    })
}

function Stop-TaskRelevantProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [datetime] $StartedAt
    )

    $results = @()
    foreach ($entry in @(Get-RelevantProcesses | Where-Object {
        $_.Id -ne $PID -and ($null -eq $_.StartTime -or $_.StartTime -ge $StartedAt)
    })) {
        $result = [ordered]@{
            Id          = $entry.Id
            ProcessName = $entry.ProcessName
            Path        = $entry.Path
            StartTime   = $entry.StartTime
            Stopped     = $false
            StillAlive  = $false
            Error       = $null
        }
        $process = Get-Process -Id $entry.Id -ErrorAction SilentlyContinue
        if (-not $process) {
            $result.Stopped = $true
            $results += [pscustomobject] $result
            continue
        }

        try {
            $process.Kill($true)
            [void] $process.WaitForExit(5000)
            $result.Stopped = $process.HasExited
        } catch {
            $result.Error = $_.Exception.Message
        }

        $live = Get-Process -Id $entry.Id -ErrorAction SilentlyContinue
        $result.StillAlive = $null -ne $live -and $live.ProcessName -eq $entry.ProcessName
        if ($result.StillAlive) {
            Write-Warning "Task-owned process $($entry.ProcessName) PID $($entry.Id) is still alive after cleanup."
        }
        $results += [pscustomobject] $result
    }

    @($results)
}

function Remove-StaleLease {
    if (-not (Test-Path -LiteralPath $leasePath)) {
        return
    }

    $lease = $null
    try {
        $lease = Get-Content -Raw -LiteralPath $leasePath | ConvertFrom-Json
    } catch {
        throw "GPU lease exists but is unreadable: $leasePath"
    }

    $createdAt = [datetimeoffset]::MinValue
    if (-not [datetimeoffset]::TryParse([string] $lease.CreatedAt, [ref] $createdAt)) {
        throw "GPU lease has an invalid CreatedAt timestamp: $leasePath"
    }

    $owner = Get-Process -Id ([int] $lease.OwnerPid) -ErrorAction SilentlyContinue
    if ($owner) {
        try {
            $ownerStartedAt = $owner.StartTime.ToUniversalTime()
            $leaseCreatedAt = $createdAt.UtcDateTime
            $recordedOwnerStart = if ($lease.OwnerStartTimeUtc) {
                [datetime]::Parse(
                    [string] $lease.OwnerStartTimeUtc,
                    [Globalization.CultureInfo]::InvariantCulture,
                    [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
            } else {
                $null
            }

            $sameOwnerLifetime = $ownerStartedAt -le $leaseCreatedAt
            if ($recordedOwnerStart) {
                $sameOwnerLifetime = [math]::Abs(($ownerStartedAt - $recordedOwnerStart).TotalSeconds) -lt 1.0
            }

            if ($sameOwnerLifetime) {
                throw "GPU lease is already held by PID $($lease.OwnerPid): $leasePath"
            }
        } catch {
            if ($_.Exception.Message -like 'GPU lease is already held*') {
                throw
            }
            # If the process exited during inspection, the lease is stale.
        }
    }

    Remove-Item -LiteralPath $leasePath -Force
}

New-Item -ItemType Directory -Path $leaseDir -Force | Out-Null
Remove-StaleLease

$leaseStream = $null
$leaseOwned = $false
$child = $null
$childStarted = $false
$taskStartTime = $null
$stdoutRead = $null
$stderrRead = $null
$limitReason = $null
$leftovers = @()
$peakUsedVramMiB = 0
$minimumFreeRamMiB = [int]::MaxValue

try {
    $leaseStream = [System.IO.File]::Open(
        $leasePath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None)
    $leaseOwned = $true

    $leaseJson = [System.Text.Encoding]::UTF8.GetBytes((@{
        OwnerPid = $PID
        CreatedAt = (Get-Date).ToString('o')
        OwnerStartTimeUtc = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
        Executable = $Executable
        GpuIndex = $GpuIndex
    } | ConvertTo-Json -Compress))
    $leaseStream.Write($leaseJson, 0, $leaseJson.Length)
    $leaseStream.Flush($true)

    $conflicts = Get-RelevantProcesses
    if ($conflicts.Count -gt 0) {
        throw "Refusing GPU task because another LLM/test process exists: $($conflicts | ConvertTo-Json -Compress)"
    }

    $before = Get-MemorySnapshot
    $peakUsedVramMiB = $before.UsedVramMiB
    $minimumFreeRamMiB = $before.FreeRamMiB
    if ($before.FreeRamMiB -lt ($MinFreeRamGiB * 1024)) {
        throw "Refusing GPU task: only $($before.FreeRamMiB) MiB system RAM is free."
    }
    if ($before.FreeVramMiB -lt $MinFreeVramMiB) {
        throw "Refusing GPU task: only $($before.FreeVramMiB) MiB VRAM is free."
    }
    if ($before.UsedVramMiB -gt $MaxUsedVramMiB) {
        throw "Refusing GPU task: $($before.UsedVramMiB) MiB VRAM is already used."
    }

    $resolvedExecutable = (Resolve-Path -LiteralPath $Executable).Path
    $resolvedWorkingDirectory = (Resolve-Path -LiteralPath $TaskWorkingDirectory).Path
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $resolvedExecutable
    $psi.WorkingDirectory = $resolvedWorkingDirectory
    $psi.UseShellExecute = $false
    # Monitor and expose the same physical GPU by UUID. The guarded child
    # therefore sees the selected adapter as CUDA0 and cannot allocate elsewhere.
    $psi.Environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    $psi.Environment['CUDA_VISIBLE_DEVICES'] = $before.GpuUuid
    foreach ($argument in $TaskArguments) {
        [void] $psi.ArgumentList.Add($argument)
    }

    if ($StdoutPath) {
        $stdoutFullPath = [System.IO.Path]::GetFullPath($StdoutPath, $resolvedWorkingDirectory)
        $stdoutParent = Split-Path -Parent $stdoutFullPath
        if ($stdoutParent) {
            New-Item -ItemType Directory -Path $stdoutParent -Force | Out-Null
        }
        $psi.RedirectStandardOutput = $true
    }
    if ($StderrPath) {
        $stderrFullPath = [System.IO.Path]::GetFullPath($StderrPath, $resolvedWorkingDirectory)
        $stderrParent = Split-Path -Parent $stderrFullPath
        if ($stderrParent) {
            New-Item -ItemType Directory -Path $stderrParent -Force | Out-Null
        }
        $psi.RedirectStandardError = $true
    }

    $child = [System.Diagnostics.Process]::new()
    $child.StartInfo = $psi
    if (-not $child.Start()) {
        throw "Failed to start GPU task: $resolvedExecutable"
    }
    $childStarted = $true
    $taskStartTime = $child.StartTime

    $stdoutRead = if ($psi.RedirectStandardOutput) { $child.StandardOutput.ReadToEndAsync() } else { $null }
    $stderrRead = if ($psi.RedirectStandardError) { $child.StandardError.ReadToEndAsync() } else { $null }

    while ($true) {
        $snapshot = Get-MemorySnapshot
        $peakUsedVramMiB = [math]::Max($peakUsedVramMiB, $snapshot.UsedVramMiB)
        $minimumFreeRamMiB = [math]::Min($minimumFreeRamMiB, $snapshot.FreeRamMiB)

        if ($snapshot.UsedVramMiB -gt $MaxUsedVramMiB) {
            $limitReason = "VRAM usage reached $($snapshot.UsedVramMiB) MiB (limit $MaxUsedVramMiB MiB)."
            break
        }
        if ($snapshot.FreeVramMiB -lt $MinFreeVramMiB) {
            $limitReason = "Free VRAM fell to $($snapshot.FreeVramMiB) MiB (minimum $MinFreeVramMiB MiB)."
            break
        }
        if ($snapshot.FreeRamMiB -lt ($MinFreeRamGiB * 1024)) {
            $limitReason = "Free system RAM fell to $($snapshot.FreeRamMiB) MiB (minimum $($MinFreeRamGiB * 1024) MiB)."
            break
        }

        if ($child.WaitForExit($PollMilliseconds)) {
            break
        }
    }

    if ($limitReason -and -not $child.HasExited) {
        try {
            $child.Kill($true)
        } catch [System.InvalidOperationException] {
            # Benign race: the child exited between HasExited and Kill.
        }
    }
    $child.WaitForExit()

    # Kill task-spawned named descendants before waiting for redirected pipes.
    # A descendant that inherited stdout/stderr can otherwise keep ReadToEnd open.
    $leftovers = @(Stop-TaskRelevantProcesses -StartedAt $taskStartTime)
    if ($stdoutRead) {
        if (-not $stdoutRead.Wait(5000)) {
            throw 'Timed out waiting for redirected stdout to close.'
        }
        [System.IO.File]::WriteAllText($stdoutFullPath, $stdoutRead.GetAwaiter().GetResult())
    }
    if ($stderrRead) {
        if (-not $stderrRead.Wait(5000)) {
            throw 'Timed out waiting for redirected stderr to close.'
        }
        [System.IO.File]::WriteAllText($stderrFullPath, $stderrRead.GetAwaiter().GetResult())
    }

    if ($limitReason) {
        throw "GPU task was terminated by the memory guard: $limitReason"
    }
    if ($child.ExitCode -ne 0) {
        throw "GPU task exited with code $($child.ExitCode)."
    }

    if ($leftovers.Count -gt 0) {
        throw "GPU task left relevant processes running; they were terminated: $($leftovers | ConvertTo-Json -Compress)"
    }

    [pscustomobject]@{
        Status = 'completed'
        ExitCode = $child.ExitCode
        PeakUsedVramMiB = $peakUsedVramMiB
        MinimumFreeRamMiB = $minimumFreeRamMiB
        GpuIndex = $GpuIndex
        GpuUuid = $before.GpuUuid
        Lease = $leasePath
    }
} finally {
    try {
        if ($childStarted) {
            try {
                if (-not $child.HasExited) {
                    try {
                        $child.Kill($true)
                    } catch [System.InvalidOperationException] {
                        # Benign race: the child exited between HasExited and Kill.
                    }
                    [void] $child.WaitForExit(5000)
                }
            } finally {
                if ($taskStartTime) {
                    try {
                        [void] @(Stop-TaskRelevantProcesses -StartedAt $taskStartTime)
                    } catch {
                        Write-Warning "Failed during task-owned process cleanup: $($_.Exception.Message)"
                    }
                }
            }
        }
    } finally {
        if ($leaseOwned) {
            try {
                if ($leaseStream) {
                    $leaseStream.Dispose()
                }
            } finally {
                if (Test-Path -LiteralPath $leasePath) {
                    Remove-Item -LiteralPath $leasePath -Force
                }
            }
        }
    }
}
