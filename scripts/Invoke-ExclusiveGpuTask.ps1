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

    [int] $MaxRuntimeSeconds = 0,

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
if ($MaxRuntimeSeconds -lt 0) {
    throw 'MaxRuntimeSeconds must be non-negative.'
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
    'test-qwen38-recurrent-parity',
    'test-qwen38-row-invariance',
    'test-recurrent-state-rollback'
)

if (-not ('AiLoaderGpuGuard.NativeMethods' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace AiLoaderGpuGuard {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    internal struct MEMORYSTATUSEX {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct IO_COUNTERS {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct JOBOBJECT_BASIC_ACCOUNTING_INFORMATION {
        public long TotalUserTime;
        public long TotalKernelTime;
        public long ThisPeriodTotalUserTime;
        public long ThisPeriodTotalKernelTime;
        public uint TotalPageFaultCount;
        public uint TotalProcesses;
        public uint ActiveProcesses;
        public uint TotalTerminatedProcesses;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    internal struct STARTUPINFO {
        public uint cb;
        public string lpReserved;
        public string lpDesktop;
        public string lpTitle;
        public uint dwX;
        public uint dwY;
        public uint dwXSize;
        public uint dwYSize;
        public uint dwXCountChars;
        public uint dwYCountChars;
        public uint dwFillAttribute;
        public uint dwFlags;
        public ushort wShowWindow;
        public ushort cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct PROCESS_INFORMATION {
        public IntPtr hProcess;
        public IntPtr hThread;
        public uint dwProcessId;
        public uint dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct SECURITY_ATTRIBUTES {
        public uint nLength;
        public IntPtr lpSecurityDescriptor;
        [MarshalAs(UnmanagedType.Bool)] public bool bInheritHandle;
    }

    public sealed class SuspendedProcess : IDisposable {
        public IntPtr ProcessHandle { get; internal set; }
        internal IntPtr ThreadHandle;
        public int ProcessId { get; internal set; }
        public bool Resumed { get; private set; }

        public void Resume() {
            if (Resumed) {
                return;
            }
            if (NativeMethods.ResumeThreadHandle(ThreadHandle) == uint.MaxValue) {
                throw new Win32Exception(Marshal.GetLastWin32Error(), "ResumeThread failed");
            }
            Resumed = true;
            NativeMethods.CloseNativeHandle(ThreadHandle);
            ThreadHandle = IntPtr.Zero;
        }

        public int GetExitCode() {
            return NativeMethods.GetNativeExitCode(ProcessHandle);
        }

        public void Dispose() {
            if (!Resumed && ProcessHandle != IntPtr.Zero) {
                NativeMethods.TerminateNativeProcess(ProcessHandle, 0x1005);
            }
            if (ThreadHandle != IntPtr.Zero) {
                NativeMethods.CloseNativeHandle(ThreadHandle);
                ThreadHandle = IntPtr.Zero;
            }
            if (ProcessHandle != IntPtr.Zero) {
                NativeMethods.CloseNativeHandle(ProcessHandle);
                ProcessHandle = IntPtr.Zero;
            }
        }
    }

    public static class NativeMethods {
        private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
        private const int JobObjectBasicAccountingInformation = 1;
        private const int JobObjectExtendedLimitInformation = 9;
        private const uint CREATE_SUSPENDED = 0x00000004;
        private const uint CREATE_UNICODE_ENVIRONMENT = 0x00000400;
        private const uint STARTF_USESTDHANDLES = 0x00000100;
        private const uint GENERIC_WRITE = 0x40000000;
        private const uint FILE_SHARE_READ = 0x00000001;
        private const uint CREATE_ALWAYS = 2;
        private const uint FILE_ATTRIBUTE_NORMAL = 0x00000080;
        private const int STD_INPUT_HANDLE = -10;
        private const int STD_OUTPUT_HANDLE = -11;
        private const int STD_ERROR_HANDLE = -12;
        private static readonly IntPtr InvalidHandleValue = new IntPtr(-1);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObject(IntPtr securityAttributes, string name);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr OpenJobObject(uint desiredAccess, bool inheritHandle, string name);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool QueryInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length, IntPtr returnLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool TerminateJobObject(IntPtr job, uint exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr handle);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool CreateProcessW(
            string applicationName,
            System.Text.StringBuilder commandLine,
            IntPtr processAttributes,
            IntPtr threadAttributes,
            bool inheritHandles,
            uint creationFlags,
            IntPtr environment,
            string currentDirectory,
            ref STARTUPINFO startupInfo,
            out PROCESS_INFORMATION processInformation);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern uint ResumeThread(IntPtr thread);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool TerminateProcess(IntPtr process, uint exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateFileW(
            string path,
            uint desiredAccess,
            uint shareMode,
            ref SECURITY_ATTRIBUTES securityAttributes,
            uint creationDisposition,
            uint flagsAndAttributes,
            IntPtr templateFile);

        [DllImport("kernel32.dll")]
        private static extern IntPtr GetStdHandle(int standardHandle);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX buffer);

        private static void ThrowLastError(string operation) {
            throw new Win32Exception(Marshal.GetLastWin32Error(), operation);
        }

        private static string QuoteArgument(string value) {
            if (value.Length > 0 && value.IndexOfAny(new[] { ' ', '\t', '\n', '\v', '"' }) < 0) {
                return value;
            }
            System.Text.StringBuilder result = new System.Text.StringBuilder();
            result.Append('"');
            int slashes = 0;
            foreach (char character in value) {
                if (character == '\\') {
                    slashes++;
                    continue;
                }
                if (character == '"') {
                    result.Append('\\', slashes * 2 + 1);
                    result.Append('"');
                    slashes = 0;
                    continue;
                }
                result.Append('\\', slashes);
                slashes = 0;
                result.Append(character);
            }
            result.Append('\\', slashes * 2);
            result.Append('"');
            return result.ToString();
        }

        private static IntPtr OpenOutput(string path, ref SECURITY_ATTRIBUTES attributes) {
            if (String.IsNullOrEmpty(path)) {
                return IntPtr.Zero;
            }
            IntPtr handle = CreateFileW(
                path,
                GENERIC_WRITE,
                FILE_SHARE_READ,
                ref attributes,
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                IntPtr.Zero);
            if (handle == InvalidHandleValue) {
                ThrowLastError("CreateFile for redirected output failed");
            }
            return handle;
        }

        public static SuspendedProcess StartSuspended(
            string executable,
            string[] arguments,
            string workingDirectory,
            System.Collections.Generic.IDictionary<string, string> environment,
            string stdoutPath,
            string stderrPath) {
            if (!String.IsNullOrEmpty(stdoutPath) && String.Equals(stdoutPath, stderrPath, StringComparison.OrdinalIgnoreCase)) {
                throw new ArgumentException("stdout and stderr paths must differ");
            }
            SECURITY_ATTRIBUTES attributes = new SECURITY_ATTRIBUTES();
            attributes.nLength = (uint) Marshal.SizeOf<SECURITY_ATTRIBUTES>();
            attributes.bInheritHandle = true;
            IntPtr stdout = IntPtr.Zero;
            IntPtr stderr = IntPtr.Zero;
            IntPtr environmentBlock = IntPtr.Zero;
            PROCESS_INFORMATION processInformation = new PROCESS_INFORMATION();
            try {
                stdout = OpenOutput(stdoutPath, ref attributes);
                stderr = OpenOutput(stderrPath, ref attributes);
                STARTUPINFO startup = new STARTUPINFO();
                startup.cb = (uint) Marshal.SizeOf<STARTUPINFO>();
                startup.dwFlags = STARTF_USESTDHANDLES;
                startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
                startup.hStdOutput = stdout != IntPtr.Zero ? stdout : GetStdHandle(STD_OUTPUT_HANDLE);
                startup.hStdError = stderr != IntPtr.Zero ? stderr : GetStdHandle(STD_ERROR_HANDLE);

                System.Text.StringBuilder commandLine = new System.Text.StringBuilder(QuoteArgument(executable));
                foreach (string argument in arguments) {
                    commandLine.Append(' ');
                    commandLine.Append(QuoteArgument(argument));
                }
                System.Text.StringBuilder block = new System.Text.StringBuilder();
                foreach (System.Collections.Generic.KeyValuePair<string, string> entry in environment) {
                    block.Append(entry.Key);
                    block.Append('=');
                    block.Append(entry.Value);
                    block.Append('\0');
                }
                block.Append('\0');
                byte[] environmentBytes = System.Text.Encoding.Unicode.GetBytes(block.ToString());
                environmentBlock = Marshal.AllocHGlobal(environmentBytes.Length);
                Marshal.Copy(environmentBytes, 0, environmentBlock, environmentBytes.Length);

                if (!CreateProcessW(
                    executable,
                    commandLine,
                    IntPtr.Zero,
                    IntPtr.Zero,
                    true,
                    CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
                    environmentBlock,
                    workingDirectory,
                    ref startup,
                    out processInformation)) {
                    ThrowLastError("CreateProcessW failed");
                }
                return new SuspendedProcess {
                    ProcessHandle = processInformation.hProcess,
                    ThreadHandle = processInformation.hThread,
                    ProcessId = (int) processInformation.dwProcessId,
                };
            } finally {
                if (environmentBlock != IntPtr.Zero) {
                    Marshal.FreeHGlobal(environmentBlock);
                }
                if (stdout != IntPtr.Zero) {
                    CloseHandle(stdout);
                }
                if (stderr != IntPtr.Zero) {
                    CloseHandle(stderr);
                }
            }
        }

        internal static uint ResumeThreadHandle(IntPtr thread) {
            return ResumeThread(thread);
        }

        internal static void TerminateNativeProcess(IntPtr process, uint exitCode) {
            if (!TerminateProcess(process, exitCode) && Marshal.GetLastWin32Error() != 5) {
                ThrowLastError("TerminateProcess failed");
            }
        }

        internal static int GetNativeExitCode(IntPtr process) {
            uint exitCode;
            if (!GetExitCodeProcess(process, out exitCode)) {
                ThrowLastError("GetExitCodeProcess failed");
            }
            return unchecked((int) exitCode);
        }

        internal static void CloseNativeHandle(IntPtr handle) {
            if (handle != IntPtr.Zero && !CloseHandle(handle)) {
                ThrowLastError("CloseHandle failed");
            }
        }

        public static ulong GetAvailablePhysicalMemory() {
            MEMORYSTATUSEX status = new MEMORYSTATUSEX();
            status.dwLength = (uint) Marshal.SizeOf<MEMORYSTATUSEX>();
            if (!GlobalMemoryStatusEx(ref status)) {
                ThrowLastError("GlobalMemoryStatusEx failed");
            }
            return status.ullAvailPhys;
        }

        public static IntPtr CreateKillOnCloseJob(string name) {
            IntPtr job = CreateJobObject(IntPtr.Zero, name);
            if (job == IntPtr.Zero) {
                ThrowLastError("CreateJobObject failed");
            }
            if (Marshal.GetLastWin32Error() == 183) {
                CloseHandle(job);
                throw new InvalidOperationException("exclusive GPU Job Object already exists");
            }
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            int size = Marshal.SizeOf<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try {
                Marshal.StructureToPtr(limits, buffer, false);
                if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, buffer, (uint) size)) {
                    int error = Marshal.GetLastWin32Error();
                    CloseHandle(job);
                    throw new Win32Exception(error, "SetInformationJobObject failed");
                }
            } finally {
                Marshal.FreeHGlobal(buffer);
            }
            return job;
        }

        public static void AssignProcess(IntPtr job, IntPtr process) {
            if (!AssignProcessToJobObject(job, process)) {
                ThrowLastError("AssignProcessToJobObject failed");
            }
        }

        public static uint GetActiveProcessCount(IntPtr job) {
            int size = Marshal.SizeOf<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>();
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try {
                if (!QueryInformationJobObject(job, JobObjectBasicAccountingInformation, buffer, (uint) size, IntPtr.Zero)) {
                    ThrowLastError("QueryInformationJobObject failed");
                }
                return Marshal.PtrToStructure<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>(buffer).ActiveProcesses;
            } finally {
                Marshal.FreeHGlobal(buffer);
            }
        }

        public static int GetNamedJobActiveProcessCount(string name) {
            IntPtr job = OpenJobObject(0x0004, false, name);
            if (job == IntPtr.Zero) {
                return -1;
            }
            try {
                return (int) GetActiveProcessCount(job);
            } finally {
                CloseHandle(job);
            }
        }

        public static void TerminateJob(IntPtr job, uint exitCode) {
            if (!TerminateJobObject(job, exitCode)) {
                ThrowLastError("TerminateJobObject failed");
            }
        }

        public static void CloseJob(IntPtr job) {
            if (job != IntPtr.Zero && !CloseHandle(job)) {
                ThrowLastError("CloseHandle failed");
            }
        }
    }
}
'@
}

function Get-MemorySnapshot {
    $freeRamMiB = [math]::Floor(
        [double] [AiLoaderGpuGuard.NativeMethods]::GetAvailablePhysicalMemory() / 1MB)

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

function Stop-JobProcessesAndWait {
    param(
        [Parameter(Mandatory = $true)][IntPtr] $Job,
        [Parameter(Mandatory = $true)][uint32] $ExitCode,
        [int] $TimeoutMilliseconds = 5000
    )

    $active = [int] [AiLoaderGpuGuard.NativeMethods]::GetActiveProcessCount($Job)
    if ($active -eq 0) {
        return 0
    }
    [AiLoaderGpuGuard.NativeMethods]::TerminateJob($Job, $ExitCode)
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    while (
        [AiLoaderGpuGuard.NativeMethods]::GetActiveProcessCount($Job) -gt 0 -and
        $stopwatch.ElapsedMilliseconds -lt $TimeoutMilliseconds) {
        Start-Sleep -Milliseconds 25
    }
    if ([AiLoaderGpuGuard.NativeMethods]::GetActiveProcessCount($Job) -gt 0) {
        throw 'Guarded Job Object still contains processes after cleanup.'
    }
    $active
}

function Write-LeaseRecord {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileStream] $Stream,
        [Parameter(Mandatory = $true)] $Record
    )
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($Record | ConvertTo-Json -Compress))
    $Stream.Position = 0
    $Stream.SetLength(0)
    $Stream.Write($bytes, 0, $bytes.Length)
    $Stream.Flush($true)
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

    try {
        $createdAt = [datetimeoffset] $lease.CreatedAt
    } catch {
        throw "GPU lease has an invalid CreatedAt timestamp: $leasePath"
    }

    if ($lease.JobName) {
        $activeJobProcesses = [AiLoaderGpuGuard.NativeMethods]::GetNamedJobActiveProcessCount(
            [string] $lease.JobName)
        if ($activeJobProcesses -gt 0) {
            throw "GPU lease Job Object still has $activeJobProcesses active process(es): $leasePath"
        }
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
$jobHandle = [IntPtr]::Zero
$jobAssigned = $false
$jobCleanupComplete = $true
$child = $null
$suspendedChild = $null
$childStarted = $false
$limitReason = $null
$descendantsTerminated = 0
$peakUsedVramMiB = 0
$minimumFreeVramMiB = [int]::MaxValue
$minimumFreeRamMiB = [int]::MaxValue
$leaseNonceBytes = New-Object byte[] 32
$leaseNonceGenerator = [Security.Cryptography.RandomNumberGenerator]::Create()
try {
    $leaseNonceGenerator.GetBytes($leaseNonceBytes)
} finally {
    $leaseNonceGenerator.Dispose()
}
$leaseNonce = ([BitConverter]::ToString($leaseNonceBytes) -replace '-', '').ToLowerInvariant()
$jobName = "Local\AiLoaderGpuGuard-$leaseNonce"
$ownerStartTimeUtc = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
$leaseRecord = [ordered]@{
    LeaseVersion = 2
    Nonce = $leaseNonce
    OwnerPid = $PID
    CreatedAt = (Get-Date).ToUniversalTime().ToString('o')
    OwnerStartTimeUtc = $ownerStartTimeUtc
    JobName = $jobName
    Executable = $Executable
    GpuIndex = $GpuIndex
    GpuUuid = $null
}

try {
    $leaseStream = [System.IO.File]::Open(
        $leasePath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::Read)
    $leaseOwned = $true
    $jobHandle = [AiLoaderGpuGuard.NativeMethods]::CreateKillOnCloseJob($jobName)
    $jobCleanupComplete = $false
    Write-LeaseRecord -Stream $leaseStream -Record $leaseRecord

    $conflicts = Get-RelevantProcesses
    if ($conflicts.Count -gt 0) {
        throw "Refusing GPU task because another LLM/test process exists: $($conflicts | ConvertTo-Json -Compress)"
    }

    $before = Get-MemorySnapshot
    $leaseRecord.GpuUuid = $before.GpuUuid
    Write-LeaseRecord -Stream $leaseStream -Record $leaseRecord
    $peakUsedVramMiB = $before.UsedVramMiB
    $minimumFreeVramMiB = $before.FreeVramMiB
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
    $environment = [System.Collections.Generic.SortedDictionary[string,string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in [System.Environment]::GetEnvironmentVariables().GetEnumerator()) {
        $environment[[string] $entry.Key] = [string] $entry.Value
    }
    # Monitor and expose the same physical GPU by UUID. The guarded child
    # therefore sees the selected adapter as CUDA0 and cannot allocate elsewhere.
    $environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    $environment['CUDA_VISIBLE_DEVICES'] = $before.GpuUuid
    $environment['AI_LOADER_EXCLUSIVE_GPU_GUARD'] = '1'
    $environment['AI_LOADER_EXCLUSIVE_GPU_UUID'] = $before.GpuUuid
    $environment['AI_LOADER_EXCLUSIVE_GPU_LEASE_PATH'] = $leasePath
    $environment['AI_LOADER_EXCLUSIVE_GPU_LEASE_NONCE'] = $leaseNonce
    $environment['AI_LOADER_EXCLUSIVE_GPU_OWNER_PID'] = [string] $PID
    $environment['AI_LOADER_EXCLUSIVE_GPU_JOB_NAME'] = $jobName

    $stdoutFullPath = ''
    if ($StdoutPath) {
        $stdoutFullPath = if ([System.IO.Path]::IsPathRooted($StdoutPath)) {
            [System.IO.Path]::GetFullPath($StdoutPath)
        } else {
            [System.IO.Path]::GetFullPath((Join-Path $resolvedWorkingDirectory $StdoutPath))
        }
        $stdoutParent = Split-Path -Parent $stdoutFullPath
        if ($stdoutParent) {
            New-Item -ItemType Directory -Path $stdoutParent -Force | Out-Null
        }
    }
    $stderrFullPath = ''
    if ($StderrPath) {
        $stderrFullPath = if ([System.IO.Path]::IsPathRooted($StderrPath)) {
            [System.IO.Path]::GetFullPath($StderrPath)
        } else {
            [System.IO.Path]::GetFullPath((Join-Path $resolvedWorkingDirectory $StderrPath))
        }
        $stderrParent = Split-Path -Parent $stderrFullPath
        if ($stderrParent) {
            New-Item -ItemType Directory -Path $stderrParent -Force | Out-Null
        }
    }

    $suspendedChild = [AiLoaderGpuGuard.NativeMethods]::StartSuspended(
        $resolvedExecutable,
        [string[]] $TaskArguments,
        $resolvedWorkingDirectory,
        $environment,
        $stdoutFullPath,
        $stderrFullPath)
    $childStarted = $true
    $child = [System.Diagnostics.Process]::GetProcessById($suspendedChild.ProcessId)
    [AiLoaderGpuGuard.NativeMethods]::AssignProcess($jobHandle, $suspendedChild.ProcessHandle)
    $jobAssigned = $true
    $suspendedChild.Resume()
    $taskStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $postStart = Get-MemorySnapshot
    $peakUsedVramMiB = [math]::Max($peakUsedVramMiB, $postStart.UsedVramMiB)
    $minimumFreeVramMiB = [math]::Min($minimumFreeVramMiB, $postStart.FreeVramMiB)
    $minimumFreeRamMiB = [math]::Min($minimumFreeRamMiB, $postStart.FreeRamMiB)
    if ($postStart.UsedVramMiB -gt $MaxUsedVramMiB) {
        $limitReason = "VRAM usage reached $($postStart.UsedVramMiB) MiB (limit $MaxUsedVramMiB MiB)."
    } elseif ($postStart.FreeVramMiB -lt $MinFreeVramMiB) {
        $limitReason = "Free VRAM fell to $($postStart.FreeVramMiB) MiB (minimum $MinFreeVramMiB MiB)."
    } elseif ($postStart.FreeRamMiB -lt ($MinFreeRamGiB * 1024)) {
        $limitReason = "Free system RAM fell to $($postStart.FreeRamMiB) MiB (minimum $($MinFreeRamGiB * 1024) MiB)."
    }

    while (-not $limitReason) {
        if ($child.HasExited) {
            break
        }
        if ($MaxRuntimeSeconds -gt 0 -and $taskStopwatch.Elapsed.TotalSeconds -ge $MaxRuntimeSeconds) {
            $limitReason = "Runtime reached $([math]::Round($taskStopwatch.Elapsed.TotalSeconds, 3)) seconds (limit $MaxRuntimeSeconds seconds)."
            break
        }
        $snapshot = Get-MemorySnapshot
        $peakUsedVramMiB = [math]::Max($peakUsedVramMiB, $snapshot.UsedVramMiB)
        $minimumFreeVramMiB = [math]::Min($minimumFreeVramMiB, $snapshot.FreeVramMiB)
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
        [AiLoaderGpuGuard.NativeMethods]::TerminateJob($jobHandle, 0x1001)
    }
    if (-not $child.WaitForExit(5000)) {
        [AiLoaderGpuGuard.NativeMethods]::TerminateJob($jobHandle, 0x1002)
        if (-not $child.WaitForExit(5000)) {
            throw 'Guarded root process did not exit within the cleanup timeout.'
        }
    }

    $after = Get-MemorySnapshot
    $peakUsedVramMiB = [math]::Max($peakUsedVramMiB, $after.UsedVramMiB)
    $minimumFreeVramMiB = [math]::Min($minimumFreeVramMiB, $after.FreeVramMiB)
    $minimumFreeRamMiB = [math]::Min($minimumFreeRamMiB, $after.FreeRamMiB)

    $descendantsTerminated = Stop-JobProcessesAndWait -Job $jobHandle -ExitCode 0x1003
    $jobCleanupComplete = $true
    $childExitCode = $suspendedChild.GetExitCode()
    $suspendedChild.Dispose()
    $suspendedChild = $null
    if ($limitReason) {
        $guardKind = if ($limitReason -like 'Runtime reached*') { 'runtime guard' } else { 'memory guard' }
        throw "GPU task was terminated by the $guardKind`: $limitReason"
    }
    if ($childExitCode -ne 0) {
        throw "GPU task exited with code $childExitCode."
    }

    if ($descendantsTerminated -gt 0) {
        throw "GPU task left $descendantsTerminated owned descendant process(es) running; the Job Object terminated them."
    }

    [pscustomobject]@{
        Status = 'completed'
        ExitCode = $childExitCode
        PeakUsedVramMiB = $peakUsedVramMiB
        MinimumFreeVramMiB = $minimumFreeVramMiB
        MinimumFreeRamMiB = $minimumFreeRamMiB
        GpuIndex = $GpuIndex
        GpuUuid = $before.GpuUuid
        RuntimeSeconds = $taskStopwatch.Elapsed.TotalSeconds
        Lease = $leasePath
    }
} finally {
    try {
        if ($suspendedChild) {
            $suspendedChild.Dispose()
            $suspendedChild = $null
        }
        if ($jobHandle -ne [IntPtr]::Zero) {
            try {
                [void] (Stop-JobProcessesAndWait -Job $jobHandle -ExitCode 0x1004)
                $jobCleanupComplete = $true
            } finally {
                [AiLoaderGpuGuard.NativeMethods]::CloseJob($jobHandle)
                $jobHandle = [IntPtr]::Zero
            }
        }
        if (-not $jobAssigned -and $childStarted -and -not $child.HasExited) {
            try {
                $child.Kill($true)
                [void] $child.WaitForExit(5000)
            } catch {
                Write-Warning "Failed during pre-assignment process cleanup: $($_.Exception.Message)"
            }
        }
    } finally {
        if ($leaseOwned -and $jobCleanupComplete) {
            try {
                if ($leaseStream) {
                    $leaseStream.Position = 0
                    $reader = [System.IO.StreamReader]::new(
                        $leaseStream,
                        [System.Text.Encoding]::UTF8,
                        $false,
                        4096,
                        $true)
                    try {
                        $currentLease = $reader.ReadToEnd() | ConvertFrom-Json
                        if ($currentLease.Nonce -ne $leaseNonce -or $currentLease.OwnerPid -ne $PID) {
                            throw 'GPU lease identity changed before release.'
                        }
                    } finally {
                        $reader.Dispose()
                    }
                    $leaseStream.Dispose()
                }
            } finally {
                if (Test-Path -LiteralPath $leasePath) {
                    Remove-Item -LiteralPath $leasePath -Force
                }
            }
        } elseif ($leaseOwned) {
            Write-Warning 'GPU lease was retained because Job Object cleanup did not reach zero active processes.'
        }
    }
}
