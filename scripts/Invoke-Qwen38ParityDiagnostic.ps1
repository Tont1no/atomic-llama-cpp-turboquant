[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Executable,
    [Parameter(Mandatory = $true)] [string] $ModelPath,
    [Parameter(Mandatory = $true)] [string] $TargetOnlyWavesPath,
    [Parameter(Mandatory = $true)] [string] $OutputPath,
    [ValidateRange(1, 247)] [int] $PredictionStart = 176,
    [ValidateRange(8, 248)] [int] $PredictionCount = 24,
    [ValidateRange(1, 248)] [int] $RollbackStart = 184,
    [ValidateRange(18, 72)] [int] $MinFreeRamGiB = 18,
    [ValidateRange(4096, 65536)] [int] $MinFreeVramMiB = 4096,
    [ValidateRange(512, 28672)] [int] $MaxUsedVramMiB = 28672,
    [ValidateRange(1, 7200)] [int] $MaxRuntimeSeconds = 1800,
    [switch] $Qwen35Nvfp4FfnProjectionDiagnostic,
    [switch] $Qwen35Bf16RecurrentProjectionDiagnostic,
    [switch] $Qwen35FullAttentionVecDiagnostic,
    [switch] $LayerBoundaryBisection,
    [switch] $RecurrentLayerInternalRefine,
    [switch] $FullAttentionInternalRefine,
    [switch] $Force
)

$ErrorActionPreference = 'Stop'

# Immutable trust anchors for the reviewed row-invariant 2026-08-20 P1 target trace.
$TrustedQualificationIdentity = 'aab1f6698569cd26ef92cf4b6d99ac996397af06cc3c169e1ecd5ad3fe23f639'
$TrustedManifestSha256 = '67da6391d2240a058562973def2949721fd9c336308d996f5e3b0de73f2405c0'
$TrustedValidationSha256 = 'fe38d4bc2702e2a5129235927f550e0269e482c121df3aeda90fbc91c0fc0cbb'
$TrustedTraceGuardReceiptSha256 = '667728f1bcd38da34830f3d416924c446ec72d9f607e6f9779b1e626b0e74492'
$TrustedModelSha256 = '46e0ec4bdab3907346fbb486b1e4f78b49a66b245374aa0d3d628a8d099c0336'
$TrustedWavesSha256 = '1e5f7b997354cebfded587fdbfc6339737c63fc025760a7f75b224dbada5f348'
$TrustedTokenTraceSha256 = '80d896aed5f33f8fb90964caeadaf1f0accb72f464c4acfe9bae642f2b10200f'
$TrustedCommandSha256 = '640418e80973413b3f7b95334a8ecf0aa639100e05339d27b5c4ec9ac0704284'
$TrustedReceiptSha256 = 'c9c1fd02566478c8eb53a75c72ba5ac671ccd1111057d8c6891c75503569a9b9'
$TrustedRuntimeBundleSha256 = '429fec7fbc3ed32cb498433d6da760a4888fb286ebd09c4b4b32540aa4bd4c1c'
$TrustedRuntimeBundleCount = 10
$TrustedDiagnosticExecutableSha256 = 'bc639a16a17da220d6ca83b7956f7db8a86e6c714ada254a246544abe98ee3f2'
$TrustedGuardSha256 = '3120e9fe7c247250a440389d0450c71741718b391d435a270fd2f16d33dc1463'

# Filled only by the separately reviewed opt-in diagnostic build. The normal
# path above remains byte-identical to the target-trace runtime.
$TrustedProjectionDiagnosticExecutableSha256 = '8dd01510919d8af3c00abc3fa9cd4df5db8adb4da7f9add5d7704fc21152d007'
$TrustedProjectionServerSha256 = '3957122672ce2def719a58b321e70334ec048e0b4d9f435c11b1b5d2d61f5b9d'
$TrustedProjectionRuntimeBundleSha256 = '05ba7aa482de89760a7208cf96b706c9ce0e6dcb4dd486ee58209e7ad0552241'
$TrustedProjectionRuntime = [ordered]@{
    'cublas64_13.dll' = '91200c2ff57b8477e94254a4501030d75e1bb35a2de3a476e7ab66df897cb046'
    'cublasLt64_13.dll' = '8f54d7b3e5173bf2659f45bad6ae789ffef8218ab6254f10c0cc0cc5e3ec874c'
    'ggml-base.dll' = 'bfd26fdf4d649489a6b002a4c8cc3a1588b086f1b3ba224eb729f914c98af10e'
    'ggml-cpu.dll' = 'b647b06bd8c215393cad2e4f9fb2b36e2c9f8338057cc5100ea7011a5e045541'
    'ggml-cuda.dll' = '2283427025dd9ee3bdc5b6318a06aff3fff62f882f0d0c6198226dbe818d5b7c'
    'ggml.dll' = 'fc367df4697a1f4bc9b996dc224402ca7fec4d3501153fe67e815e4086566edc'
    'llama-common.dll' = '62efb05af206ca1889b663b6b5ce78cdc0b971247c5ce019afc5389d1420d68f'
    'llama-server-impl.dll' = '36e4a42787ade6dd1978f27059df3ae4f7882d53ee8e4b933e178869786fd5f8'
    'llama.dll' = '30cb9e4f47fba2a4dc72851d91f73a3494f4d851090f098f644f5b6aeadc5330'
    'mtmd.dll' = '12c49c4fa900254bca491156829564df4439c644b0c0f06b1ed969e41a5166d7'
}

$useCandidateRuntime = [bool] ($Qwen35Nvfp4FfnProjectionDiagnostic -or
    $Qwen35Bf16RecurrentProjectionDiagnostic -or $Qwen35FullAttentionVecDiagnostic)
$diagnosticCandidateIdentity = if ($Qwen35Nvfp4FfnProjectionDiagnostic -and
        $Qwen35Bf16RecurrentProjectionDiagnostic -and $Qwen35FullAttentionVecDiagnostic) {
    'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PLUS_FATTN_VEC_V3'
} elseif ($Qwen35Nvfp4FfnProjectionDiagnostic -and $Qwen35Bf16RecurrentProjectionDiagnostic) {
    'QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PROJECTIONS_V2'
} elseif ($Qwen35Nvfp4FfnProjectionDiagnostic) {
    'QWEN35_NVFP4_FFN_PROJECTIONS_V1'
} elseif ($Qwen35Bf16RecurrentProjectionDiagnostic) {
    'QWEN35_BF16_RECURRENT_PROJECTIONS_V1'
} else { '' }

if ($Qwen35FullAttentionVecDiagnostic -and
        (-not $Qwen35Nvfp4FfnProjectionDiagnostic -or -not $Qwen35Bf16RecurrentProjectionDiagnostic)) {
    throw 'Qwen35FullAttentionVecDiagnostic requires both existing candidate switches and the explicit V3 identity.'
}

function Resolve-RequiredFile([string] $Path, [string] $Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-Sha256([string] $Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-StringSha256([string] $Value) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Value)
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Test-LoopbackPortFree([int] $Port) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    $listener.Server.ExclusiveAddressUse = $true
    try {
        $listener.Start()
        return $true
    } catch [System.Net.Sockets.SocketException] {
        return $false
    } finally {
        try { $listener.Stop() } catch {}
    }
}

function Test-StringArrayEqual([object[]] $Left, [object[]] $Right) {
    if ($Left.Count -ne $Right.Count) { return $false }
    for ($i = 0; $i -lt $Left.Count; ++$i) {
        if ([string] $Left[$i] -cne [string] $Right[$i]) { return $false }
    }
    return $true
}

function Get-CommandArgument([object[]] $Arguments, [string] $Name) {
    $indices = @()
    for ($i = 0; $i -lt $Arguments.Count; ++$i) {
        if ([string] $Arguments[$i] -ceq $Name) { $indices += $i }
    }
    if ($indices.Count -ne 1 -or ($indices[0] + 1) -ge $Arguments.Count) {
        throw "Expected exactly one value for qualification argument $Name."
    }
    return [string] $Arguments[$indices[0] + 1]
}

function Assert-CommandFlag([object[]] $Arguments, [string] $Name) {
    if (@($Arguments | Where-Object { [string] $_ -ceq $Name }).Count -ne 1) {
        throw "Expected exactly one qualification flag $Name."
    }
}

$executablePath = Resolve-RequiredFile $Executable 'Diagnostic executable'
$modelResolved = Resolve-RequiredFile $ModelPath 'Target model'
$wavesResolved = Resolve-RequiredFile $TargetOnlyWavesPath 'Target-only waves artifact'
$guardPath = Resolve-RequiredFile (Join-Path $PSScriptRoot 'Invoke-ExclusiveGpuTask.ps1') 'Exclusive GPU guard'
$outputResolved = [System.IO.Path]::GetFullPath($OutputPath)
$stderrPath = "$outputResolved.stderr.log"
$guardReceiptPath = "$outputResolved.guard-receipt.json"
$outputPaths = @($outputResolved, $stderrPath, $guardReceiptPath)

$trustedLaunchExecutableSha256 = if ($useCandidateRuntime) {
    $TrustedProjectionDiagnosticExecutableSha256
} else {
    $TrustedDiagnosticExecutableSha256
}
if ((Get-Sha256 $executablePath) -cne $trustedLaunchExecutableSha256) {
    throw 'Diagnostic executable SHA-256 does not match the reviewed CUDA build.'
}
if ((Get-Sha256 $guardPath) -cne $TrustedGuardSha256) {
    throw 'Exclusive GPU guard SHA-256 does not match the reviewed wrapper dependency.'
}

if (-not (Test-LoopbackPortFree 18136)) {
    throw 'Qualification port 18136 is already occupied.'
}
if (($PredictionCount % 8) -ne 0) {
    throw 'PredictionCount must be divisible by 8 so every width uses complete chunks.'
}
$layerCandidateCount = [int] [bool] $Qwen35Nvfp4FfnProjectionDiagnostic +
    [int] [bool] $Qwen35Bf16RecurrentProjectionDiagnostic +
    [int] [bool] $Qwen35FullAttentionVecDiagnostic
if ($LayerBoundaryBisection -and ($layerCandidateCount -lt 1 -or $PredictionCount -ne 8)) {
    throw 'LayerBoundaryBisection requires at least one explicit reviewed Qwen35 candidate and PredictionCount=8.'
}
if ($RecurrentLayerInternalRefine -and -not $LayerBoundaryBisection) {
    throw 'RecurrentLayerInternalRefine requires LayerBoundaryBisection.'
}
if ($FullAttentionInternalRefine -and -not $LayerBoundaryBisection) {
    throw 'FullAttentionInternalRefine requires LayerBoundaryBisection.'
}
if ($FullAttentionInternalRefine -and
        (-not $Qwen35Nvfp4FfnProjectionDiagnostic -or -not $Qwen35Bf16RecurrentProjectionDiagnostic)) {
    throw 'FullAttentionInternalRefine requires both explicit candidate switches and the combined V2 identity.'
}
if ($FullAttentionInternalRefine -and $Qwen35FullAttentionVecDiagnostic) {
    throw 'FullAttentionInternalRefine traces the pre-selector combined V2 candidate; do not combine it with the V3 selector.'
}
if ($FullAttentionInternalRefine -and ($PredictionStart -ne 176 -or $PredictionCount -ne 8 -or $RollbackStart -ne 184)) {
    throw 'FullAttentionInternalRefine is pinned to PredictionStart=176, PredictionCount=8, and RollbackStart=184.'
}
if ($RecurrentLayerInternalRefine -and $FullAttentionInternalRefine) {
    throw 'RecurrentLayerInternalRefine and FullAttentionInternalRefine are mutually exclusive.'
}
if (($PredictionStart + $PredictionCount) -gt 256 -or ($RollbackStart + 8) -gt 256) {
    throw 'The requested diagnostic window exceeds the 256-token P1 reference trace.'
}

# Bind the input to the exact immutable P1 target-only qualification layout.
$armDirectory = Split-Path -Parent $wavesResolved
$armsDirectory = Split-Path -Parent $armDirectory
$qualificationRoot = Split-Path -Parent $armsDirectory
$traceGenerationRoot = Split-Path -Parent $qualificationRoot
if ((Split-Path -Leaf $wavesResolved) -cne 'waves.json' -or
        (Split-Path -Leaf $armDirectory) -cne 'target-only' -or
        (Split-Path -Leaf $armsDirectory) -cne 'arms' -or
        (Split-Path -Leaf $qualificationRoot) -cne 'p1-final-v5' -or
        (Split-Path -Leaf $traceGenerationRoot) -cne 'qualification-row-invariant-20260820') {
    throw 'TargetOnlyWavesPath must be qualification-row-invariant-20260820\p1-final-v5\arms\target-only\waves.json.'
}

$manifestPath = Resolve-RequiredFile (Join-Path $qualificationRoot 'manifest.json') 'P1 manifest'
$validationPath = Resolve-RequiredFile (Join-Path $qualificationRoot 'validation.json') 'P1 validation'
$traceGuardReceiptPath = Resolve-RequiredFile (Join-Path $qualificationRoot 'guard-receipt.json') 'P1 trace guard receipt'
$commandPath = Resolve-RequiredFile (Join-Path $armDirectory 'command.json') 'Target-only command artifact'
$receiptPath = Resolve-RequiredFile (Join-Path $armDirectory 'receipt.json') 'Target-only artifact receipt'
if ((Get-Sha256 $commandPath) -cne $TrustedCommandSha256 -or
        (Get-Sha256 $receiptPath) -cne $TrustedReceiptSha256) {
    throw 'Command or receipt SHA-256 does not match the approved P1 target-only run.'
}
if ((Get-Sha256 $validationPath) -cne $TrustedValidationSha256 -or
        (Get-Sha256 $traceGuardReceiptPath) -cne $TrustedTraceGuardReceiptSha256) {
    throw 'Validation or trace guard receipt SHA-256 does not match the reviewed fresh P1 trace.'
}

$protectedPaths = @(
    $executablePath, $modelResolved, $wavesResolved, $manifestPath, $validationPath,
    $traceGuardReceiptPath, $commandPath, $receiptPath, $guardPath
) | ForEach-Object { [System.IO.Path]::GetFullPath($_) }
if (@($protectedPaths | Where-Object { $outputPaths -icontains $_ }).Count -ne 0) {
    throw 'An output or receipt path collides with a protected input artifact.'
}
if (-not $Force -and @($outputPaths | Where-Object { Test-Path -LiteralPath $_ }).Count -ne 0) {
    throw "Output already exists. Use -Force to replace it: $outputResolved"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$traceValidation = Get-Content -LiteralPath $validationPath -Raw | ConvertFrom-Json
$traceGuardReceipt = Get-Content -LiteralPath $traceGuardReceiptPath -Raw | ConvertFrom-Json
$command = Get-Content -LiteralPath $commandPath -Raw | ConvertFrom-Json
$receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
$artifact = Get-Content -LiteralPath $wavesResolved -Raw | ConvertFrom-Json

if ($manifest.schema -cne 'ai-loader-qwen38-dspark-qualification-manifest/v1' -or
        $manifest.components.schema -cne 'ai-loader-qwen38-dspark-qualification-identity/v1' -or
        $manifest.components.tier -cne 'P1' -or
        [string] $manifest.identity -cne $TrustedQualificationIdentity -or
        (Get-Sha256 $manifestPath) -cne $TrustedManifestSha256) {
    throw 'Qualification manifest is not the expected P1 schema/tier.'
}
if ($traceValidation.schema -cne 'ai-loader-qwen38-target-trace-validation/v1' -or
        -not [bool] $traceValidation.valid -or
        [string] $traceValidation.identity -cne $TrustedQualificationIdentity -or
        [string] $traceValidation.qualification_identity -cne $TrustedQualificationIdentity -or
        [string] $traceValidation.manifest_sha256 -cne $TrustedManifestSha256 -or
        [string] $traceValidation.model_sha256 -cne $TrustedModelSha256 -or
        [string] $traceValidation.waves_sha256 -cne $TrustedWavesSha256 -or
        [string] $traceValidation.token_trace_sha256 -cne $TrustedTokenTraceSha256 -or
        [string] $traceValidation.command_sha256 -cne $TrustedCommandSha256 -or
        [string] $traceValidation.receipt_sha256 -cne $TrustedReceiptSha256 -or
        [string] $traceValidation.runtime_bundle_sha256 -cne $TrustedRuntimeBundleSha256 -or
        [int] $traceValidation.runtime_bundle_count -ne $TrustedRuntimeBundleCount) {
    throw 'Fresh P1 validation artifact does not bind every immutable diagnostic input.'
}
if ($traceGuardReceipt.schema -cne 'ai-loader-qwen38-target-trace-guard-receipt/v1' -or
        $traceGuardReceipt.stage -cne 'completed' -or
        $traceGuardReceipt.guard_status -cne 'completed' -or
        $traceGuardReceipt.trace_validation -cne 'passed' -or
        -not [bool] $traceGuardReceipt.guard_started -or
        $null -ne $traceGuardReceipt.error -or
        [string] $traceGuardReceipt.guard_sha256 -cne $TrustedGuardSha256 -or
        [string] $traceGuardReceipt.server_sha256 -cne [string] $manifest.components.server_sha256 -or
        [string] $traceGuardReceipt.model_sha256 -cne $TrustedModelSha256 -or
        [string] $traceGuardReceipt.manifest_sha256 -cne $TrustedManifestSha256 -or
        [string] $traceGuardReceipt.validation_sha256 -cne $TrustedValidationSha256 -or
        $traceGuardReceipt.guard.Status -cne 'completed' -or
        [int] $traceGuardReceipt.guard.ExitCode -ne 0 -or
        [int] $traceGuardReceipt.guard.GpuIndex -ne 0 -or
        [string] $traceGuardReceipt.guard.GpuUuid -cne [string] $manifest.components.gpu.uuid -or
        [string] $manifest.components.server_environment.cuda_visible_devices -cne [string] $manifest.components.gpu.uuid -or
        [int] $traceGuardReceipt.postflight.relevant_processes -ne 0 -or
        -not [bool] $traceGuardReceipt.postflight.port_18136_free -or
        [int] $traceGuardReceipt.postflight.recursive_gpu_leases -ne 0) {
    throw 'Fresh P1 trace guard receipt does not prove the reviewed GPU run and clean postflight.'
}

# The reference trace is valid only with the exact server executable and every
# adjacent DLL that produced it. Refuse the GPU lease before launch if the
# current directory has a missing, extra, renamed, or changed runtime file.
$runtimeBundle = [ordered]@{}
$executableDirectory = Split-Path -Parent $executablePath
$manifestRuntimeEntries = @($manifest.components.runtime_bundle)
if ($manifestRuntimeEntries.Count -eq 0) {
    throw 'P1 manifest must bind the complete adjacent DLL runtime bundle.'
}
if ($manifestRuntimeEntries.Count -ne $TrustedRuntimeBundleCount) {
    throw 'P1 manifest runtime bundle count does not match the reviewed fresh trace.'
}
$manifestRuntimeByName = @{}
$referenceNames = @()
foreach ($entry in $manifestRuntimeEntries) {
    $name = [string] $entry.name
    $sha256 = [string] $entry.sha256
    if ((Split-Path -Leaf $name) -cne $name -or
            [System.IO.Path]::GetExtension($name) -ine '.dll' -or
            $sha256 -cnotmatch '^[0-9a-f]{64}$') {
        throw 'P1 manifest contains an invalid adjacent runtime entry.'
    }
    $referenceNames += $name
    $manifestRuntimeByName[$name.ToLowerInvariant()] = $sha256
}
$referenceNamesFolded = @($referenceNames | ForEach-Object { $_.ToLowerInvariant() })
if (@($referenceNamesFolded | Select-Object -Unique).Count -ne $referenceNames.Count) {
    throw 'P1 manifest runtime bundle contains case-colliding DLL names.'
}
$manifestRuntimeEntries = @($manifestRuntimeEntries | Sort-Object { ([string] $_.name).ToLowerInvariant() })
$expectedRuntimeEntries = if ($useCandidateRuntime) {
    @($TrustedProjectionRuntime.GetEnumerator() | ForEach-Object {
        [pscustomobject]@{ name = [string] $_.Key; sha256 = [string] $_.Value }
    } | Sort-Object { ([string] $_.name).ToLowerInvariant() })
} else {
    @($manifestRuntimeEntries)
}
$referenceRuntimeEntries = @($manifestRuntimeEntries)
$referenceNamesFolded = @($referenceRuntimeEntries | ForEach-Object { ([string] $_.name).ToLowerInvariant() })
$currentRuntimeFiles = @(Get-ChildItem -LiteralPath $executableDirectory -File | Where-Object {
    $_.Extension -ieq '.dll'
} | Sort-Object { $_.Name.ToLowerInvariant() })
$currentNames = @($currentRuntimeFiles | ForEach-Object { $_.Name })
if (-not (Test-StringArrayEqual $referenceNamesFolded @($currentNames | ForEach-Object { $_.ToLowerInvariant() }))) {
    throw 'Adjacent DLL set does not exactly match the reference trace runtime bundle.'
}

foreach ($entry in $expectedRuntimeEntries) {
    $name = [string] $entry.name
    $currentPath = Resolve-RequiredFile (Join-Path $executableDirectory $name) "Adjacent diagnostic runtime $name"
    $runtimeBundle[$name] = [ordered]@{
        path = $currentPath
        reference_sha256 = [string] $manifestRuntimeByName[$name.ToLowerInvariant()]
        expected_sha256 = [string] $entry.sha256
        current_sha256 = Get-Sha256 $currentPath
    }
    $protectedPaths += [System.IO.Path]::GetFullPath($currentPath)
}
$serverPath = Resolve-RequiredFile (Join-Path $executableDirectory 'llama-server.exe') 'Adjacent reference llama-server.exe'
$referenceServerSha256 = [string] $manifest.components.server_sha256
if ($referenceServerSha256 -cnotmatch '^[0-9a-f]{64}$') {
    throw 'P1 manifest does not bind a valid server executable SHA-256.'
}
$currentServerSha256 = Get-Sha256 $serverPath
$expectedServerSha256 = if ($useCandidateRuntime) {
    $TrustedProjectionServerSha256
} else {
    $referenceServerSha256
}
$protectedPaths += [System.IO.Path]::GetFullPath($serverPath)
if (@($protectedPaths | Where-Object { $outputPaths -icontains $_ }).Count -ne 0) {
    throw 'OutputPath or its stderr sidecar collides with the diagnostic runtime bundle.'
}
$referenceBundleLines = @($referenceRuntimeEntries | ForEach-Object {
    (([string] $_.name).ToLowerInvariant() + ':' + ([string] $_.sha256).ToLowerInvariant())
})
$currentBundleLines = @($referenceRuntimeEntries | ForEach-Object {
    $item = $runtimeBundle[[string] $_.name]
    (([string] $_.name).ToLowerInvariant() + ':' + $item.current_sha256)
})
$referenceBundleSha256 = Get-StringSha256 ($referenceBundleLines -join "`n")
$currentBundleSha256 = Get-StringSha256 ($currentBundleLines -join "`n")
$runtimeBundleMatchesReference = $currentServerSha256 -ceq $referenceServerSha256 -and
        $currentBundleSha256 -ceq $referenceBundleSha256
$expectedBundleSha256 = if ($useCandidateRuntime) {
    $TrustedProjectionRuntimeBundleSha256
} else {
    $referenceBundleSha256
}
$runtimeBundleMatchesExpected = $currentServerSha256 -ceq $expectedServerSha256 -and
        $currentBundleSha256 -ceq $expectedBundleSha256 -and
        @($runtimeBundle.GetEnumerator() | Where-Object {
            $_.Value.current_sha256 -cne $_.Value.expected_sha256
        }).Count -eq 0
if (-not $runtimeBundleMatchesExpected) {
    $changed = @($runtimeBundle.GetEnumerator() | Where-Object {
        $_.Value.current_sha256 -cne $_.Value.expected_sha256
    } | ForEach-Object { $_.Key })
    if ($currentServerSha256 -cne $expectedServerSha256) { $changed += 'llama-server.exe' }
    throw ('Current runtime is not identical to the pinned diagnostic bundle: ' + ($changed -join ', '))
}
if ($command.schema -cne 'ai-loader-qwen38-dspark-qualification-arm/v1' -or
        $command.arm -cne 'target-only' -or $command.tier -cne 'P1' -or
        [bool] $command.dynamic_rs) {
    throw 'Command artifact is not the static P1 target-only arm.'
}
if ($receipt.schema -cne 'ai-loader-qwen38-dspark-qualification-arm/v1' -or
        $receipt.arm -cne 'target-only' -or $receipt.tier -cne 'P1' -or
        [bool] $receipt.dynamic_rs) {
    throw 'Receipt is not the static P1 target-only arm.'
}
if ($artifact.schema -cne 'ai-loader-qwen38-dspark-qualification-arm/v1') {
    throw 'Waves artifact schema does not match the P1 arm schema.'
}

$modelSha256 = Get-Sha256 $modelResolved
if ($modelSha256 -cne $TrustedModelSha256 -or
        [string] $manifest.components.target_model_sha256 -cne $modelSha256) {
    throw 'ModelPath SHA-256 does not match the P1 target model identity.'
}
if ((Get-Sha256 $wavesResolved) -cne $TrustedWavesSha256) {
    throw 'Target-only waves SHA-256 does not match the approved P1 trace artifact.'
}

$commandArguments = @($command.arguments)
if ($commandArguments.Count -eq 0 -or @($commandArguments | Where-Object { [string] $_ -like '--spec-*' }).Count -ne 0) {
    throw 'Target-only command is empty or contains speculative arguments.'
}
$targetModelFromCommand = [System.IO.Path]::GetFullPath((Get-CommandArgument $commandArguments '--model'))
if ($targetModelFromCommand -ine $modelResolved) {
    throw 'ModelPath does not match the target-only command model path.'
}
$expectedValues = [ordered]@{
    '--ctx-size' = '2048'; '--parallel' = '1'; '--cache-type-k' = 'q8_0';
    '--cache-type-v' = 'q8_0'; '--flash-attn' = 'on'; '--batch-size' = '2048';
    '--ubatch-size' = '128'; '--n-gpu-layers' = 'all'; '--device' = 'CUDA0';
    '--split-mode' = 'none'; '--fit' = 'off'; '--cache-ram' = '0';
    '--ctx-checkpoints' = '0'; '--reasoning' = 'off'
}
foreach ($entry in $expectedValues.GetEnumerator()) {
    if ((Get-CommandArgument $commandArguments $entry.Key) -cne $entry.Value) {
        throw "Target-only command has the wrong value for $($entry.Key)."
    }
}
Assert-CommandFlag $commandArguments '--kv-unified'
foreach ($flag in @('--no-cache-idle-slots', '--no-cache-prompt', '--no-webui', '--metrics')) {
    Assert-CommandFlag $commandArguments $flag
}

$manifestTargetArm = @($manifest.components.arm_arguments | Where-Object { $_.name -ceq 'target-only' })
if ($manifestTargetArm.Count -ne 1 -or
        -not (Test-StringArrayEqual @($manifestTargetArm[0].arguments) $commandArguments)) {
    throw 'Manifest and command artifact disagree on the target-only command.'
}

# Verify every artifact bound by the receipt, including command.json and waves.json.
$receiptCommand = $receipt.artifacts.command
$receiptWaves = $receipt.artifacts.waves
if ($null -eq $receiptCommand -or $null -eq $receiptWaves -or
        [System.IO.Path]::GetFullPath((Join-Path $armDirectory ([string] $receiptCommand.path))) -ine $commandPath -or
        [System.IO.Path]::GetFullPath((Join-Path $armDirectory ([string] $receiptWaves.path))) -ine $wavesResolved) {
    throw 'Receipt must canonically bind command.json and the supplied waves.json.'
}
$receiptArtifactPaths = @()
foreach ($property in $receipt.artifacts.PSObject.Properties) {
    $relative = [string] $property.Value.path
    $artifactPath = [System.IO.Path]::GetFullPath((Join-Path $armDirectory $relative))
    if (-not $artifactPath.StartsWith($armDirectory + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Receipt artifact escapes the target-only directory: $relative"
    }
    $artifactPath = Resolve-RequiredFile $artifactPath "Receipt artifact $($property.Name)"
    $receiptArtifactPaths += $artifactPath
    if ((Get-Sha256 $artifactPath) -cne [string] $property.Value.sha256) {
        throw "Receipt SHA-256 mismatch for $($property.Name)."
    }
}
if (@($receiptArtifactPaths | Where-Object { $outputPaths -icontains $_ }).Count -ne 0) {
    throw 'OutputPath or its stderr sidecar collides with receipt-bound qualification evidence.'
}

$waves = @($artifact.waves)
if ($waves.Count -ne 7) {
    throw "Expected exactly seven target-only waves, got $($waves.Count)."
}
$reference = @($waves[0].clients[0].tokens)
if ($reference.Count -ne 256) {
    throw "Expected 256 reference tokens, got $($reference.Count)."
}
$referenceCsv = ($reference -join ',')
for ($waveIndex = 0; $waveIndex -lt $waves.Count; ++$waveIndex) {
    $wave = $waves[$waveIndex]
    $clients = @($wave.clients)
    if ([int] $wave.wave -ne $waveIndex -or [int] $wave.parallel -ne 1 -or
            $clients.Count -ne 1 -or [int] $clients[0].seed -ne 20260819 -or
            [int] $clients[0].generated_tokens -ne 256 -or
            [string] $clients[0].token_sha256 -cne $TrustedTokenTraceSha256 -or
            [string] $clients[0].stop_type -cne 'limit' -or [bool] $clients[0].truncated -or
            @($clients[0].tokens).Count -ne 256 -or
            (($clients[0].tokens -join ',') -cne $referenceCsv) -or
            [double] $wave.metric_delta.prompt_tokens -ne 736 -or
            [double] $wave.metric_delta.cached_prompt_tokens -ne 0 -or
            [double] $wave.metric_delta.tokens -ne 256 -or
            [double] $wave.metric_delta.draft_tokens -ne 0 -or
            [double] $wave.metric_delta.draft_steps -ne 0) {
        throw "Wave $waveIndex does not match the exact deterministic P1 target-only contract."
    }
}

$hashes = [ordered]@{
    LLAMACPP_QWEN38_PARITY_EXECUTABLE_SHA256 = Get-Sha256 $executablePath
    LLAMACPP_QWEN38_PARITY_MODEL_SHA256 = $modelSha256
    LLAMACPP_QWEN38_PARITY_REFERENCE_SHA256 = Get-Sha256 $wavesResolved
    LLAMACPP_QWEN38_PARITY_MANIFEST_SHA256 = Get-Sha256 $manifestPath
    LLAMACPP_QWEN38_PARITY_COMMAND_SHA256 = Get-Sha256 $commandPath
    LLAMACPP_QWEN38_PARITY_RECEIPT_SHA256 = Get-Sha256 $receiptPath
}

$outputDirectory = Split-Path -Parent $outputResolved
if ($outputDirectory -and -not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

$environment = [ordered]@{
    LLAMACPP_QWEN38_PARITY_GUARD = 'ONE_CONTEXT_NO_SERVER'
    LLAMACPP_QWEN38_PARITY_REFERENCE_TOKENS = $referenceCsv
    LLAMACPP_QWEN38_PARITY_PROMPT_TOKEN = '1'
    LLAMACPP_QWEN38_PARITY_PROMPT_COUNT = '736'
    LLAMACPP_QWEN38_PARITY_PREDICTION_START = [string] $PredictionStart
    LLAMACPP_QWEN38_PARITY_PREDICTION_COUNT = [string] $PredictionCount
    LLAMACPP_QWEN38_PARITY_ROLLBACK_START = [string] $RollbackStart
    LLAMACPP_QWEN38_PARITY_RUNTIME_BUNDLE_MATCH = if ($runtimeBundleMatchesReference) { 'true' } else { 'false' }
    LLAMACPP_QWEN38_PARITY_BUNDLE_COUNT = [string] $runtimeBundle.Count
    LLAMACPP_QWEN38_PARITY_CURRENT_BUNDLE_SHA256 = $currentBundleSha256
    LLAMACPP_QWEN38_PARITY_REFERENCE_BUNDLE_SHA256 = $referenceBundleSha256
    LLAMACPP_QWEN38_PARITY_CURRENT_SERVER_SHA256 = $currentServerSha256
    LLAMACPP_QWEN38_PARITY_REFERENCE_SERVER_SHA256 = $referenceServerSha256
    LLAMACPP_QWEN38_PARITY_CURRENT_LLAMA_SHA256 = $runtimeBundle['llama.dll'].current_sha256
    LLAMACPP_QWEN38_PARITY_REFERENCE_LLAMA_SHA256 = $runtimeBundle['llama.dll'].reference_sha256
    LLAMACPP_QWEN38_PARITY_CURRENT_COMMON_SHA256 = $runtimeBundle['llama-common.dll'].current_sha256
    LLAMACPP_QWEN38_PARITY_REFERENCE_COMMON_SHA256 = $runtimeBundle['llama-common.dll'].reference_sha256
}
foreach ($entry in $hashes.GetEnumerator()) { $environment[$entry.Key] = $entry.Value }
if ($Qwen35Nvfp4FfnProjectionDiagnostic) {
    $environment.LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS = '1'
}
if ($Qwen35Bf16RecurrentProjectionDiagnostic) {
    $environment.LLAMA_QWEN35_BF16_ROW_INVARIANT_RECURRENT_PROJECTIONS = '1'
}
if ($Qwen35FullAttentionVecDiagnostic) {
    $environment.LLAMA_QWEN35_FATTN_D256_Q8_GQA6_VEC_QCOLS3_8 = '1'
}
if ($diagnosticCandidateIdentity) {
    $environment.LLAMACPP_QWEN38_PARITY_DIAGNOSTIC_CANDIDATE = $diagnosticCandidateIdentity
}
if ($LayerBoundaryBisection) {
    $environment.LLAMACPP_QWEN38_PARITY_LAYER_BOUNDARY_BISECTION = '1'
}
if ($RecurrentLayerInternalRefine) {
    $environment.LLAMACPP_QWEN38_PARITY_RECURRENT_INTERNAL_REFINE = '1'
}
if ($FullAttentionInternalRefine) {
    $environment.LLAMACPP_QWEN38_PARITY_FULL_ATTENTION_INTERNAL_REFINE = '1'
}

$arguments = @(
    '--model', $modelResolved,
    '--ctx-size', '2048', '--parallel', '1',
    '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
    '--flash-attn', 'on', '--batch-size', '2048', '--ubatch-size', '128',
    '--n-gpu-layers', 'all', '--device', 'CUDA0', '--split-mode', 'none', '--kv-unified'
    '--fit', 'off', '--cache-ram', '0', '--ctx-checkpoints', '0',
    '--no-cache-idle-slots', '--no-cache-prompt', '--no-webui', '--metrics',
    '--reasoning', 'off', '-lv', '4'
)

function Assert-LaunchRuntimeIdentity {
    if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf) -or
            -not (Test-Path -LiteralPath $guardPath -PathType Leaf) -or
            -not (Test-Path -LiteralPath $modelResolved -PathType Leaf) -or
            -not (Test-Path -LiteralPath $serverPath -PathType Leaf) -or
            (Get-Sha256 $executablePath) -cne $trustedLaunchExecutableSha256 -or
            (Get-Sha256 $guardPath) -cne $TrustedGuardSha256 -or
            (Get-Sha256 $modelResolved) -cne $TrustedModelSha256 -or
            (Get-Sha256 $serverPath) -cne $expectedServerSha256) {
        throw 'Bound diagnostic executable, guard, model, or server changed at the guarded launch boundary.'
    }
    $launchRuntimeFiles = @(Get-ChildItem -LiteralPath $executableDirectory -File | Where-Object {
        $_.Extension -ieq '.dll'
    } | Sort-Object { $_.Name.ToLowerInvariant() })
    $launchNames = @($launchRuntimeFiles | ForEach-Object { $_.Name.ToLowerInvariant() })
    if (-not (Test-StringArrayEqual $referenceNamesFolded $launchNames)) {
        throw 'Adjacent DLL set changed at the guarded launch boundary.'
    }
    foreach ($entry in $expectedRuntimeEntries) {
        $launchPath = Join-Path $executableDirectory ([string] $entry.name)
        if (-not (Test-Path -LiteralPath $launchPath -PathType Leaf) -or
                (Get-Sha256 $launchPath) -cne [string] $entry.sha256) {
            throw "Adjacent runtime $([string] $entry.name) changed at the guarded launch boundary."
        }
    }
}

$mutex = [System.Threading.Mutex]::new($false, 'Global\AiLoaderQwen38ParityDiagnostic')
$mutexHeld = $false
try {
    $mutexHeld = $mutex.WaitOne(0)
} catch [System.Threading.AbandonedMutexException] {
    $mutexHeld = $true
}
if (-not $mutexHeld) {
    $mutex.Dispose()
    throw 'Another Qwen3.8 parity diagnostic is already running.'
}

try {
$runtimeEnvironmentPrior = @{}
$prior = @{}
$guardResult = $null
$guardFailure = $null
try {
    Assert-LaunchRuntimeIdentity
    if ($Force) {
        Remove-Item -LiteralPath $outputResolved, $stderrPath, $guardReceiptPath -Force -ErrorAction SilentlyContinue
    }
    foreach ($entry in [System.Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
        $name = [string] $entry.Key
        if ($name.StartsWith('LLAMA_', [System.StringComparison]::OrdinalIgnoreCase) -or
                $name.StartsWith('LLAMACPP_QWEN38_PARITY_', [System.StringComparison]::OrdinalIgnoreCase) -or
                $name.StartsWith('GGML_', [System.StringComparison]::OrdinalIgnoreCase) -or
                $name.StartsWith('CUDA_', [System.StringComparison]::OrdinalIgnoreCase)) {
            $runtimeEnvironmentPrior[$name] = [string] $entry.Value
            [System.Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
    }
    foreach ($name in $environment.Keys) {
        $prior[$name] = [System.Environment]::GetEnvironmentVariable($name, 'Process')
        [System.Environment]::SetEnvironmentVariable($name, $environment[$name], 'Process')
    }
    try {
        $guardResult = & $guardPath `
            -Executable $executablePath `
            -TaskArguments $arguments `
            -TaskWorkingDirectory (Split-Path -Parent $executablePath) `
            -GpuIndex 0 `
            -MinFreeRamGiB $MinFreeRamGiB `
            -MinFreeVramMiB $MinFreeVramMiB `
            -MaxUsedVramMiB $MaxUsedVramMiB `
            -MaxRuntimeSeconds $MaxRuntimeSeconds `
            -StdoutPath $outputResolved `
            -StderrPath $stderrPath
    } catch {
        $guardFailure = $_.Exception
    }
} catch {
    $guardFailure = $_.Exception
} finally {
    foreach ($name in $prior.Keys) {
        if ($null -eq $prior[$name]) {
            Remove-Item -LiteralPath ("Env:" + $name) -ErrorAction SilentlyContinue
        } else {
            [System.Environment]::SetEnvironmentVariable($name, $prior[$name], 'Process')
        }
    }
    foreach ($entry in $runtimeEnvironmentPrior.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
}
try {
    Assert-LaunchRuntimeIdentity
} catch {
    if ($guardFailure) {
        $guardFailure = [InvalidOperationException]::new(
            ($guardFailure.Message + ' Post-guard runtime identity check also failed: ' + $_.Exception.Message))
    } else {
        $guardFailure = $_.Exception
    }
}

$relevantProcessNames = @(
    'llama-server', 'llama-cli', 'test-backend-ops', 'test-dflash-fusion-determinism',
    'test-fp8-e4m3', 'test-qwen38-recurrent-parity', 'test-qwen38-row-invariance',
    'test-recurrent-state-rollback'
)
$relevantProcesses = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -in $relevantProcessNames
})
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$mainRoot = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot '..\..'))
$checkedLeaseRoots = @(
    (Join-Path $mainRoot '.codex-deploy'),
    (Join-Path $workspaceRoot '.codex-deploy')
) | ForEach-Object { [System.IO.Path]::GetFullPath($_) } | Select-Object -Unique
$remainingLeases = @($checkedLeaseRoots | Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
    ForEach-Object {
        Get-ChildItem -LiteralPath $_ -Filter 'gpu-exclusive.lock' -File -Recurse -ErrorAction SilentlyContinue
    } | Sort-Object FullName -Unique)
$postflight = [ordered]@{
    relevant_processes = $relevantProcesses.Count
    port_18136_free = Test-LoopbackPortFree 18136
    recursive_gpu_leases = $remainingLeases.Count
    checked_lease_roots = $checkedLeaseRoots
    remaining_lease_paths = @($remainingLeases | ForEach-Object { $_.FullName })
}
if ($postflight.relevant_processes -ne 0 -or -not $postflight.port_18136_free -or
        $postflight.recursive_gpu_leases -ne 0) {
    if (-not $guardFailure) { $guardFailure = [InvalidOperationException]::new('Postflight proof failed after the exclusive GPU task.') }
}

function Write-DiagnosticGuardReceipt([string] $Validation, [string] $ValidationError) {
    $guardStatus = if ($guardFailure) { 'failed' } elseif ($guardResult -and
            $guardResult.Status -ceq 'completed' -and [int] $guardResult.ExitCode -eq 0) { 'completed' } else { 'invalid' }
    $guardReceipt = [ordered]@{
        schema = 'ai-loader-qwen38-parity-guard-receipt/v1'
        created_at = [DateTimeOffset]::UtcNow.ToString('o')
        guard_status = $guardStatus
        diagnostic_validation = $Validation
        error = if ($ValidationError) { $ValidationError } elseif ($guardFailure) { $guardFailure.Message } else { $null }
        executable_sha256 = $hashes.LLAMACPP_QWEN38_PARITY_EXECUTABLE_SHA256
        guard_sha256 = $TrustedGuardSha256
        server_sha256 = $currentServerSha256
        model_sha256 = $modelSha256
        reference_sha256 = $hashes.LLAMACPP_QWEN38_PARITY_REFERENCE_SHA256
        manifest_sha256 = $hashes.LLAMACPP_QWEN38_PARITY_MANIFEST_SHA256
        trace_validation_sha256 = $TrustedValidationSha256
        trace_guard_receipt_sha256 = $TrustedTraceGuardReceiptSha256
        command_sha256 = $hashes.LLAMACPP_QWEN38_PARITY_COMMAND_SHA256
        qualification_receipt_sha256 = $hashes.LLAMACPP_QWEN38_PARITY_RECEIPT_SHA256
        runtime_bundle_sha256 = $currentBundleSha256
        runtime_bundle_count = $runtimeBundle.Count
        diagnostic_candidate = $diagnosticCandidateIdentity
        qwen35_nvfp4_head_hint_compiled = $true
        qwen35_nvfp4_ffn_projections = [bool] $Qwen35Nvfp4FfnProjectionDiagnostic
        qwen35_bf16_recurrent_projections = [bool] $Qwen35Bf16RecurrentProjectionDiagnostic
        qwen35_fattn_d256_q8_gqa6_vec_qcols3_8 = [bool] $Qwen35FullAttentionVecDiagnostic
        layer_boundary_bisection = [bool] $LayerBoundaryBisection
        recurrent_internal_refine = [bool] $RecurrentLayerInternalRefine
        full_attention_internal_refine = [bool] $FullAttentionInternalRefine
        guard = $guardResult
        postflight = $postflight
        stdout_sha256 = if (Test-Path -LiteralPath $outputResolved -PathType Leaf) { Get-Sha256 $outputResolved } else { $null }
        stderr_sha256 = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) { Get-Sha256 $stderrPath } else { $null }
    }
    [System.IO.File]::WriteAllText(
        $guardReceiptPath,
        ($guardReceipt | ConvertTo-Json -Depth 10),
        [System.Text.UTF8Encoding]::new($false))
}

if ($guardFailure) {
    Write-DiagnosticGuardReceipt 'not_run' $guardFailure.Message
    throw $guardFailure
}
if (-not $guardResult -or $guardResult.Status -cne 'completed' -or [int] $guardResult.ExitCode -ne 0) {
    Write-DiagnosticGuardReceipt 'not_run' 'Exclusive GPU guard returned an invalid receipt.'
    throw 'Exclusive GPU guard did not return a successful completion receipt.'
}
if ([int] $guardResult.GpuIndex -ne 0 -or
        [string] $guardResult.GpuUuid -cne [string] $manifest.components.gpu.uuid) {
    Write-DiagnosticGuardReceipt 'not_run' 'Diagnostic guard GPU identity does not match the reviewed P1 trace GPU.'
    throw 'Diagnostic guard GPU identity does not match the reviewed P1 trace GPU.'
}
Write-DiagnosticGuardReceipt 'pending' ''

try {
$records = @()
$lineNumber = 0
foreach ($line in Get-Content -LiteralPath $outputResolved) {
    ++$lineNumber
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    try { $records += ($line | ConvertFrom-Json) }
    catch { throw "Diagnostic stdout line $lineNumber is not valid JSON." }
}
$complete = @($records | Where-Object { $_.type -ceq 'complete' })
if ($complete.Count -ne 1 -or -not [bool] $complete[0].operational_success -or
        [bool] $complete[0].exact_parity_gate_bypassed) {
    throw 'Diagnostic did not emit one successful fail-closed completion record.'
}
$config = @($records | Where-Object { $_.type -ceq 'config' })
$expectedDiagnosticCandidate = $diagnosticCandidateIdentity
$recurrentMapSource = [string] $config[0].recurrent_layer_map_source
$validRecurrentMapSource = if ($LayerBoundaryBisection) {
    $recurrentMapSource -ceq 'qwen35_runtime_default_interval_4' -or
        $recurrentMapSource.EndsWith('.full_attention_interval', [System.StringComparison]::Ordinal) -or
        $recurrentMapSource.EndsWith('.attention.recurrent_layers', [System.StringComparison]::Ordinal)
} else { $recurrentMapSource -ceq '' }
if ($config.Count -ne 1 -or -not [bool] $config[0].exclusive_gpu_guard -or
        -not [bool] $config[0].repeat_stability_required -or
        [string] $config[0].reference_sampler -cne 'production_cpu_chain' -or
        [double] $config[0].temperature -ne 0 -or [int64] $config[0].seed -ne 20260819 -or
        -not [bool] $config[0].ignore_eos -or [bool] $config[0].backend_sampling -or
        [int] $config[0].eog_biases -ne 5 -or
        -not [bool] $config[0].server_warmup_replayed -or
        -not [bool] $config[0].server_threadpools_attached -or
        -not [bool] $config[0].full_reference_trajectory_required -or
        [int] $config[0].prediction_start -ne $PredictionStart -or
        [int] $config[0].prediction_count -ne $PredictionCount -or
        [int] $config[0].rollback_start -ne $RollbackStart -or
        [int] $config[0].prompt_tokens_accepted_per_pass -ne 736 -or
        [int] $config[0].generated_tokens_accepted_per_pass -ne 256 -or
        [string] $config[0].diagnostic_candidate -cne $expectedDiagnosticCandidate -or
        [bool] $config[0].qwen35_nvfp4_ffn_projections -ne [bool] $Qwen35Nvfp4FfnProjectionDiagnostic -or
        [bool] $config[0].qwen35_bf16_recurrent_projections -ne [bool] $Qwen35Bf16RecurrentProjectionDiagnostic -or
        [bool] $config[0].qwen35_fattn_d256_q8_gqa6_vec_qcols3_8 -ne [bool] $Qwen35FullAttentionVecDiagnostic -or
        -not [bool] $config[0].qwen35_nvfp4_head_hint_compiled -or
        [bool] $config[0].layer_boundary_bisection -ne [bool] $LayerBoundaryBisection -or
        [int] $config[0].layer_boundary_width -ne $(if ($LayerBoundaryBisection) { 8 } else { 0 }) -or
        [bool] $config[0].recurrent_internal_refine -ne [bool] $RecurrentLayerInternalRefine -or
        [int] $config[0].recurrent_internal_layer -ne $(if ($RecurrentLayerInternalRefine) { 0 } else { -1 }) -or
        [bool] $config[0].full_attention_internal_refine -ne [bool] $FullAttentionInternalRefine -or
        [int] $config[0].full_attention_internal_layer -ne $(if ($FullAttentionInternalRefine) { 3 } else { -1 }) -or
        [bool] $config[0].layer_type_map_attested -ne [bool] $LayerBoundaryBisection -or
        [string] $config[0].recurrent_layer_map_hash -cne $(if ($LayerBoundaryBisection) { 'a01852462c981ca5' } else { '0000000000000000' }) -or
        -not $validRecurrentMapSource -or
        [int] $config[0].recurrent_layers -ne $(if ($LayerBoundaryBisection) { 48 } else { 0 }) -or
        [int] $config[0].full_attention_layers -ne $(if ($LayerBoundaryBisection) { 16 } else { 0 })) {
    throw 'Diagnostic config does not attest the exclusive guard and repeat-stability contract.'
}
if ([bool] $config[0].reference_runtime_bundle_match -ne $runtimeBundleMatchesReference -or
        [int] $config[0].runtime_bundle_count -ne $runtimeBundle.Count -or
        [string] $config[0].current_bundle_sha256 -cne $currentBundleSha256 -or
        [string] $config[0].reference_bundle_sha256 -cne $referenceBundleSha256 -or
        [string] $config[0].current_server_sha256 -cne $currentServerSha256 -or
        [string] $config[0].reference_server_sha256 -cne $referenceServerSha256 -or
        [string] $config[0].current_llama_sha256 -cne $runtimeBundle['llama.dll'].current_sha256 -or
        [string] $config[0].reference_llama_sha256 -cne $runtimeBundle['llama.dll'].reference_sha256 -or
        [string] $config[0].current_common_sha256 -cne $runtimeBundle['llama-common.dll'].current_sha256 -or
        [string] $config[0].reference_common_sha256 -cne $runtimeBundle['llama-common.dll'].reference_sha256) {
    throw 'Diagnostic runtime-bundle audit fields do not match the wrapper preflight.'
}
$referenceValidation = @($records | Where-Object { $_.type -ceq 'reference_validation' })
if ($referenceValidation.Count -ne 1 -or
        [int] $referenceValidation[0].first_prediction -ne 0 -or
        [int] $referenceValidation[0].last_prediction -ne 255 -or
        [int] $referenceValidation[0].predictions_checked -ne 256 -or
        [int] $referenceValidation[0].prompt_tokens_accepted_per_pass -ne 736 -or
        [int] $referenceValidation[0].generated_tokens_accepted_per_pass -ne 256 -or
        -not [bool] $referenceValidation[0].repeat_stable) {
    throw 'Diagnostic did not validate the complete 256-token server-equivalent trace twice.'
}
if ($LayerBoundaryBisection) {
    $hex64 = '^[0-9a-f]{16}$'
    $blocks = @($records | Where-Object { $_.type -ceq 'layer_block' } | Sort-Object { [int] $_.layer })
    if ($blocks.Count -ne 64) { throw 'Layer bisection did not emit exactly 64 coarse block records.' }
    for ($layer = 0; $layer -lt 64; ++$layer) {
        $record = $blocks[$layer]
        $expectedLayerType = if ((($layer + 1) % 4) -eq 0) { 'full_attention' } else { 'recurrent' }
        if ([int] $record.layer -ne $layer -or [string] $record.layer_type -cne $expectedLayerType -or
                [string] $record.boundary -cne 'block_output' -or [int] $record.rows -ne 8 -or
                [int64] $record.row_bytes -le 0 -or [int] $record.mismatch_count -lt 0 -or
                [int] $record.mismatch_count -gt 8 -or -not [bool] $record.finite -or
                -not [bool] $record.exact_row_comparison -or -not [bool] $record.repeat_stable -or
                [string] $record.scalar_window_hash -cnotmatch $hex64 -or
                [string] $record.batch_window_hash -cnotmatch $hex64) {
            throw "Invalid coarse layer-boundary record for layer $layer."
        }
        $first = [int] $record.first_mismatch_prediction
        if (($record.mismatch_count -eq 0 -and $first -ne -1) -or
                ($record.mismatch_count -gt 0 -and ($first -lt $PredictionStart -or $first -ge ($PredictionStart + 8)))) {
            throw "Invalid coarse first-mismatch evidence for layer $layer."
        }
        if (([int] $record.mismatch_count -eq 0) -ne
                ([string] $record.scalar_window_hash -ceq [string] $record.batch_window_hash)) {
            throw "Coarse exact comparison and compact hashes disagree for layer $layer."
        }
    }

    $summary = @($records | Where-Object { $_.type -ceq 'layer_boundary_summary' })
    if ($summary.Count -ne 1 -or [int] $summary[0].window_start -ne $PredictionStart -or
            [int] $summary[0].window_width -ne 8 -or [int] $summary[0].layers -ne 64 -or
            [int] $summary[0].recurrent_layers -ne 48 -or [int] $summary[0].full_attention_layers -ne 16 -or
            [string] $summary[0].recurrent_layer_map_hash -cne 'a01852462c981ca5' -or
            [string] $summary[0].recurrent_layer_map_source -cne $recurrentMapSource -or
            -not [bool] $summary[0].layer_type_map_attested -or
            -not [bool] $summary[0].one_context_at_a_time -or -not [bool] $summary[0].content_free -or
            -not [bool] $summary[0].observer_top2_and_state_digest_invariant) {
        throw 'Layer-boundary summary does not match the pinned 64-layer Qwen3.8 contract.'
    }
    $firstBadLayer = [int] $summary[0].first_bad_layer
    $refineExpected = $firstBadLayer -ge 0
    if ($firstBadLayer -lt -1 -or $firstBadLayer -ge 64 -or
            [bool] $summary[0].refine_executed -ne $refineExpected) {
        throw 'Layer-boundary refinement decision is inconsistent.'
    }
    $computedBadBlocks = @($blocks | Where-Object { [int] $_.mismatch_count -gt 0 })
    if (($computedBadBlocks.Count -eq 0 -and ($firstBadLayer -ne -1 -or
                [int] $summary[0].first_bad_prediction -ne -1)) -or
            ($computedBadBlocks.Count -gt 0 -and
                ($firstBadLayer -ne [int] $computedBadBlocks[0].layer -or
                 [int] $summary[0].first_bad_prediction -ne [int] $computedBadBlocks[0].first_mismatch_prediction))) {
        throw 'Layer-boundary summary does not identify the first mismatching coarse block.'
    }

    $observer = @($records | Where-Object { $_.type -ceq 'observer_invariance' } | Sort-Object stage)
    $expectedObserverStages = if ($RecurrentLayerInternalRefine) {
        @('coarse', 'recurrent_internal', 'refine')
    } elseif ($FullAttentionInternalRefine) {
        @('coarse', 'full_attention_internal', 'refine')
    } elseif ($refineExpected) { @('coarse', 'refine') } else { @('coarse') }
    if ($observer.Count -ne $expectedObserverStages.Count -or
            -not (Test-StringArrayEqual @($observer | ForEach-Object { [string] $_.stage }) $expectedObserverStages)) {
        throw 'Observer-invariance stage coverage is incomplete.'
    }
    foreach ($record in $observer) {
        if (-not [bool] $record.scalar_top2_match_unobserved -or
                -not [bool] $record.batch_top2_match_unobserved -or
                -not [bool] $record.scalar_partial_state_digest_match_unobserved -or
                -not [bool] $record.batch_partial_state_digest_match_unobserved -or
                -not [bool] $record.repeat_stable) {
            throw 'Observed results do not match the callback-free control.'
        }
        if ([string] $record.stage -ceq 'full_attention_internal' -and
                ((@($record.PSObject.Properties.Name | Sort-Object) -join ',') -cne
                    (@('batch_full_state_digest_match_unobserved','batch_partial_state_digest_match_unobserved',
                       'batch_top2_match_unobserved','repeat_stable','route_snapshot_matches_unobserved',
                       'scalar_full_state_digest_match_unobserved','scalar_partial_state_digest_match_unobserved',
                       'scalar_top2_match_unobserved','stage','type') -join ',') -or
                 -not [bool] $record.scalar_full_state_digest_match_unobserved -or
                 -not [bool] $record.batch_full_state_digest_match_unobserved -or
                 -not [bool] $record.route_snapshot_matches_unobserved)) {
            throw 'Full-attention observer evidence does not match callback-free full state and routes.'
        }
    }

    $boundaries = @($records | Where-Object { $_.type -ceq 'layer_boundary' } | Sort-Object boundary)
    $stateBoundaries = @($records | Where-Object { $_.type -ceq 'recurrent_state_boundary' })
    if ($refineExpected) {
        $expectedBoundaries = @('attention_output', 'attention_residual', 'block_input', 'block_output', 'ffn_output')
        if ($boundaries.Count -ne 5 -or
                -not (Test-StringArrayEqual @($boundaries | ForEach-Object { [string] $_.boundary }) $expectedBoundaries)) {
            throw 'First-bad-layer refinement did not emit the exact five boundaries.'
        }
        $expectedLayerType = if ((($firstBadLayer + 1) % 4) -eq 0) { 'full_attention' } else { 'recurrent' }
        foreach ($record in $boundaries) {
            if ([int] $record.layer -ne $firstBadLayer -or [string] $record.layer_type -cne $expectedLayerType -or
                    [int] $record.rows -ne 8 -or [int64] $record.row_bytes -le 0 -or
                    [int] $record.mismatch_count -lt 0 -or [int] $record.mismatch_count -gt 8 -or
                    -not [bool] $record.finite -or -not [bool] $record.exact_row_comparison -or
                    -not [bool] $record.repeat_stable -or
                    [string] $record.scalar_window_hash -cnotmatch $hex64 -or
                    [string] $record.batch_window_hash -cnotmatch $hex64) {
                throw 'Invalid first-bad-layer boundary evidence.'
            }
            $first = [int] $record.first_mismatch_prediction
            if (([int] $record.mismatch_count -eq 0 -and
                    ($first -ne -1 -or [string] $record.scalar_window_hash -cne [string] $record.batch_window_hash)) -or
                    ([int] $record.mismatch_count -gt 0 -and
                    ($first -lt $PredictionStart -or $first -ge ($PredictionStart + 8) -or
                     [string] $record.scalar_window_hash -ceq [string] $record.batch_window_hash))) {
                throw 'Refined exact comparison, first mismatch, and compact hashes disagree.'
            }
        }
        $boundaryOrder = @('block_input', 'attention_output', 'attention_residual', 'ffn_output', 'block_output')
        $computedFirstBoundary = $null
        foreach ($name in $boundaryOrder) {
            $candidate = @($boundaries | Where-Object { [string] $_.boundary -ceq $name })[0]
            if ([int] $candidate.mismatch_count -gt 0) { $computedFirstBoundary = $candidate; break }
        }
        if ($null -eq $computedFirstBoundary -or
                [string] $summary[0].first_bad_boundary -cne [string] $computedFirstBoundary.boundary -or
                [int] $summary[0].first_boundary_mismatch_prediction -ne
                    [int] $computedFirstBoundary.first_mismatch_prediction) {
            throw 'Layer-boundary summary does not identify the first refined mismatch.'
        }
        if ($expectedLayerType -ceq 'recurrent') {
            if ($stateBoundaries.Count -ne 1 -or [int] $stateBoundaries[0].layer -ne $firstBadLayer -or
                    [string] $stateBoundaries[0].boundary -cne 'state_predelta' -or
                    [int] $stateBoundaries[0].scalar_steps -ne 8 -or
                    [int] $stateBoundaries[0].batch_prefix_states -ne 1 -or
                    [int64] $stateBoundaries[0].state_bytes -le 0 -or
                    -not [bool] $stateBoundaries[0].prefix_exact_match -or
                    -not [bool] $stateBoundaries[0].finite -or -not [bool] $stateBoundaries[0].repeat_stable -or
                    [string] $stateBoundaries[0].scalar_evolution_hash -cnotmatch $hex64 -or
                    [string] $stateBoundaries[0].batch_prefix_hash -cnotmatch $hex64) {
                throw 'Recurrent refinement lacks exact state-input evidence.'
            }
        } elseif ($stateBoundaries.Count -ne 0) {
            throw 'Full-attention refinement emitted an unexpected recurrent-state record.'
        }
    } elseif ($boundaries.Count -ne 0 -or $stateBoundaries.Count -ne 0 -or
            [string] $summary[0].first_bad_boundary -cne '' -or
            [int] $summary[0].first_boundary_mismatch_prediction -ne -1) {
        throw 'Refinement evidence or a refined mismatch was emitted despite identical block outputs.'
    }

    $partialState = @($records | Where-Object { $_.type -ceq 'recurrent_partial_state' })
    if ($partialState.Count -ne 1 -or [string] $partialState[0].scope -cne 'recurrent_only' -or
            [string] $partialState[0].prefix_hash -cnotmatch $hex64 -or
            [string] $partialState[0].scalar_evolution_hash -cnotmatch $hex64 -or
            [string] $partialState[0].scalar_final_hash -cnotmatch $hex64 -or
            [string] $partialState[0].batch_final_hash -cnotmatch $hex64 -or
            [int64] $partialState[0].prefix_bytes -le 0 -or [int64] $partialState[0].final_bytes -le 0 -or
            -not [bool] $partialState[0].prefix_match -or -not [bool] $partialState[0].repeat_stable) {
        throw 'Recurrent partial-state evidence is incomplete.'
    }
    $internalBoundaries = @($records | Where-Object { $_.type -ceq 'recurrent_internal_boundary' })
    $internalCaches = @($records | Where-Object { $_.type -ceq 'recurrent_internal_cache' })
    $internalSummary = @($records | Where-Object { $_.type -ceq 'recurrent_internal_summary' })
    if ($RecurrentLayerInternalRefine) {
        if ($firstBadLayer -ne 0 -or [string] $summary[0].first_bad_boundary -cne 'attention_output' -or
                [int] $summary[0].first_boundary_mismatch_prediction -ne $PredictionStart) {
            throw 'Recurrent-internal mode requires the reviewed layer-0 attention-output mismatch.'
        }
        $internalOrder = @(
            'attn_norm', 'linear_attn_qkv_mixed', 'z', 'beta', 'beta_sigmoid',
            'alpha', 'gate', 'conv_output_silu',
            'q_conv_predelta', 'k_conv_predelta', 'v_conv_predelta', 'gdn_core_output',
            'final_output', 'linear_attn_out'
        )
        $internalNames = @($internalBoundaries | ForEach-Object { [string] $_.boundary } | Sort-Object)
        $expectedInternalNames = @($internalOrder | Sort-Object)
        if ($internalBoundaries.Count -ne $internalOrder.Count -or
                -not (Test-StringArrayEqual $internalNames $expectedInternalNames)) {
            throw 'Recurrent-internal refinement did not emit the exact 14-boundary set.'
        }
        $featureRank = @{
            attn_norm = 1; linear_attn_qkv_mixed = 1; z = 1; beta = 2; beta_sigmoid = 2
            alpha = 1; gate = 1; conv_output_silu = 1
            q_conv_predelta = 2; k_conv_predelta = 2; v_conv_predelta = 2
            gdn_core_output = 2; final_output = 1; linear_attn_out = 1
        }
        $expectedFeatures = @{
            attn_norm = @(5120); linear_attn_qkv_mixed = @(10240); z = @(6144)
            beta = @(1,48); beta_sigmoid = @(1,48); alpha = @(48); gate = @(48)
            conv_output_silu = @(10240)
            q_conv_predelta = @(128,16); k_conv_predelta = @(128,16); v_conv_predelta = @(128,48)
            gdn_core_output = @(128,48); final_output = @(6144); linear_attn_out = @(5120)
        }
        foreach ($record in $internalBoundaries) {
            $name = [string] $record.boundary
            $rank = [int] $featureRank[$name]
            $expectedIndex = [Array]::IndexOf([object[]] $internalOrder, $name)
            $scalarShape = @($record.scalar_shape | ForEach-Object { [int64] $_ })
            $batchShape = @($record.batch_shape | ForEach-Object { [int64] $_ })
            if ([int] $record.layer -ne 0 -or [int] $record.source_sequence_index -ne $expectedIndex -or
                    [int] $record.rows -ne 8 -or
                    $scalarShape.Count -ne 4 -or $batchShape.Count -ne 4 -or
                    @($scalarShape | Where-Object { $_ -le 0 }).Count -ne 0 -or
                    @($batchShape | Where-Object { $_ -le 0 }).Count -ne 0 -or
                    $scalarShape[$rank] -ne 1 -or $batchShape[$rank] -ne 8 -or
                    -not [bool] $record.finite -or -not [bool] $record.exact_token_slice_comparison -or
                    -not [bool] $record.repeat_stable -or [int] $record.mismatch_count -lt 0 -or
                    [int] $record.mismatch_count -gt 8 -or
                    [string] $record.scalar_window_hash -cnotmatch $hex64 -or
                    [string] $record.batch_window_hash -cnotmatch $hex64) {
                throw "Invalid recurrent-internal evidence for $name."
            }
            $featureCount = [int64] 1
            for ($axis = 0; $axis -lt $rank; ++$axis) {
                if ($scalarShape[$axis] -ne $batchShape[$axis] -or
                        $scalarShape[$axis] -ne [int64] $expectedFeatures[$name][$axis]) {
                    throw "Recurrent-internal feature shape differs across widths for $name."
                }
                $featureCount *= $scalarShape[$axis]
            }
            for ($axis = $rank + 1; $axis -lt 4; ++$axis) {
                if ($scalarShape[$axis] -ne 1 -or $batchShape[$axis] -ne 1) {
                    throw "Recurrent-internal sequence axes are not singleton for $name."
                }
            }
            if ([int64] $record.row_bytes -ne ($featureCount * 4)) {
                throw "Recurrent-internal row byte count is inconsistent for $name."
            }
            $first = [int] $record.first_mismatch_prediction
            if (([int] $record.mismatch_count -eq 0 -and
                    ($first -ne -1 -or [string] $record.scalar_window_hash -cne [string] $record.batch_window_hash)) -or
                    ([int] $record.mismatch_count -gt 0 -and
                    ($first -lt $PredictionStart -or $first -ge ($PredictionStart + 8) -or
                     [string] $record.scalar_window_hash -ceq [string] $record.batch_window_hash))) {
                throw "Recurrent-internal exact comparison is inconsistent for $name."
            }
        }
        $internalCacheNames = @($internalCaches | ForEach-Object { [string] $_.boundary } | Sort-Object)
        if ($internalCaches.Count -ne 2 -or
                -not (Test-StringArrayEqual $internalCacheNames @('conv_states', 'state_predelta'))) {
            throw 'Recurrent-internal cache evidence is incomplete.'
        }
        foreach ($record in $internalCaches) {
            $scalarShape = @($record.scalar_shape | ForEach-Object { [int64] $_ })
            $batchShape = @($record.batch_shape | ForEach-Object { [int64] $_ })
            $expectedCacheShape = if ([string] $record.boundary -ceq 'conv_states') {
                @('30720','1','1','1')
            } else { @('128','128','48','1') }
            $expectedCacheBytes = if ([string] $record.boundary -ceq 'conv_states') {
                [int64] 122880
            } else { [int64] 3145728 }
            $scalarCacheShape = @($scalarShape | ForEach-Object { [string] $_ })
            $batchCacheShape = @($batchShape | ForEach-Object { [string] $_ })
            if ([int] $record.layer -ne 0 -or
                    [string] $record.scope -cne 'cache_input_prefix_and_scalar_evolution' -or
                    [int] $record.scalar_steps -ne 8 -or
                    [int] $record.batch_prefix_states -ne 1 -or
                    [int64] $record.state_bytes -ne $expectedCacheBytes -or
                    -not [bool] $record.prefix_exact_match -or -not [bool] $record.finite -or
                    -not [bool] $record.repeat_stable -or
                    [string] $record.scalar_evolution_hash -cnotmatch $hex64 -or
                    [string] $record.batch_prefix_hash -cnotmatch $hex64 -or
                    $scalarShape.Count -ne 4 -or $batchShape.Count -ne 4 -or
                    -not (Test-StringArrayEqual $scalarCacheShape $batchCacheShape) -or
                    -not (Test-StringArrayEqual $scalarCacheShape $expectedCacheShape)) {
                throw 'Invalid recurrent-internal cache prefix/evolution evidence.'
            }
        }
        $computedInternalFirst = $null
        foreach ($name in $internalOrder) {
            $candidate = @($internalBoundaries | Where-Object { [string] $_.boundary -ceq $name })[0]
            if ([int] $candidate.mismatch_count -gt 0) { $computedInternalFirst = $candidate; break }
        }
        if ($internalSummary.Count -ne 1 -or $null -eq $computedInternalFirst -or
                [int] $internalSummary[0].layer -ne 0 -or [int] $internalSummary[0].rows -ne 8 -or
                [int] $internalSummary[0].boundaries -ne 14 -or
                [int] $internalSummary[0].cache_boundaries -ne 2 -or
                [string] $internalSummary[0].first_bad_observed_boundary -cne [string] $computedInternalFirst.boundary -or
                [int] $internalSummary[0].first_bad_prediction -ne [int] $computedInternalFirst.first_mismatch_prediction -or
                -not [bool] $internalSummary[0].boundary_order_semantic_not_causal -or
                [int64] $internalSummary[0].scalar_gdn -ne 0 -or
                [int64] $internalSummary[0].scalar_gdn_fused_cache -ne 768 -or
                [int64] $internalSummary[0].batch_gdn -ne 0 -or
                [int64] $internalSummary[0].batch_gdn_fused_cache -ne 96 -or
                -not [bool] $internalSummary[0].one_context_at_a_time -or
                -not [bool] $internalSummary[0].content_free -or
                -not [bool] $internalSummary[0].observer_top2_and_state_digest_invariant) {
            throw 'Recurrent-internal summary does not identify the first internal mismatch.'
        }
    } elseif ($internalBoundaries.Count -ne 0 -or $internalCaches.Count -ne 0 -or $internalSummary.Count -ne 0) {
        throw 'Recurrent-internal evidence was emitted without the opt-in mode.'
    }
    $fullInternalBoundaries = @($records | Where-Object { $_.type -ceq 'full_attention_internal_boundary' })
    $fullInternalSummary = @($records | Where-Object { $_.type -ceq 'full_attention_internal_summary' })
    if ($FullAttentionInternalRefine) {
        if ($firstBadLayer -ne 3 -or [string] $summary[0].first_bad_boundary -cne 'attention_output' -or
                [int] $summary[0].first_boundary_mismatch_prediction -ne $PredictionStart) {
            throw 'Full-attention internal mode requires the reviewed layer-3 attention-output mismatch.'
        }
        $fullInternalOrder = @(
            'attn_norm', 'qg_projection', 'q_pre_norm', 'q_post_norm',
            'k_projection', 'k_post_norm', 'v_projection', 'gate_pre_sigmoid',
            'q_post_norm_rope', 'k_post_norm_rope', 'v_reshaped',
            'kv_fa_output_pregate', 'attn_gated', 'output_projection'
        )
        $fullInternalNames = @($fullInternalBoundaries | ForEach-Object { [string] $_.boundary } | Sort-Object)
        if ($fullInternalBoundaries.Count -ne $fullInternalOrder.Count -or
                -not (Test-StringArrayEqual $fullInternalNames @($fullInternalOrder | Sort-Object))) {
            throw 'Full-attention internal refinement did not emit the exact 14-boundary set.'
        }
        $fullFeatureRank = @{
            attn_norm = 1; qg_projection = 1; q_pre_norm = 2; q_post_norm = 2
            k_projection = 1; k_post_norm = 2; v_projection = 1; gate_pre_sigmoid = 1
            q_post_norm_rope = 2; k_post_norm_rope = 2; v_reshaped = 2
            kv_fa_output_pregate = 1; attn_gated = 1; output_projection = 1
        }
        $fullExpectedFeatures = @{
            attn_norm = @(5120); qg_projection = @(12288); q_pre_norm = @(256,24); q_post_norm = @(256,24)
            k_projection = @(1024); k_post_norm = @(256,4); v_projection = @(1024); gate_pre_sigmoid = @(6144)
            q_post_norm_rope = @(256,24); k_post_norm_rope = @(256,4); v_reshaped = @(256,4)
            kv_fa_output_pregate = @(6144); attn_gated = @(6144); output_projection = @(5120)
        }
        $fullBoundaryFields = @(
            'batch_shape','batch_window_hash','boundary','exact_token_slice_comparison','finite',
            'first_mismatch_prediction','layer','mismatch_count','repeat_stable','row_bytes','rows',
            'scalar_shape','scalar_window_hash','source_sequence_index','type'
        )
        foreach ($record in $fullInternalBoundaries) {
            $name = [string] $record.boundary
            $rank = [int] $fullFeatureRank[$name]
            $expectedIndex = [Array]::IndexOf([object[]] $fullInternalOrder, $name)
            $scalarShape = @($record.scalar_shape | ForEach-Object { [int64] $_ })
            $batchShape = @($record.batch_shape | ForEach-Object { [int64] $_ })
            if ((@($record.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($fullBoundaryFields -join ',') -or
                    [int] $record.layer -ne 3 -or [int] $record.source_sequence_index -ne $expectedIndex -or
                    [int] $record.rows -ne 8 -or $scalarShape.Count -ne 4 -or $batchShape.Count -ne 4 -or
                    @($scalarShape | Where-Object { $_ -le 0 }).Count -ne 0 -or
                    @($batchShape | Where-Object { $_ -le 0 }).Count -ne 0 -or
                    $scalarShape[$rank] -ne 1 -or $batchShape[$rank] -ne 8 -or
                    -not [bool] $record.finite -or -not [bool] $record.exact_token_slice_comparison -or
                    -not [bool] $record.repeat_stable -or [int] $record.mismatch_count -lt 0 -or
                    [int] $record.mismatch_count -gt 8 -or
                    [string] $record.scalar_window_hash -cnotmatch $hex64 -or
                    [string] $record.batch_window_hash -cnotmatch $hex64) {
                throw "Invalid full-attention internal evidence for $name."
            }
            $featureCount = [int64] 1
            for ($axis = 0; $axis -lt $rank; ++$axis) {
                if ($scalarShape[$axis] -ne $batchShape[$axis] -or
                        $scalarShape[$axis] -ne [int64] $fullExpectedFeatures[$name][$axis]) {
                    throw "Full-attention feature shape differs across widths for $name."
                }
                $featureCount *= $scalarShape[$axis]
            }
            for ($axis = $rank + 1; $axis -lt 4; ++$axis) {
                if ($scalarShape[$axis] -ne 1 -or $batchShape[$axis] -ne 1) {
                    throw "Full-attention sequence axes are not singleton for $name."
                }
            }
            if ([int64] $record.row_bytes -ne ($featureCount * 4)) {
                throw "Full-attention row byte count is inconsistent for $name."
            }
            $first = [int] $record.first_mismatch_prediction
            if (([int] $record.mismatch_count -eq 0 -and
                    ($first -ne -1 -or [string] $record.scalar_window_hash -cne [string] $record.batch_window_hash)) -or
                    ([int] $record.mismatch_count -gt 0 -and
                    ($first -lt $PredictionStart -or $first -ge ($PredictionStart + 8) -or
                     [string] $record.scalar_window_hash -ceq [string] $record.batch_window_hash))) {
                throw "Full-attention exact comparison is inconsistent for $name."
            }
        }
        $computedFullInternalFirst = $null
        foreach ($name in $fullInternalOrder) {
            $candidate = @($fullInternalBoundaries | Where-Object { [string] $_.boundary -ceq $name })[0]
            if ([int] $candidate.mismatch_count -gt 0) { $computedFullInternalFirst = $candidate; break }
        }
        $fullSummaryFields = @(
            'batch_fattn_mma_f16','batch_fattn_tile','batch_fattn_vec','batch_rope_view_set_rows',
            'batch_set_rows','batch_sigmoid','batch_sigmoid_mul','boundaries',
            'boundary_order_semantic_not_causal','content_free','first_bad_observed_boundary',
            'first_bad_prediction','full_state_digest_matches_unobserved','layer','one_context_at_a_time',
            'route_snapshot_matches_unobserved','rows','scalar_fattn_mma_f16','scalar_fattn_tile',
            'scalar_fattn_vec','scalar_rope_view_set_rows','scalar_set_rows','scalar_sigmoid',
            'scalar_sigmoid_mul','type'
        )
        if ($fullInternalSummary.Count -ne 1 -or
                (@($fullInternalSummary[0].PSObject.Properties.Name | Sort-Object) -join ',') -cne ($fullSummaryFields -join ',') -or
                $null -eq $computedFullInternalFirst -or
                [int] $fullInternalSummary[0].layer -ne 3 -or [int] $fullInternalSummary[0].rows -ne 8 -or
                [int] $fullInternalSummary[0].boundaries -ne 14 -or
                [string] $fullInternalSummary[0].first_bad_observed_boundary -cne [string] $computedFullInternalFirst.boundary -or
                [int] $fullInternalSummary[0].first_bad_prediction -ne [int] $computedFullInternalFirst.first_mismatch_prediction -or
                -not [bool] $fullInternalSummary[0].boundary_order_semantic_not_causal -or
                [int64] $fullInternalSummary[0].scalar_fattn_vec -ne 128 -or
                [int64] $fullInternalSummary[0].scalar_fattn_mma_f16 -ne 0 -or
                [int64] $fullInternalSummary[0].scalar_fattn_tile -ne 0 -or
                [int64] $fullInternalSummary[0].batch_fattn_vec -ne 0 -or
                [int64] $fullInternalSummary[0].batch_fattn_mma_f16 -ne 16 -or
                [int64] $fullInternalSummary[0].batch_fattn_tile -ne 0 -or
                [int64] $fullInternalSummary[0].scalar_sigmoid -ne 0 -or
                [int64] $fullInternalSummary[0].scalar_sigmoid_mul -ne 128 -or
                [int64] $fullInternalSummary[0].batch_sigmoid -ne 0 -or
                [int64] $fullInternalSummary[0].batch_sigmoid_mul -ne 16 -or
                [int64] $fullInternalSummary[0].scalar_set_rows -ne 256 -or
                [int64] $fullInternalSummary[0].batch_set_rows -ne 32 -or
                [int64] $fullInternalSummary[0].scalar_rope_view_set_rows -ne 0 -or
                [int64] $fullInternalSummary[0].batch_rope_view_set_rows -ne 0 -or
                -not [bool] $fullInternalSummary[0].route_snapshot_matches_unobserved -or
                -not [bool] $fullInternalSummary[0].full_state_digest_matches_unobserved -or
                -not [bool] $fullInternalSummary[0].one_context_at_a_time -or
                -not [bool] $fullInternalSummary[0].content_free) {
            throw 'Full-attention summary does not identify the first internal mismatch and exact routes.'
        }
    } elseif ($fullInternalBoundaries.Count -ne 0 -or $fullInternalSummary.Count -ne 0) {
        throw 'Full-attention internal evidence was emitted without the opt-in mode.'
    }
    $knownRecordCount = 1 + 1 + 64 + $observer.Count + $boundaries.Count + $stateBoundaries.Count +
        $internalBoundaries.Count + $internalCaches.Count + $internalSummary.Count +
        $fullInternalBoundaries.Count + $fullInternalSummary.Count + 1 + 1 + 1
} else {
$expectedWidths = if ($Qwen35FullAttentionVecDiagnostic) { @(1, 2, 3, 4, 5, 6, 7, 8) } else { @(1, 2, 4, 8) }
$widthRows = @($records | Where-Object { $_.type -ceq 'width' })
foreach ($width in $expectedWidths) {
    $rows = @($widthRows | Where-Object { [int] $_.width -eq $width } | Sort-Object { [int] $_.prediction_index })
    if ($rows.Count -ne $PredictionCount) {
        throw "Diagnostic width $width did not emit exactly $PredictionCount rows."
    }
    for ($index = 0; $index -lt $PredictionCount; ++$index) {
        if ([int] $rows[$index].prediction_index -ne ($PredictionStart + $index)) {
            throw "Diagnostic width $width prediction coverage is incomplete."
        }
    }
}
$widthStates = @($records | Where-Object { $_.type -ceq 'width_state' })
$widthStateIds = @($widthStates | ForEach-Object { [string] $_.width } | Sort-Object { [int] $_ })
$expectedWidthIds = @($expectedWidths | ForEach-Object { [string] $_ })
if ($widthStates.Count -ne $expectedWidths.Count -or
        @($widthStates | Where-Object { -not [bool] $_.repeat_stable }).Count -ne 0 -or
        -not (Test-StringArrayEqual $widthStateIds $expectedWidthIds)) {
    throw 'Diagnostic did not emit the exact stable width-state set.'
}
if ($Qwen35FullAttentionVecDiagnostic -and
        @($widthStates | Where-Object { [string] $_.state_scope -cne 'full' }).Count -ne 0) {
    throw 'Full-attention candidate width states must attest full sequence-state scope.'
}
if ($Qwen35FullAttentionVecDiagnostic) {
    $widthStateFields = @('consumed_token_index','prefix_state_bytes','prefix_state_hash','repeat_stable',
        'state_bytes','state_hash','state_scope','type','width')
    $scalarWidthState = @($widthStates | Where-Object { [int] $_.width -eq 1 })[0]
    foreach ($state in $widthStates) {
        if ((@($state.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($widthStateFields -join ',') -or
                [int] $state.consumed_token_index -ne ($PredictionStart + $PredictionCount - 2) -or
                [string] $state.prefix_state_hash -cnotmatch '^[0-9a-f]{16}$' -or
                [string] $state.state_hash -cnotmatch '^[0-9a-f]{16}$' -or
                [int64] $state.prefix_state_bytes -le 0 -or [int64] $state.state_bytes -le 0 -or
                [string] $state.prefix_state_hash -cne [string] $scalarWidthState.prefix_state_hash -or
                [int64] $state.prefix_state_bytes -ne [int64] $scalarWidthState.prefix_state_bytes -or
                [string] $state.state_hash -cne [string] $scalarWidthState.state_hash -or
                [int64] $state.state_bytes -ne [int64] $scalarWidthState.state_bytes) {
            throw 'Full-attention candidate emitted an invalid or non-matching full width-state record.'
        }
    }
}
$fattnRoutes = @($records | Where-Object { $_.type -ceq 'fattn_candidate_route' } | Sort-Object { [int] $_.width })
$fattnHintRoutes = @($records | Where-Object { $_.type -ceq 'fattn_graph_hint_route' } | Sort-Object { [int] $_.width })
$fattnPredicates = @($records | Where-Object { $_.type -ceq 'fattn_candidate_predicate' } | Sort-Object { [int] $_.width })
if ($Qwen35FullAttentionVecDiagnostic) {
    $fattnPredicateFields = @('compiled_arch','content_free','device_cc','dst_contiguous','dst_ne','dst_nb',
        'dst_type','evaluated','fail_mask','hint','k_ne','k_nb','k_type','mask_contiguous','mask_ne','mask_nb',
        'mask_type','max_bias_bits','op','prec','q_ne','q_nb','q_type','scale_bits','softcap_bits','src4_null',
        'type','v_ne','v_nb','v_type','width') | Sort-Object
    $fattnPredicateIds = @($fattnPredicates | ForEach-Object { [string] $_.width })
    if ($fattnPredicates.Count -ne 8 -or
            -not (Test-StringArrayEqual $fattnPredicateIds @('1','2','3','4','5','6','7','8'))) {
        throw 'Full-attention candidate did not emit exact width 1..8 CUDA predicate observations.'
    }
    foreach ($predicate in $fattnPredicates) {
        $width = [int] $predicate.width
        $candidateWidth = $width -ge 3 -and $width -le 8
        $expectedFailMask = if ($candidateWidth) { '0000000000000000' } else { '0000000000000044' }
        $nkv = if (@($predicate.k_ne).Count -eq 4) { [int64] $predicate.k_ne[1] } else { 0 }
        if ((@($predicate.PSObject.Properties.Name | Sort-Object) -join ',') -cne
                ($fattnPredicateFields -join ',') -or
                -not [bool] $predicate.evaluated -or -not [bool] $predicate.content_free -or
                [string] $predicate.fail_mask -cnotmatch '^[0-9a-f]{16}$' -or
                [string] $predicate.fail_mask -cne $expectedFailMask -or
                [int] $predicate.device_cc -ne 1200 -or [int] $predicate.compiled_arch -lt 1200 -or
                -not [bool] $predicate.src4_null -or
                [int] $predicate.hint -ne $(if ($candidateWidth) { 8 } else { 0 }) -or
                [int] $predicate.op -ne 42 -or [int] $predicate.q_type -ne 0 -or
                [int] $predicate.k_type -ne 8 -or [int] $predicate.v_type -ne 8 -or
                [int] $predicate.mask_type -ne 1 -or [int] $predicate.dst_type -ne 0 -or
                [int] $predicate.prec -ne 10 -or
                [uint32] $predicate.scale_bits -ne 1031798784 -or
                [uint32] $predicate.max_bias_bits -ne 0 -or [uint32] $predicate.softcap_bits -ne 0 -or
                -not [bool] $predicate.mask_contiguous -or -not [bool] $predicate.dst_contiguous -or
                @($predicate.q_ne).Count -ne 4 -or @($predicate.q_nb).Count -ne 4 -or
                @($predicate.k_ne).Count -ne 4 -or @($predicate.k_nb).Count -ne 4 -or
                @($predicate.v_ne).Count -ne 4 -or @($predicate.v_nb).Count -ne 4 -or
                @($predicate.mask_ne).Count -ne 4 -or @($predicate.mask_nb).Count -ne 4 -or
                @($predicate.dst_ne).Count -ne 4 -or @($predicate.dst_nb).Count -ne 4 -or
                (@($predicate.q_ne | ForEach-Object { [string] $_ }) -join ',') -cne "256,$width,24,1" -or
                $nkv -le 0 -or $nkv % 256 -ne 0 -or
                (@($predicate.k_ne | ForEach-Object { [string] $_ }) -join ',') -cne "256,$nkv,4,1" -or
                (@($predicate.v_ne | ForEach-Object { [string] $_ }) -join ',') -cne "256,$nkv,4,1" -or
                (@($predicate.mask_ne | ForEach-Object { [string] $_ }) -join ',') -cne "$nkv,$width,1,1" -or
                (@($predicate.dst_ne | ForEach-Object { [string] $_ }) -join ',') -cne "256,24,$width,1" -or
                [uint64] $predicate.q_nb[0] -ne 4 -or [uint64] $predicate.q_nb[1] -ne 24576 -or
                [uint64] $predicate.q_nb[2] -ne 1024 -or [uint64] $predicate.q_nb[3] -ne (24576 * $width) -or
                [uint64] $predicate.k_nb[0] -ne 34 -or [uint64] $predicate.k_nb[1] -ne 1088 -or
                [uint64] $predicate.k_nb[2] -ne 272 -or [uint64] $predicate.k_nb[3] -lt (1088 * $nkv) -or
                [uint64] $predicate.k_nb[3] % 1088 -ne 0 -or
                [uint64] $predicate.v_nb[0] -ne 34 -or [uint64] $predicate.v_nb[1] -ne 1088 -or
                [uint64] $predicate.v_nb[2] -ne 272 -or [uint64] $predicate.v_nb[3] -lt (1088 * $nkv) -or
                [uint64] $predicate.v_nb[3] % 1088 -ne 0 -or
                [uint64] $predicate.mask_nb[0] -ne 2 -or
                [uint64] $predicate.mask_nb[1] -ne (2 * $nkv) -or
                [uint64] $predicate.mask_nb[2] -ne (2 * $nkv * $width) -or
                [uint64] $predicate.mask_nb[3] -ne (2 * $nkv * $width) -or
                [uint64] $predicate.dst_nb[0] -ne 4 -or [uint64] $predicate.dst_nb[1] -ne 1024 -or
                [uint64] $predicate.dst_nb[2] -ne 24576 -or
                [uint64] $predicate.dst_nb[3] -ne (24576 * $width)) {
            throw "Invalid full-attention CUDA predicate observation for width $width."
        }
    }
    $fattnHintRouteFields = @('content_free','first_single_batch_probe','hint_expected','hint_seen','type','width')
    $fattnHintRouteIds = @($fattnHintRoutes | ForEach-Object { [string] $_.width })
    if ($fattnHintRoutes.Count -ne 8 -or
            -not (Test-StringArrayEqual $fattnHintRouteIds @('1','2','3','4','5','6','7','8'))) {
        throw 'Full-attention candidate did not emit exact width 1..8 graph-hint coverage.'
    }
    foreach ($hintRoute in $fattnHintRoutes) {
        $width = [int] $hintRoute.width
        $candidateWidth = $width -ge 3 -and $width -le 8
        $expectedHintHits = if ($candidateWidth) { 16 } else { 0 }
        if ((@($hintRoute.PSObject.Properties.Name | Sort-Object) -join ',') -cne
                ($fattnHintRouteFields -join ',') -or
                [bool] $hintRoute.hint_expected -ne $candidateWidth -or
                [int64] $hintRoute.hint_seen -ne $expectedHintHits -or
                -not [bool] $hintRoute.first_single_batch_probe -or
                -not [bool] $hintRoute.content_free) {
            throw "Invalid full-attention graph-hint proof for width $width."
        }
    }
    $fattnRouteFields = @(
        'candidate_selected','candidate_vec','content_free','fattn_mma_f16','fattn_tile','fattn_vec',
        'first_single_batch_probe','shape_guard_attested','type','width'
    )
    $fattnRouteIds = @($fattnRoutes | ForEach-Object { [string] $_.width })
    if ($fattnRoutes.Count -ne 8 -or
            -not (Test-StringArrayEqual $fattnRouteIds @('1','2','3','4','5','6','7','8'))) {
        throw 'Full-attention candidate did not emit exact width 1..8 route coverage.'
    }
    foreach ($route in $fattnRoutes) {
        $width = [int] $route.width
        $selected = [int64] $route.candidate_vec
        $candidateWidth = $width -ge 3 -and $width -le 8
        if ((@($route.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($fattnRouteFields -join ',') -or
                [bool] $route.candidate_selected -ne $candidateWidth -or
                [int64] $route.candidate_vec -lt 0 -or [int64] $route.fattn_vec -lt 0 -or
                [int64] $route.fattn_mma_f16 -lt 0 -or [int64] $route.fattn_tile -lt 0 -or
                ($candidateWidth -and ($selected -ne 16 -or [int64] $route.fattn_vec -ne 16 -or
                    [int64] $route.fattn_mma_f16 -ne 0 -or [int64] $route.fattn_tile -ne 0)) -or
                (-not $candidateWidth -and ($selected -ne 0 -or
                    [int64] $route.fattn_vec -ne 16 -or
                    [int64] $route.fattn_mma_f16 -ne 0 -or [int64] $route.fattn_tile -ne 0)) -or
                -not [bool] $route.first_single_batch_probe -or -not [bool] $route.shape_guard_attested -or
                -not [bool] $route.content_free) {
            throw "Invalid full-attention candidate route proof for width $width."
        }
    }
} elseif ($fattnRoutes.Count -ne 0 -or $fattnHintRoutes.Count -ne 0 -or $fattnPredicates.Count -ne 0) {
    throw 'Full-attention candidate route evidence was emitted without its explicit switch.'
}
$widthSummaries = @($records | Where-Object { $_.type -ceq 'width_summary' })
$expectedSummaryIds = if ($Qwen35FullAttentionVecDiagnostic) { @('2','3','4','5','6','7','8') } else { @('2','4','8') }
$widthSummaryIds = @($widthSummaries | ForEach-Object { [string] $_.width } | Sort-Object { [int] $_ })
if ($widthSummaries.Count -ne $expectedSummaryIds.Count -or
        -not (Test-StringArrayEqual $widthSummaryIds $expectedSummaryIds)) {
    throw 'Diagnostic did not emit the complete scalar width comparison summaries.'
}
if ($Qwen35FullAttentionVecDiagnostic -and @($widthSummaries | Where-Object {
            [int] $_.first_top1_difference -ne -1 -or [int] $_.first_top2_difference -ne -1 -or
            [int] $_.first_top2_value_difference -ne -1 -or
            -not [bool] $_.prefix_state_match_scalar -or -not [bool] $_.final_state_match_scalar -or
            [string] $_.state_scope -cne 'full'
        }).Count -ne 0) {
    throw 'Full-attention candidate failed exact logits or full-state width parity.'
}
if ($Qwen35FullAttentionVecDiagnostic) {
    $widthSummaryFields = @('final_state_match_scalar','first_top1_difference','first_top2_difference',
        'first_top2_value_difference','prefix_state_match_scalar','state_scope','type','width')
    if (@($widthSummaries | Where-Object {
            (@($_.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($widthSummaryFields -join ',')
        }).Count -ne 0) {
        throw 'Full-attention candidate width-summary schema is not exact.'
    }
}
$rollbackBatch = @($records | Where-Object { $_.type -ceq 'rollback_batch' })
if ($rollbackBatch.Count -ne 8 -or @($rollbackBatch | Where-Object { [int] $_.width -ne 8 }).Count -ne 0) {
    throw 'Diagnostic did not emit the complete width-8 rollback batch.'
}
$rollbackPredictionIndices = @($rollbackBatch | ForEach-Object { [int] $_.prediction_index } | Sort-Object)
for ($index = 0; $index -lt 8; ++$index) {
    if ($rollbackPredictionIndices[$index] -ne ($RollbackStart + $index)) {
        throw 'Diagnostic rollback batch prediction coverage is incomplete.'
    }
}
$rollbackBatchSummary = @($records | Where-Object { $_.type -ceq 'rollback_batch_summary' })
if ($rollbackBatchSummary.Count -ne 1 -or [int] $rollbackBatchSummary[0].width -ne 8 -or
        -not [bool] $rollbackBatchSummary[0].repeat_stable) {
    throw 'Diagnostic did not emit one stable width-8 rollback batch summary.'
}
if ($Qwen35FullAttentionVecDiagnostic -and
        (-not [bool] $rollbackBatchSummary[0].row_top2_ids_match_scalar -or
         -not [bool] $rollbackBatchSummary[0].row_top2_values_match_scalar -or
         -not [bool] $rollbackBatchSummary[0].final_state_match_scalar -or
         [string] $rollbackBatchSummary[0].state_scope -cne 'full')) {
    throw 'Full-attention candidate failed exact rollback-batch logits or full-state parity.'
}
if ($Qwen35FullAttentionVecDiagnostic) {
    $rollbackBatchSummaryFields = @('final_state_bytes','final_state_hash','final_state_match_scalar',
        'intermediate_snapshot_generation_vs_selection_separable','oracle_final_state_bytes',
        'oracle_final_state_hash','repeat_stable','row_top2_ids_match_scalar',
        'row_top2_values_match_scalar','state_scope','type','width')
    $batchSummary = $rollbackBatchSummary[0]
    if ((@($batchSummary.PSObject.Properties.Name | Sort-Object) -join ',') -cne
            ($rollbackBatchSummaryFields -join ',') -or
            [string] $batchSummary.final_state_hash -cnotmatch '^[0-9a-f]{16}$' -or
            [string] $batchSummary.oracle_final_state_hash -cnotmatch '^[0-9a-f]{16}$' -or
            [string] $batchSummary.final_state_hash -cne [string] $batchSummary.oracle_final_state_hash -or
            [int64] $batchSummary.final_state_bytes -le 0 -or
            [int64] $batchSummary.final_state_bytes -ne [int64] $batchSummary.oracle_final_state_bytes -or
            [bool] $batchSummary.intermediate_snapshot_generation_vs_selection_separable) {
        throw 'Full-attention candidate rollback-batch full-state schema is invalid.'
    }
}
$rollbacks = @($records | Where-Object { $_.type -ceq 'rollback' } | Sort-Object { [int] $_.rollback })
if ($rollbacks.Count -ne 7 -or @($rollbacks | Where-Object {
            [int] $_.width -ne 8 -or -not [bool] $_.repeat_stable
        }).Count -ne 0) {
    throw 'Diagnostic did not emit seven stable width-8 rollback records.'
}
if ($Qwen35FullAttentionVecDiagnostic -and @($rollbacks | Where-Object {
            -not [bool] $_.selected_state_match -or -not [bool] $_.continued_state_match -or
            -not [bool] $_.continued_top2_match -or -not [bool] $_.continued_top2_values_match -or
            [string] $_.state_scope -cne 'full'
        }).Count -ne 0) {
    throw 'Full-attention candidate failed exact rollback continuation or full-state parity.'
}
if ($Qwen35FullAttentionVecDiagnostic) {
    $rollbackFields = @('committed_token_index','continuation_prediction_index','continuation_token_index',
        'continued_margin','continued_state_hash','continued_state_match','continued_top1',
        'continued_top2_match','continued_top2_values_match','oracle_continued_state_bytes',
        'oracle_continued_state_hash','oracle_margin','oracle_selected_state_bytes',
        'oracle_selected_state_hash','oracle_top1','repeat_stable','rollback','selected_state_hash',
        'selected_state_match','snapshot_generation_vs_selection_separable','state_bytes','state_scope','type','width')
    foreach ($rollbackRecord in $rollbacks) {
        $depth = [int] $rollbackRecord.rollback
        $expectedCommitted = $RollbackStart + 6 - $depth
        if ((@($rollbackRecord.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($rollbackFields -join ',') -or
                [int] $rollbackRecord.committed_token_index -ne $expectedCommitted -or
                [int] $rollbackRecord.continuation_token_index -ne ($expectedCommitted + 1) -or
                [int] $rollbackRecord.continuation_prediction_index -ne ($expectedCommitted + 2) -or
                [int] $rollbackRecord.continued_top1 -ne [int] $rollbackRecord.oracle_top1 -or
                [double] $rollbackRecord.continued_margin -ne [double] $rollbackRecord.oracle_margin -or
                [string] $rollbackRecord.selected_state_hash -cnotmatch '^[0-9a-f]{16}$' -or
                [string] $rollbackRecord.continued_state_hash -cnotmatch '^[0-9a-f]{16}$' -or
                [string] $rollbackRecord.selected_state_hash -cne [string] $rollbackRecord.oracle_selected_state_hash -or
                [string] $rollbackRecord.continued_state_hash -cne [string] $rollbackRecord.oracle_continued_state_hash -or
                [int64] $rollbackRecord.state_bytes -le 0 -or
                [int64] $rollbackRecord.state_bytes -ne [int64] $rollbackRecord.oracle_selected_state_bytes -or
                [int64] $rollbackRecord.state_bytes -ne [int64] $rollbackRecord.oracle_continued_state_bytes -or
                [bool] $rollbackRecord.snapshot_generation_vs_selection_separable) {
            throw "Full-attention candidate rollback schema is invalid at depth $depth."
        }
    }
}
for ($rollback = 1; $rollback -le 7; ++$rollback) {
    if ([int] $rollbacks[$rollback - 1].rollback -ne $rollback) {
        throw 'Diagnostic rollback depth coverage is incomplete.'
    }
}
$knownRecordCount = 1 + 1 + ($expectedWidths.Count * $PredictionCount) +
    $widthStates.Count + $fattnRoutes.Count + $fattnHintRoutes.Count + $fattnPredicates.Count +
    $widthSummaries.Count + 8 + 1 + 7 + 1
}
if ($records.Count -ne $knownRecordCount) {
    throw 'Diagnostic emitted an unexpected or incomplete JSONL record set.'
}
foreach ($entry in $hashes.GetEnumerator()) {
    $field = ($entry.Key -replace '^LLAMACPP_QWEN38_PARITY_', '').ToLowerInvariant()
    if ([string] $config[0].$field -cne $entry.Value) {
        throw "Diagnostic audit hash mismatch for $field."
    }
}
    Write-DiagnosticGuardReceipt 'passed' ''
} catch {
    Write-DiagnosticGuardReceipt 'failed' $_.Exception.Message
    throw
}

Write-Host "Qwen3.8 parity diagnostic completed under exclusive guard: $outputResolved"
Write-Host "Diagnostic log: $stderrPath"
Write-Host "Guard receipt and postflight proof: $guardReceiptPath"
} finally {
    if ($mutexHeld) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
