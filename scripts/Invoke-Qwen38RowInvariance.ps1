[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Executable,
    [Parameter(Mandatory = $true)] [string] $OutputDirectory,
    [ValidateSet('head-candidate', 'projection-candidate', 'fp8', 'gdn', 'bf16-projections', 'bf16-candidate', 'rms', 'ssm-conv', 'l2', 'gated-norm', 'fattn-vec-pb1', 'quick', 'full')] [string] $Mode = 'head-candidate',
    [ValidateRange(0, 15)] [int] $GpuIndex = 0,
    [ValidateRange(1, 72)] [int] $MinFreeRamGiB = 20,
    [ValidateRange(4096, 65536)] [int] $MinFreeVramMiB = 12000,
    [ValidateRange(512, 28672)] [int] $MaxUsedVramMiB = 20000,
    [ValidateRange(1, 7200)] [int] $MaxRuntimeSeconds = 1800,
    [switch] $Force
)

$ErrorActionPreference = 'Stop'

$TrustedExecutableSha256 = '6f1789a12fe5863f5c597afd99a35109364e24023c82716d509422bd84799baf'
$TrustedGuardSha256 = '3120e9fe7c247250a440389d0450c71741718b391d435a270fd2f16d33dc1463'
$TrustedCudartName = 'cudart64_13.dll'
$TrustedCudartSha256 = '352ba4ebe61e9a3b171f357a3daf5dd15b6af4a9857673ff893bd6fd2964c075'
$TrustedAdjacentDlls = [ordered]@{
    'cublas64_13.dll'       = '91200c2ff57b8477e94254a4501030d75e1bb35a2de3a476e7ab66df897cb046'
    'cublasLt64_13.dll'     = '8f54d7b3e5173bf2659f45bad6ae789ffef8218ab6254f10c0cc0cc5e3ec874c'
    'ggml-base.dll'         = 'bfd26fdf4d649489a6b002a4c8cc3a1588b086f1b3ba224eb729f914c98af10e'
    'ggml-cpu.dll'          = 'b647b06bd8c215393cad2e4f9fb2b36e2c9f8338057cc5100ea7011a5e045541'
    'ggml-cuda.dll'         = '2283427025dd9ee3bdc5b6318a06aff3fff62f882f0d0c6198226dbe818d5b7c'
    'ggml.dll'              = 'fc367df4697a1f4bc9b996dc224402ca7fec4d3501153fe67e815e4086566edc'
    'llama-common.dll'      = '62efb05af206ca1889b663b6b5ce78cdc0b971247c5ce019afc5389d1420d68f'
    'llama-server-impl.dll' = '36e4a42787ade6dd1978f27059df3ae4f7882d53ee8e4b933e178869786fd5f8'
    'llama.dll'             = '30cb9e4f47fba2a4dc72851d91f73a3494f4d851090f098f644f5b6aeadc5330'
    'mtmd.dll'              = '12c49c4fa900254bca491156829564df4439c644b0c0f06b1ed969e41a5166d7'
}

function Get-Sha256([string] $Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-AtomicJson([string] $Path, [object] $Value) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $temporary = "$Path.tmp.$PID.$([Guid]::NewGuid().ToString('N'))"
    try {
        [System.IO.File]::WriteAllText(
            $temporary,
            ($Value | ConvertTo-Json -Depth 12),
            [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
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

function Get-Postflight([string] $WorkspaceRoot) {
    $names = @(
        'llama-server', 'llama-cli', 'test-backend-ops', 'test-dflash-fusion-determinism',
        'test-fp8-e4m3', 'test-qwen38-recurrent-parity', 'test-qwen38-row-invariance',
        'test-recurrent-state-rollback'
    )
    $processes = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in $names })
    $mainRoot = [System.IO.Path]::GetFullPath((Join-Path $WorkspaceRoot '..\..'))
    $leaseRoots = @(
        (Join-Path $mainRoot '.codex-deploy'),
        (Join-Path $WorkspaceRoot '.codex-deploy')
    ) | ForEach-Object { [System.IO.Path]::GetFullPath($_) } | Select-Object -Unique
    $leases = @($leaseRoots | Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
        ForEach-Object {
            Get-ChildItem -LiteralPath $_ -Filter 'gpu-exclusive.lock' -File -Recurse -ErrorAction SilentlyContinue
        } | Sort-Object FullName -Unique)
    return [ordered]@{
        relevant_processes = $processes.Count
        process_details = @($processes | ForEach-Object { [ordered]@{ pid = $_.Id; name = $_.ProcessName } })
        port_18136_free = Test-LoopbackPortFree 18136
        checked_lease_paths = $leaseRoots
        remaining_lease_paths = @($leases | ForEach-Object { $_.FullName })
        remaining_gpu_leases = $leases.Count
    }
}

function Resolve-Cudart([string] $ExecutableDirectory) {
    $adjacent = Join-Path $ExecutableDirectory $TrustedCudartName
    if (Test-Path -LiteralPath $adjacent -PathType Leaf) {
        return (Resolve-Path -LiteralPath $adjacent).Path
    }
    $matches = @(& where.exe $TrustedCudartName 2>$null | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
    if ($matches.Count -eq 0) {
        throw "$TrustedCudartName is neither adjacent nor resolvable through PATH."
    }
    return (Resolve-Path -LiteralPath $matches[0]).Path
}

function Assert-CurrentRuntimeIdentity(
        [string] $ExecutablePath,
        [string] $GuardPath,
        [string] $ExecutableDirectory,
        [string] $ExpectedCudartPath) {
    if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf) -or
            (Get-Sha256 $ExecutablePath) -cne $TrustedExecutableSha256) {
        throw 'Diagnostic executable identity changed before or during the guarded run.'
    }
    if (-not (Test-Path -LiteralPath $GuardPath -PathType Leaf) -or
            (Get-Sha256 $GuardPath) -cne $TrustedGuardSha256) {
        throw 'Exclusive GPU guard identity changed before or during the guarded run.'
    }
    $currentAdjacent = @(Get-ChildItem -LiteralPath $ExecutableDirectory -Filter '*.dll' -File |
        Where-Object Name -cne $TrustedCudartName | Sort-Object Name)
    if (($currentAdjacent.Name -join "`n") -cne (@($TrustedAdjacentDlls.Keys | Sort-Object) -join "`n")) {
        throw 'Adjacent runtime DLL set changed before or during the guarded run.'
    }
    foreach ($dll in $currentAdjacent) {
        if ((Get-Sha256 $dll.FullName) -cne $TrustedAdjacentDlls[$dll.Name]) {
            throw "Adjacent runtime identity changed before or during the guarded run: $($dll.Name)"
        }
    }
    $currentCudartPath = Resolve-Cudart $ExecutableDirectory
    if ((Split-Path -Leaf $currentCudartPath) -cne $TrustedCudartName -or
            (Get-Sha256 $currentCudartPath) -cne $TrustedCudartSha256 -or
            (-not [string]::IsNullOrWhiteSpace($ExpectedCudartPath) -and
                [System.IO.Path]::GetFullPath($currentCudartPath) -cne [System.IO.Path]::GetFullPath($ExpectedCudartPath))) {
        throw 'Resolved CUDA runtime path or identity changed before or during the guarded run.'
    }
    return $currentCudartPath
}

function Assert-Records([object[]] $Records, [string] $SelectedMode) {
    $expectedIds = switch ($SelectedMode) {
        'head-candidate' { @('nvfp4-qwen35-lm-head-row-invariant') }
        'projection-candidate' { @(
            'nvfp4-qwen35-ffn-gate-up-swiglu-row-invariant',
            'nvfp4-qwen35-ffn-down-row-invariant'
        ) }
        'fp8' { @('fp8-k5120-n1024', 'fp8-k5120-n6144', 'fp8-k6144-n5120', 'fp8-k5120-n10240', 'fp8-k5120-n12288') }
        'gdn' { @('gdn-k1-vs-k8', 'gdn-seq1-vs-batch8') }
        'bf16-projections' { @('qwen35-bf16-beta-k5120-n48', 'qwen35-bf16-alpha-k5120-n48') }
        'bf16-candidate' { @('qwen35-bf16-beta-k5120-n48-row-invariant', 'qwen35-bf16-alpha-k5120-n48-row-invariant') }
        'rms' { @('qwen35-rms-norm-e5120') }
        'ssm-conv' { @('qwen35-ssm-conv-d4-c10240') }
        'l2' { @('qwen35-l2-norm-q-s128-h16', 'qwen35-l2-norm-k-s128-h16') }
        'gated-norm' { @('qwen35-post-gdn-gated-norm-s128-h48') }
        'fattn-vec-pb1' { @('qwen35-fattn-d256-q8-gqa6-vec-cols2-pb1') }
        'quick' { @('fp8-k5120-n1024', 'nvfp4-k5120-n128', 'nvfp4-k17408-n128', 'nvfp4-qwen-ffn', 'gdn-k1-vs-k8', 'gdn-seq1-vs-batch8') }
        'full' { @('fp8-k5120-n1024', 'fp8-k5120-n6144', 'fp8-k6144-n5120', 'fp8-k5120-n10240', 'fp8-k5120-n12288', 'nvfp4-k5120-n128', 'nvfp4-k17408-n128', 'nvfp4-qwen-ffn', 'gdn-k1-vs-k8', 'gdn-seq1-vs-batch8') }
    }
    $configs = @($Records | Where-Object type -ceq 'config')
    $cases = @($Records | Where-Object type -ceq 'case')
    $routes = @($Records | Where-Object type -ceq 'routes')
    $complete = @($Records | Where-Object type -ceq 'complete')
    $negative = @($Records | Where-Object type -ceq 'route_negative_control')
    $widthCoverage = @($Records | Where-Object type -ceq 'width_coverage')
    $widthRoutes = @($Records | Where-Object type -ceq 'width_route')
    $bf16RouteProbes = @($Records | Where-Object type -ceq 'bf16_route_probe')
    $bf16CandidateRouteProbes = @($Records | Where-Object type -ceq 'bf16_candidate_route_probe')
    $microbenchmarks = @($Records | Where-Object type -ceq 'microbenchmark')
    $normRouteProbes = @($Records | Where-Object type -ceq 'norm_route_probe')
    $ssmEvolution = @($Records | Where-Object type -ceq 'ssm_conv_evolution')
    $fattnPb1RouteProbes = @($Records | Where-Object type -ceq 'fattn_pb1_route_probe')
    $allowed = @('config', 'case', 'routes', 'complete', 'route_negative_control', 'width_coverage', 'width_route', 'bf16_route_probe', 'bf16_candidate_route_probe', 'microbenchmark', 'norm_route_probe', 'ssm_conv_evolution', 'fattn_pb1_route_probe')
    $expectedNegative = if ($SelectedMode -ceq 'head-candidate' -or $SelectedMode -ceq 'fattn-vec-pb1') { 1 } elseif ($SelectedMode -ceq 'projection-candidate' -or $SelectedMode -ceq 'bf16-candidate') { 2 } else { 0 }
    $expectedCoverage = if ($SelectedMode -ceq 'head-candidate') { 1 } elseif ($SelectedMode -ceq 'projection-candidate' -or $SelectedMode -ceq 'bf16-candidate') { 2 } elseif ($SelectedMode -ceq 'fp8') { 5 } elseif ($SelectedMode -ceq 'bf16-projections' -or $SelectedMode -ceq 'l2') { 2 } elseif ($SelectedMode -ceq 'rms' -or $SelectedMode -ceq 'gated-norm') { 1 } else { 0 }
    $expectedWidthRoutes = if ($SelectedMode -ceq 'head-candidate') { 15 } elseif ($SelectedMode -ceq 'projection-candidate') { 30 } elseif ($SelectedMode -ceq 'fp8') { 35 } elseif ($SelectedMode -ceq 'bf16-projections' -or $SelectedMode -ceq 'bf16-candidate' -or $SelectedMode -ceq 'l2') { 14 } elseif ($SelectedMode -ceq 'rms' -or $SelectedMode -ceq 'gated-norm') { 7 } elseif ($SelectedMode -ceq 'fattn-vec-pb1') { 6 } else { 0 }
    $expectedBf16RouteProbes = if ($SelectedMode -ceq 'bf16-projections') { 14 } else { 0 }
    $expectedBf16CandidateRouteProbes = if ($SelectedMode -ceq 'bf16-candidate') { 14 } else { 0 }
    $expectedMicrobenchmarks = if ($SelectedMode -ceq 'bf16-candidate') { 2 } else { 0 }
    $expectedNormRouteProbes = if ($SelectedMode -ceq 'l2') { 14 } elseif ($SelectedMode -ceq 'rms' -or $SelectedMode -ceq 'gated-norm') { 7 } else { 0 }
    $expectedSsmEvolution = if ($SelectedMode -ceq 'ssm-conv') { 1 } else { 0 }
    $expectedFattnPb1RouteProbes = if ($SelectedMode -ceq 'fattn-vec-pb1') { 6 } else { 0 }
    $expectedRoutes = if ($SelectedMode -ceq 'fattn-vec-pb1') { 0 } else { $expectedIds.Count }
    if (@($Records | Where-Object { $_.type -cnotin $allowed }).Count -ne 0 -or
            $configs.Count -ne 1 -or $cases.Count -ne $expectedIds.Count -or
            $routes.Count -ne $expectedRoutes -or $complete.Count -ne 1 -or
            $negative.Count -ne $expectedNegative -or
            $widthCoverage.Count -ne $expectedCoverage -or
            $widthRoutes.Count -ne $expectedWidthRoutes -or
            $bf16RouteProbes.Count -ne $expectedBf16RouteProbes -or
            $bf16CandidateRouteProbes.Count -ne $expectedBf16CandidateRouteProbes -or
            $microbenchmarks.Count -ne $expectedMicrobenchmarks -or
            $normRouteProbes.Count -ne $expectedNormRouteProbes -or
            $ssmEvolution.Count -ne $expectedSsmEvolution -or
            $fattnPb1RouteProbes.Count -ne $expectedFattnPb1RouteProbes) {
        throw 'Diagnostic JSONL record cardinality or type set is invalid.'
    }
    if ($configs[0].mode -cne $SelectedMode -or [int] $configs[0].cases -ne $expectedIds.Count -or
            [int] $configs[0].replays -ne 8 -or $configs[0].order -cne 'alternating' -or
            $configs[0].fixture -cne $(if ($SelectedMode -ceq 'fattn-vec-pb1') { 'deterministic_qwen35_fattn_exact' } elseif ($SelectedMode -cin @('bf16-projections','bf16-candidate','rms','ssm-conv','l2','gated-norm')) { 'deterministic_nonzero_shape_exact' } else { 'deterministic_cancellation_sensitive' }) -or
            $configs[0].comparison -cne 'bitwise_f32') {
        throw 'Diagnostic config contract is invalid.'
    }
    $caseIds = @($cases | ForEach-Object { [string] $_.id } | Sort-Object)
    $routeIds = @($routes | ForEach-Object { [string] $_.id } | Sort-Object)
    $isolatedRouteFields = @(
        'type', 'id', 'fp8', 'fp8_batch', 'nvfp4_mmvq_m1', 'nvfp4_mmvq_m8',
        'nvfp4_fused_ffn', 'nvfp4_row_invariant_head_scaled',
        'nvfp4_row_invariant_ffn_fused', 'nvfp4_row_invariant_ffn_down_scaled',
        'gdn', 'gdn_fused_cache', 'rms_norm', 'rms_norm_mul', 'ssm_conv',
        'ssm_conv_silu', 'l2_norm', 'silu', 'silu_mul', 'bf16_mmvf', 'bf16_cublas',
        'bf16_row_invariant_beta', 'bf16_row_invariant_alpha'
    ) | Sort-Object
    $routeCounterFields = @($isolatedRouteFields | Where-Object { $_ -cnotin @('type', 'id') })
    $assertOnlyRoutes = {
        param([object] $Record, [string[]] $Selected)
        if ((@($Record.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($isolatedRouteFields -join ',')) {
            throw "Diagnostic route schema is invalid: $($Record.id)"
        }
        foreach ($field in $routeCounterFields) {
            $value = [uint64] $Record.$field
            if (($field -cin $Selected -and $value -eq 0) -or
                    ($field -cnotin $Selected -and $value -ne 0)) {
                throw "Diagnostic isolated-route proof is invalid: $($Record.id) / $field=$value"
            }
        }
    }
    if (($caseIds -join "`n") -cne (@($expectedIds | Sort-Object) -join "`n") -or
            ($routeIds -join "`n") -cne ($(if ($SelectedMode -ceq 'fattn-vec-pb1') { '' } else { @($expectedIds | Sort-Object) -join "`n" })) -or
            @($cases | Where-Object { $_.status -cne 'passed' -or [int] $_.replays -ne 8 }).Count -ne 0) {
        throw 'Diagnostic case or route identity set is invalid.'
    }
    if ($SelectedMode -ceq 'fattn-vec-pb1') {
        $id = $expectedIds[0]
        $negativeFields = @('type','id','m','tagged','pb1_candidate','fattn_vec','fattn_mma_f16','fattn_tile','content_free') | Sort-Object
        $probeFields = @('type','id','m','scalar_rows','scalar_q_cols','scalar_ncols','scalar_ntiles_x',
            'scalar_ntiles_kv','scalar_parallel_blocks','batch_q_cols','batch_ncols','batch_ntiles_x',
            'batch_ntiles_kv','batch_parallel_blocks','pb1_candidate','fattn_vec','fattn_mma_f16',
            'fattn_tile','causal_mask_attested','content_free') | Sort-Object
        $widthFields = @('type','id','m','replays','finite','comparison','scalar_rows_independent','causal_mask_attested') | Sort-Object
        $caseFields = @('type','id','status','device_scope','d','n_kv','h_q','h_kv','gqa','q_type','k_type',
            'v_type','mask_type','causal_mask_attested','scale','max_bias','softcap','prec','scalar_m',
            'min_batch_m','max_batch_m','widths','replays','ncols_scalar','ncols_batch',
            'forced_parallel_blocks','finite','comparison') | Sort-Object
        $case = $cases[0]
        $negativeRecord = $negative[0]
        $probes = @($fattnPb1RouteProbes | Sort-Object { [int] $_.m })
        $widthRecords = @($widthRoutes | Sort-Object { [int] $_.m })
        if ((@($negativeRecord.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($negativeFields -join ',') -or
                $negativeRecord.id -cne $id -or [int] $negativeRecord.m -ne 3 -or [bool] $negativeRecord.tagged -or
                [uint64] $negativeRecord.pb1_candidate -ne 0 -or [uint64] $negativeRecord.fattn_vec -ne 0 -or
                [uint64] $negativeRecord.fattn_mma_f16 -ne 1 -or [uint64] $negativeRecord.fattn_tile -ne 0 -or
                -not [bool] $negativeRecord.content_free -or
                (@($case.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($caseFields -join ',') -or
                $case.id -cne $id -or $case.status -cne 'passed' -or $case.device_scope -cne 'sm120' -or
                [int] $case.d -ne 256 -or [int] $case.n_kv -ne 1024 -or [int] $case.h_q -ne 24 -or
                [int] $case.h_kv -ne 4 -or [int] $case.gqa -ne 6 -or $case.q_type -cne 'f32' -or
                $case.k_type -cne 'q8_0' -or $case.v_type -cne 'q8_0' -or $case.mask_type -cne 'f16' -or
                -not [bool] $case.causal_mask_attested -or [double] $case.scale -ne 0.0625 -or
                [double] $case.max_bias -ne 0.0 -or [double] $case.softcap -ne 0.0 -or $case.prec -cne 'f32' -or
                [int] $case.scalar_m -ne 1 -or [int] $case.min_batch_m -ne 3 -or [int] $case.max_batch_m -ne 8 -or
                [int] $case.widths -ne 6 -or [int] $case.replays -ne 8 -or [int] $case.ncols_scalar -ne 1 -or
                [int] $case.ncols_batch -ne 2 -or [int] $case.forced_parallel_blocks -ne 1 -or
                -not [bool] $case.finite -or $case.comparison -cne 'bitwise_f32' -or
                (@($probes | ForEach-Object { [int] $_.m }) -join ',') -cne ((3..8) -join ',') -or
                (@($widthRecords | ForEach-Object { [int] $_.m }) -join ',') -cne ((3..8) -join ',')) {
            throw "FATTN PB1 identity, negative-control, or exact tuple contract is invalid: negative=$($negativeRecord | ConvertTo-Json -Compress); case=$($case | ConvertTo-Json -Compress); probe_widths=$(@($probes | ForEach-Object { [int] $_.m }) -join ','); widths=$(@($widthRecords | ForEach-Object { [int] $_.m }) -join ',')"
        }
        foreach ($record in $probes) {
            $m = [int] $record.m
            if ((@($record.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($probeFields -join ',') -or
                    $record.id -cne $id -or [int] $record.scalar_rows -ne $m -or
                    [int] $record.scalar_q_cols -ne 1 -or [int] $record.scalar_ncols -ne 1 -or
                    [int] $record.scalar_ntiles_x -ne 1 -or [int] $record.scalar_ntiles_kv -ne 4 -or
                    [int] $record.scalar_parallel_blocks -ne 1 -or [int] $record.batch_q_cols -ne $m -or
                    [int] $record.batch_ncols -ne 2 -or [int] $record.batch_ntiles_x -ne [int](($m + 1) / 2) -or
                    [int] $record.batch_ntiles_kv -ne 4 -or [int] $record.batch_parallel_blocks -ne 1 -or
                    [uint64] $record.pb1_candidate -ne [uint64]($m + 1) -or
                    [uint64] $record.fattn_vec -ne [uint64]($m + 1) -or
                    [uint64] $record.fattn_mma_f16 -ne 0 -or [uint64] $record.fattn_tile -ne 0 -or
                    -not [bool] $record.causal_mask_attested -or -not [bool] $record.content_free) {
                throw "FATTN PB1 route/launch geometry proof is invalid at M=$m."
            }
        }
        foreach ($record in $widthRecords) {
            if ((@($record.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($widthFields -join ',') -or
                    $record.id -cne $id -or [int] $record.replays -ne 8 -or -not [bool] $record.finite -or
                    $record.comparison -cne 'bitwise_f32' -or -not [bool] $record.scalar_rows_independent -or
                    -not [bool] $record.causal_mask_attested) {
                throw "FATTN PB1 bitwise/replay proof is invalid at M=$($record.m)."
            }
        }
    }
    if ($SelectedMode -ceq 'head-candidate') {
        $sortedWidthRoutes = @($widthRoutes | Sort-Object { [int] $_.m })
        $observedWidths = @($sortedWidthRoutes | ForEach-Object { [int] $_.m } | Sort-Object -Unique)
        $widthRouteChainInvalid = $false
        $previousScaledAfter = $null
        $previousM1 = $null
        foreach ($record in $sortedWidthRoutes) {
            $m1 = [uint64] $record.observed.m1
            $before = [uint64] $record.observed.tagged_scaled_before
            $after = [uint64] $record.observed.tagged_scaled_after
            $delta = [uint64] $record.observed.tagged_scaled_delta
            if ($after -le $before -or $delta -ne ($after - $before) -or
                    ($null -ne $previousScaledAfter -and $before -ne $previousScaledAfter) -or
                    ($null -ne $previousM1 -and $m1 -lt $previousM1)) {
                $widthRouteChainInvalid = $true
            }
            $previousScaledAfter = $after
            $previousM1 = $m1
        }
        if ([int] $cases[0].fixtures -ne 3 -or -not [bool] $cases[0].output_scale -or
                [uint64] $routes[0].nvfp4_mmvq_m1 -eq 0 -or
                [uint64] $routes[0].nvfp4_mmvq_m8 -ne 0 -or
                [uint64] $routes[0].nvfp4_row_invariant_head_scaled -eq 0 -or
                [uint64] $negative[0].row_invariant_head -ne 0 -or
                $widthCoverage[0].status -cne 'passed' -or [int] $widthCoverage[0].min_m -ne 2 -or
                [int] $widthCoverage[0].max_m -ne 16 -or -not [bool] $widthCoverage[0].route_checked_each_width -or
                $widthCoverage[0].comparison -cne 'bitwise_f32' -or
                ($observedWidths -join ',') -cne ((2..16) -join ',') -or
                $widthRouteChainInvalid -or
                @($widthRoutes | Where-Object {
                    $_.id -cne 'nvfp4-qwen35-lm-head-row-invariant' -or
                    [uint64] $_.observed.m1 -lt 1 -or
                    [uint64] $_.observed.tagged_scaled_delta -lt 1 -or
                    [uint64] $_.observed.generic_m8 -ne 0 -or
                    [uint64] $_.expected.m1_min -ne 1 -or
                    [uint64] $_.expected.tagged_scaled_delta_min -ne 1 -or
                    [uint64] $_.expected.generic_m8 -ne 0
                }).Count -ne 0) {
            throw 'Tagged LM-head route, negative control, fixture, or fused-scale proof is invalid.'
        }
    }
    if ($SelectedMode -ceq 'projection-candidate') {
        foreach ($id in $expectedIds) {
            $case = @($cases | Where-Object id -ceq $id)
            $route = @($routes | Where-Object id -ceq $id)
            $negativeRecord = @($negative | Where-Object id -ceq $id)
            $coverage = @($widthCoverage | Where-Object id -ceq $id)
            $perWidth = @($widthRoutes | Where-Object id -ceq $id | Sort-Object { [int] $_.m })
            $routeField = if ($id -like '*gate-up*') {
                'nvfp4_row_invariant_ffn_fused'
            } else {
                'nvfp4_row_invariant_ffn_down_scaled'
            }
            $projectionRouteChainInvalid = $false
            $previousSelectedAfter = $null
            foreach ($record in $perWidth) {
                if ($null -ne $previousSelectedAfter -and
                        [uint64] $record.selected_before -ne $previousSelectedAfter) {
                    $projectionRouteChainInvalid = $true
                }
                $previousSelectedAfter = [uint64] $record.selected_after
            }
            if ($case.Count -ne 1 -or [int] $case[0].fixtures -ne 3 -or -not [bool] $case[0].output_scale -or
                    $route.Count -ne 1 -or [uint64] $route[0].$routeField -eq 0 -or
                    [uint64] $route[0].nvfp4_mmvq_m8 -ne 0 -or
                    $negativeRecord.Count -ne 1 -or [uint64] $negativeRecord[0].selected_route -ne 0 -or
                    $coverage.Count -ne 1 -or $coverage[0].status -cne 'passed' -or
                    [int] $coverage[0].min_m -ne 2 -or [int] $coverage[0].max_m -ne 16 -or
                    -not [bool] $coverage[0].route_checked_each_width -or
                    $coverage[0].comparison -cne 'bitwise_f32' -or $perWidth.Count -ne 15 -or
                    $projectionRouteChainInvalid -or
                    (@($perWidth | ForEach-Object { [int] $_.m }) -join ',') -cne ((2..16) -join ',') -or
                    @($perWidth | Where-Object {
                        [uint64] $_.scalar_route -eq 0 -or
                        [uint64] $_.selected_after -le [uint64] $_.selected_before -or
                        [uint64] $_.selected_delta -ne ([uint64] $_.selected_after - [uint64] $_.selected_before) -or
                        [uint64] $_.generic_m8 -ne 0
                    }).Count -ne 0) {
                throw "Qwen3.5 projection route or width proof is invalid: $id"
            }
        }
    }
    if ($SelectedMode -ceq 'fp8') {
        foreach ($id in $expectedIds) {
            $case = @($cases | Where-Object id -ceq $id)
            $route = @($routes | Where-Object id -ceq $id)
            if ($route.Count -eq 1) { & $assertOnlyRoutes $route[0] @('fp8', 'fp8_batch') }
            $coverage = @($widthCoverage | Where-Object id -ceq $id)
            $perWidth = @($widthRoutes | Where-Object id -ceq $id | Sort-Object { [int] $_.m })
            $chainInvalid = $false
            $previousAfter = $null
            foreach ($record in $perWidth) {
                $before = [uint64] $record.fp8_batch_before
                $after = [uint64] $record.fp8_batch_after
                $delta = [uint64] $record.fp8_batch_delta
                if ($after -le $before -or $delta -ne ($after - $before) -or
                        ($null -ne $previousAfter -and $before -ne $previousAfter)) {
                    $chainInvalid = $true
                }
                $previousAfter = $after
            }
            if ($case.Count -ne 1 -or [int] $case[0].replays -ne 8 -or
                    [int] $case[0].min_batch_m -ne 2 -or [int] $case[0].max_batch_m -ne 8 -or
                    [int] $case[0].widths -ne 7 -or -not [bool] $case[0].finite -or
                    $case[0].comparison -cne 'bitwise_f32' -or
                    $route.Count -ne 1 -or
                    ((@($route[0].PSObject.Properties.Name | Sort-Object) -join ',') -cne ($isolatedRouteFields -join ',')) -or
                    [uint64] $route[0].fp8_batch -eq 0 -or
                    [uint64] $route[0].fp8 -le [uint64] $route[0].fp8_batch -or
                    [uint64] $route[0].nvfp4_mmvq_m1 -ne 0 -or
                    [uint64] $route[0].nvfp4_mmvq_m8 -ne 0 -or
                    [uint64] $route[0].nvfp4_fused_ffn -ne 0 -or
                    [uint64] $route[0].nvfp4_row_invariant_head_scaled -ne 0 -or
                    [uint64] $route[0].nvfp4_row_invariant_ffn_fused -ne 0 -or
                    [uint64] $route[0].nvfp4_row_invariant_ffn_down_scaled -ne 0 -or
                    [uint64] $route[0].gdn -ne 0 -or
                    [uint64] $route[0].gdn_fused_cache -ne 0 -or
                    $coverage.Count -ne 1 -or $coverage[0].status -cne 'passed' -or
                    [int] $coverage[0].min_m -ne 2 -or [int] $coverage[0].max_m -ne 8 -or
                    [int] $coverage[0].widths -ne 7 -or [int] $coverage[0].replays -ne 8 -or
                    -not [bool] $coverage[0].finite -or $coverage[0].comparison -cne 'bitwise_f32' -or
                    -not [bool] $coverage[0].route_checked_each_width -or $perWidth.Count -ne 7 -or
                    $chainInvalid -or
                    (@($perWidth | ForEach-Object { [int] $_.m }) -join ',') -cne ((2..8) -join ',') -or
                    @($perWidth | Where-Object {
                        [int] $_.replays -ne 8 -or -not [bool] $_.finite -or
                        $_.comparison -cne 'bitwise_f32'
                    }).Count -ne 0) {
                throw "FP8 width/route/finite/bitwise proof is invalid: $id"
            }
        }
    }
    if ($SelectedMode -ceq 'gdn') {
        foreach ($id in $expectedIds) {
            $case = @($cases | Where-Object id -ceq $id)
            $route = @($routes | Where-Object id -ceq $id)
            if ($route.Count -eq 1) { & $assertOnlyRoutes $route[0] @('gdn_fused_cache') }
            $semanticValid = if ($id -ceq 'gdn-k1-vs-k8') {
                [int] $case[0].slot -eq 0
            } else {
                [string] $case[0].snapshot_order -ceq 'newest_first'
            }
            if ($case.Count -ne 1 -or [int] $case[0].replays -ne 8 -or
                    -not [bool] $case[0].finite -or $case[0].comparison -cne 'bitwise_f32' -or
                    -not $semanticValid -or $route.Count -ne 1 -or
                    ((@($route[0].PSObject.Properties.Name | Sort-Object) -join ',') -cne ($isolatedRouteFields -join ',')) -or
                    [uint64] $route[0].gdn_fused_cache -eq 0 -or
                    [uint64] $route[0].fp8 -ne 0 -or
                    [uint64] $route[0].fp8_batch -ne 0 -or
                    [uint64] $route[0].nvfp4_mmvq_m1 -ne 0 -or
                    [uint64] $route[0].nvfp4_mmvq_m8 -ne 0 -or
                    [uint64] $route[0].nvfp4_fused_ffn -ne 0 -or
                    [uint64] $route[0].nvfp4_row_invariant_head_scaled -ne 0 -or
                    [uint64] $route[0].nvfp4_row_invariant_ffn_fused -ne 0 -or
                    [uint64] $route[0].nvfp4_row_invariant_ffn_down_scaled -ne 0 -or
                    [uint64] $route[0].gdn -ne 0) {
                throw "GDN route/finite/bitwise/semantic proof is invalid: $id"
            }
        }
    }
    if ($SelectedMode -ceq 'bf16-candidate') {
        $negativeFields = @('type','id','untagged_m8_cublas','tagged_noncontiguous_m8_cublas','candidate_beta','candidate_alpha') | Sort-Object
        $probeFields = @('type','id','m','scalar_mmvf_delta','scalar_candidate_delta','scalar_cublas_delta',
            'batch_mmvf_delta','batch_candidate_delta','batch_cublas_delta','input_bf16_roundtrip_sensitive') | Sort-Object
        $widthFields = @('type','id','m','replays','finite','comparison','raw_projection_bitwise','transformed_path') | Sort-Object
        $coverageFields = @('type','id','status','min_m','max_m','widths','replays','finite','comparison','route_checked_each_width') | Sort-Object
        $caseFields = @('type','id','status','semantic','transformed_path','replays','min_batch_m','max_batch_m','widths','finite','comparison','raw_projection_bitwise') | Sort-Object
        $microFields = @('type','id','m','warmups','samples_per_arm','order','baseline','candidate',
            'baseline_median_us','baseline_p95_us','candidate_median_us','candidate_p95_us','median_ratio',
            'baseline_samples_us','candidate_samples_us') | Sort-Object
        foreach ($id in $expectedIds) {
            $isAlpha = $id -like '*alpha*'
            $semantic = if ($isAlpha) { 'alpha' } else { 'beta' }
            $transformedPath = if ($isAlpha) { 'add_softplus_mul' } else { 'sigmoid' }
            $candidateRoute = if ($isAlpha) { 'bf16_row_invariant_alpha' } else { 'bf16_row_invariant_beta' }
            $case = @($cases | Where-Object id -ceq $id)
            $route = @($routes | Where-Object id -ceq $id)
            $negativeRecord = @($negative | Where-Object id -ceq $id)
            $coverage = @($widthCoverage | Where-Object id -ceq $id)
            $perWidth = @($widthRoutes | Where-Object id -ceq $id | Sort-Object { [int] $_.m })
            $probes = @($bf16CandidateRouteProbes | Where-Object id -ceq $id | Sort-Object { [int] $_.m })
            $micro = @($microbenchmarks | Where-Object id -ceq $id)
            if ($route.Count -eq 1) { & $assertOnlyRoutes $route[0] @('bf16_mmvf','bf16_cublas',$candidateRoute) }
            if ($case.Count -ne 1 -or $route.Count -ne 1 -or $negativeRecord.Count -ne 1 -or
                    $coverage.Count -ne 1 -or $perWidth.Count -ne 7 -or $probes.Count -ne 7 -or $micro.Count -ne 1 -or
                    ((@($case[0].PSObject.Properties.Name | Sort-Object) -join ',') -cne ($caseFields -join ',')) -or
                    ((@($coverage[0].PSObject.Properties.Name | Sort-Object) -join ',') -cne ($coverageFields -join ',')) -or
                    ((@($negativeRecord[0].PSObject.Properties.Name | Sort-Object) -join ',') -cne ($negativeFields -join ',')) -or
                    ((@($micro[0].PSObject.Properties.Name | Sort-Object) -join ',') -cne ($microFields -join ',')) -or
                    $case[0].status -cne 'passed' -or $case[0].semantic -cne $semantic -or
                    $case[0].transformed_path -cne $transformedPath -or [int] $case[0].replays -ne 8 -or
                    [int] $case[0].min_batch_m -ne 2 -or [int] $case[0].max_batch_m -ne 8 -or
                    [int] $case[0].widths -ne 7 -or -not [bool] $case[0].finite -or
                    $case[0].comparison -cne 'bitwise_f32' -or -not [bool] $case[0].raw_projection_bitwise -or
                    $coverage[0].status -cne 'passed' -or
                    [int] $coverage[0].min_m -ne 2 -or [int] $coverage[0].max_m -ne 8 -or
                    [int] $coverage[0].widths -ne 7 -or [int] $coverage[0].replays -ne 8 -or
                    -not [bool] $coverage[0].finite -or $coverage[0].comparison -cne 'bitwise_f32' -or
                    -not [bool] $coverage[0].route_checked_each_width -or
                    [uint64] $negativeRecord[0].untagged_m8_cublas -eq 0 -or
                    [uint64] $negativeRecord[0].tagged_noncontiguous_m8_cublas -eq 0 -or
                    [uint64] $negativeRecord[0].candidate_beta -ne 0 -or
                    [uint64] $negativeRecord[0].candidate_alpha -ne 0 -or
                    (@($perWidth | ForEach-Object { [int] $_.m }) -join ',') -cne ((2..8) -join ',') -or
                    (@($probes | ForEach-Object { [int] $_.m }) -join ',') -cne ((2..8) -join ',')) {
                throw "BF16 recurrent projection candidate identity/cardinality proof is invalid: $id"
            }
            foreach ($record in $perWidth) {
                if ((@($record.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($widthFields -join ',') -or
                        [int] $record.replays -ne 8 -or -not [bool] $record.finite -or
                        $record.comparison -cne 'bitwise_f32' -or -not [bool] $record.raw_projection_bitwise -or
                        $record.transformed_path -cne $transformedPath) {
                    throw "BF16 recurrent projection candidate width proof is invalid: $id"
                }
            }
            foreach ($record in $probes) {
                if ((@($record.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($probeFields -join ',') -or
                        [uint64] $record.scalar_mmvf_delta -eq 0 -or [uint64] $record.scalar_candidate_delta -ne 0 -or
                        [uint64] $record.scalar_cublas_delta -ne 0 -or [uint64] $record.batch_mmvf_delta -ne 0 -or
                        [uint64] $record.batch_candidate_delta -eq 0 -or [uint64] $record.batch_cublas_delta -ne 0 -or
                        -not [bool] $record.input_bf16_roundtrip_sensitive) {
                    throw "BF16 recurrent projection candidate route probe is invalid: $id"
                }
            }
            $baselineSamples = @($micro[0].baseline_samples_us)
            $candidateSamples = @($micro[0].candidate_samples_us)
            $numbers = @([double] $micro[0].baseline_median_us, [double] $micro[0].baseline_p95_us,
                [double] $micro[0].candidate_median_us, [double] $micro[0].candidate_p95_us,
                [double] $micro[0].median_ratio) + @($baselineSamples | ForEach-Object { [double] $_ }) +
                @($candidateSamples | ForEach-Object { [double] $_ })
            $baselineSorted = @($baselineSamples | ForEach-Object { [double] $_ } | Sort-Object)
            $candidateSorted = @($candidateSamples | ForEach-Object { [double] $_ } | Sort-Object)
            $baselineMedian = [double] $micro[0].baseline_median_us
            $candidateMedian = [double] $micro[0].candidate_median_us
            if ([int] $micro[0].m -ne 8 -or [int] $micro[0].warmups -ne 4 -or
                    [int] $micro[0].samples_per_arm -ne 16 -or $micro[0].order -cne 'AB_BA' -or
                    $micro[0].baseline -cne 'untagged_cublas' -or
                    $micro[0].candidate -cne 'tagged_row_invariant_mmvf' -or
                    $baselineSamples.Count -ne 16 -or $candidateSamples.Count -ne 16 -or
                    @($numbers | Where-Object { $_ -le 0 -or [double]::IsNaN($_) -or [double]::IsInfinity($_) }).Count -ne 0 -or
                    [Math]::Abs($baselineMedian - $baselineSorted[7]) -gt 0.002 -or
                    [Math]::Abs([double]$micro[0].baseline_p95_us - $baselineSorted[15]) -gt 0.002 -or
                    [Math]::Abs($candidateMedian - $candidateSorted[7]) -gt 0.002 -or
                    [Math]::Abs([double]$micro[0].candidate_p95_us - $candidateSorted[15]) -gt 0.002 -or
                    [Math]::Abs([double]$micro[0].median_ratio - ($candidateMedian / $baselineMedian)) -gt 0.002) {
                throw "BF16 recurrent projection candidate microbenchmark evidence is invalid: $id"
            }
        }
    }
    if ($SelectedMode -cin @('bf16-projections', 'rms', 'l2', 'gated-norm')) {
        foreach ($id in $expectedIds) {
            $case = @($cases | Where-Object id -ceq $id)
            $route = @($routes | Where-Object id -ceq $id)
            $coverage = @($widthCoverage | Where-Object id -ceq $id)
            $perWidth = @($widthRoutes | Where-Object id -ceq $id | Sort-Object { [int] $_.m })
            if ($case.Count -ne 1 -or $route.Count -ne 1 -or $coverage.Count -ne 1 -or
                    [int] $case[0].replays -ne 8 -or [int] $case[0].min_batch_m -ne 2 -or
                    [int] $case[0].max_batch_m -ne 8 -or [int] $case[0].widths -ne 7 -or
                    -not [bool] $case[0].finite -or $case[0].comparison -cne 'bitwise_f32' -or
                    $coverage[0].status -cne 'passed' -or [int] $coverage[0].min_m -ne 2 -or
                    [int] $coverage[0].max_m -ne 8 -or [int] $coverage[0].widths -ne 7 -or
                    [int] $coverage[0].replays -ne 8 -or -not [bool] $coverage[0].finite -or
                    $coverage[0].comparison -cne 'bitwise_f32' -or
                    -not [bool] $coverage[0].route_checked_each_width -or $perWidth.Count -ne 7 -or
                    (@($perWidth | ForEach-Object { [int] $_.m }) -join ',') -cne ((2..8) -join ',')) {
                throw "Selective operator width/cardinality proof is invalid: $id"
            }
            $selectedRoutes = switch ($SelectedMode) {
                'bf16-projections' { @('bf16_mmvf', 'bf16_cublas') }
                'rms' { @('rms_norm_mul') }
                'l2' { @('l2_norm') }
                'gated-norm' { @('rms_norm_mul', 'silu_mul') }
            }
            & $assertOnlyRoutes $route[0] $selectedRoutes

            $previousAfter = $null
            $normWidthFields = @('type','id','m','selected_before','selected_after','selected_delta',
                'scalar_selected_delta','batch_selected_delta','scalar_secondary_delta','batch_secondary_delta',
                'scalar_fallback_delta','batch_fallback_delta','finite','comparison','replays') | Sort-Object
            foreach ($record in $perWidth) {
                if ([int] $record.replays -ne 8 -or -not [bool] $record.finite -or
                        $record.comparison -cne 'bitwise_f32') {
                    throw "Selective operator per-width finite/bitwise proof is invalid: $id"
                }
                if ($SelectedMode -ceq 'bf16-projections') {
                    $before = [uint64] $record.bf16_cublas_before
                    $after = [uint64] $record.bf16_cublas_after
                    $delta = [uint64] $record.bf16_cublas_delta
                    if ([uint64] $record.bf16_mmvf -eq 0) {
                        throw "BF16 scalar MMVF proof is missing: $id"
                    }
                } else {
                    if ((@($record.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($normWidthFields -join ',')) {
                        throw "Selective operator width-route schema is invalid: $id"
                    }
                    $before = [uint64] $record.selected_before
                    $after = [uint64] $record.selected_after
                    $delta = [uint64] $record.selected_delta
                    if ([uint64] $record.scalar_selected_delta -eq 0 -or
                            [uint64] $record.batch_selected_delta -eq 0 -or
                            [uint64] $record.scalar_fallback_delta -ne 0 -or
                            [uint64] $record.batch_fallback_delta -ne 0 -or
                            ($SelectedMode -ceq 'gated-norm' -and
                                ([uint64] $record.scalar_secondary_delta -eq 0 -or
                                 [uint64] $record.batch_secondary_delta -eq 0)) -or
                            ($SelectedMode -cne 'gated-norm' -and
                                ([uint64] $record.scalar_secondary_delta -ne 0 -or
                                 [uint64] $record.batch_secondary_delta -ne 0))) {
                        throw "Selective operator scalar/batch route separation is invalid: $id"
                    }
                }
                if ($after -le $before -or $delta -ne ($after - $before) -or
                        ($null -ne $previousAfter -and $before -ne $previousAfter)) {
                    throw "Selective operator cumulative route proof is invalid: $id"
                }
                $previousAfter = $after
            }
            if ($SelectedMode -cne 'bf16-projections') {
                $probes = @($normRouteProbes | Where-Object id -ceq $id | Sort-Object { [int] $_.m })
                $probeFields = @('type','id','m','scalar_selected_delta','batch_selected_delta',
                    'scalar_secondary_delta','batch_secondary_delta','scalar_fallback_delta','batch_fallback_delta') | Sort-Object
                if ($probes.Count -ne 7 -or
                        (@($probes | ForEach-Object { [int] $_.m }) -join ',') -cne ((2..8) -join ',') -or
                        @($probes | Where-Object {
                            ((@($_.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($probeFields -join ',')) -or
                            [uint64] $_.scalar_selected_delta -eq 0 -or [uint64] $_.batch_selected_delta -eq 0 -or
                            [uint64] $_.scalar_fallback_delta -ne 0 -or [uint64] $_.batch_fallback_delta -ne 0 -or
                            ($SelectedMode -ceq 'gated-norm' -and
                                ([uint64] $_.scalar_secondary_delta -eq 0 -or [uint64] $_.batch_secondary_delta -eq 0)) -or
                            ($SelectedMode -cne 'gated-norm' -and
                                ([uint64] $_.scalar_secondary_delta -ne 0 -or [uint64] $_.batch_secondary_delta -ne 0))
                        }).Count -ne 0) {
                    throw "Selective operator scalar/batch route probe is invalid: $id"
                }
            }
            if ($SelectedMode -ceq 'bf16-projections') {
                $probes = @($bf16RouteProbes | Where-Object id -ceq $id | Sort-Object { [int] $_.m })
                $probeFields = @('type','id','m','scalar_mmvf_delta','scalar_cublas_delta','batch_mmvf_delta','batch_cublas_delta','input_bf16_roundtrip_sensitive') | Sort-Object
                if ($probes.Count -ne 7 -or
                        (@($probes | ForEach-Object { [int] $_.m }) -join ',') -cne ((2..8) -join ',') -or
                        @($probes | Where-Object {
                            ((@($_.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($probeFields -join ',')) -or
                            [uint64] $_.scalar_mmvf_delta -eq 0 -or [uint64] $_.scalar_cublas_delta -ne 0 -or
                            [uint64] $_.batch_mmvf_delta -ne 0 -or [uint64] $_.batch_cublas_delta -eq 0 -or
                            -not [bool] $_.input_bf16_roundtrip_sensitive
                        }).Count -ne 0) {
                    throw "BF16 MMVF-versus-cuBLAS route probe is invalid: $id"
                }
            }
        }
    }
    if ($SelectedMode -ceq 'ssm-conv') {
        & $assertOnlyRoutes $routes[0] @('ssm_conv_silu')
        $state = $ssmEvolution[0]
        $stateFields = @('type','id','d_conv','channels','initial_cache_rows','tokens','scalar_steps','batch_rows','output_exact','evolved_cache_exact','cache_mapping','finite','replays','output_fnv1a64','cache_fnv1a64') | Sort-Object
        if ((@($state.PSObject.Properties.Name | Sort-Object) -join ',') -cne ($stateFields -join ',') -or
                $state.id -cne @($expectedIds)[0] -or [int] $state.d_conv -ne 4 -or
                [int] $state.channels -ne 10240 -or [int] $state.initial_cache_rows -ne 3 -or
                [int] $state.tokens -ne 8 -or [int] $state.scalar_steps -ne 8 -or
                [int] $state.batch_rows -ne 8 -or -not [bool] $state.output_exact -or
                -not [bool] $state.evolved_cache_exact -or $state.cache_mapping -cne 'eight_slots_newest_first' -or
                -not [bool] $state.finite -or [int] $state.replays -ne 8 -or
                [string] $state.output_fnv1a64 -cnotmatch '^[0-9a-f]{16}$' -or
                [string] $state.cache_fnv1a64 -cnotmatch '^[0-9a-f]{16}$' -or
                [int] $cases[0].replays -ne 8 -or -not [bool] $cases[0].finite -or
                $cases[0].comparison -cne 'bitwise_f32') {
            throw "SSM_CONV output/cache evolution proof is invalid: state=$($state | ConvertTo-Json -Compress) case=$($cases[0] | ConvertTo-Json -Compress)"
        }
    }
    if ($complete[0].status -cne 'passed' -or $complete[0].mode -cne $SelectedMode -or
            [int] $complete[0].cases -ne $expectedIds.Count) {
        throw 'Diagnostic completion record is invalid.'
    }
}

$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$stdoutPath = Join-Path $outputRoot "$Mode.jsonl"
$stderrPath = Join-Path $outputRoot "$Mode.stderr.log"
$receiptPath = Join-Path $outputRoot "$Mode.guard-receipt.json"
$receiptTargetPath = $receiptPath
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$guardPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'Invoke-ExclusiveGpuTask.ps1')).Path
$wrapperSha256 = Get-Sha256 $PSCommandPath

$mutex = [System.Threading.Mutex]::new($false, 'Global\AiLoaderQwen38RowInvariance')
$mutexHeld = $false
$guardStarted = $false
$guardResult = $null
$guardFailure = $null
$failure = $null
$validation = 'not_run'
$stage = 'setup'
$executablePath = $null
$executableSha256 = $null
$guardSha256 = $null
$runtime = [ordered]@{}
$cudartPath = $null
$scrubPrefixes = @('LLAMA_', 'GGML_', 'CUDA_')
$savedEnvironment = [ordered]@{}
$effectiveEnvironment = [ordered]@{
    scrubbed_prefixes = $scrubPrefixes
    cleared_names = @()
    wrapper_overrides = [ordered]@{ LLAMACPP_QWEN38_ROW_INVARIANCE_GUARD = 'EXCLUSIVE_GPU' }
    guarded_child_overrides = [ordered]@{
        CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
        CUDA_VISIBLE_DEVICES = 'guard-selected GPU UUID'
        AI_LOADER_EXCLUSIVE_GPU_GUARD = '1'
        AI_LOADER_EXCLUSIVE_GPU_UUID = 'guard-selected GPU UUID'
        AI_LOADER_EXCLUSIVE_GPU_LEASE_PATH = 'guard-owned lease path'
        AI_LOADER_EXCLUSIVE_GPU_LEASE_NONCE = 'guard-generated nonce'
        AI_LOADER_EXCLUSIVE_GPU_OWNER_PID = 'guard process PID'
        AI_LOADER_EXCLUSIVE_GPU_JOB_NAME = 'guard-created job name'
    }
}
$priorPath = [Environment]::GetEnvironmentVariable('PATH', 'Process')
$priorSentinel = [Environment]::GetEnvironmentVariable('LLAMACPP_QWEN38_ROW_INVARIANCE_GUARD', 'Process')
$environmentChanged = $false
$postflight = $null

try {
    try {
        $mutexHeld = $mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $mutexHeld = $true
    }
    if (-not $mutexHeld) { throw 'Another Qwen3.8 row-invariance diagnostic is already running.' }

    $stage = 'preflight'
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Diagnostic executable does not exist: $Executable" }
    $executablePath = (Resolve-Path -LiteralPath $Executable).Path
    $executableDirectory = Split-Path -Parent $executablePath
    $executableSha256 = Get-Sha256 $executablePath
    $guardSha256 = Get-Sha256 $guardPath
    if ($executableSha256 -cne $TrustedExecutableSha256) { throw 'Diagnostic executable hash is not the reviewed build.' }
    if ($guardSha256 -cne $TrustedGuardSha256) { throw 'Exclusive GPU guard hash is not trusted.' }

    $adjacent = @(Get-ChildItem -LiteralPath $executableDirectory -Filter '*.dll' -File |
        Where-Object Name -cne $TrustedCudartName | Sort-Object Name)
    if (($adjacent.Name -join "`n") -cne (@($TrustedAdjacentDlls.Keys | Sort-Object) -join "`n")) {
        throw 'Adjacent runtime DLL set does not exactly match the reviewed bundle.'
    }
    foreach ($dll in $adjacent) {
        $hash = Get-Sha256 $dll.FullName
        if ($hash -cne $TrustedAdjacentDlls[$dll.Name]) { throw "Adjacent runtime hash mismatch: $($dll.Name)" }
        $runtime[$dll.Name] = [ordered]@{ path = $dll.FullName; sha256 = $hash }
    }
    $cudartPath = Resolve-Cudart $executableDirectory
    if ((Split-Path -Leaf $cudartPath) -cne $TrustedCudartName -or (Get-Sha256 $cudartPath) -cne $TrustedCudartSha256) {
        throw 'Resolved CUDA runtime does not match the reviewed cudart identity.'
    }
    $runtime[$TrustedCudartName] = [ordered]@{ path = $cudartPath; sha256 = Get-Sha256 $cudartPath }

    if (-not $Force -and ((Test-Path -LiteralPath $stdoutPath -PathType Leaf) -or
            (Test-Path -LiteralPath $stderrPath -PathType Leaf) -or
            (Test-Path -LiteralPath $receiptPath -PathType Leaf))) {
        throw "Output already exists. Use -Force to replace it: $outputRoot"
    }
    if ($Force) { Remove-Item -LiteralPath $stdoutPath, $stderrPath, $receiptPath -Force -ErrorAction SilentlyContinue }

    $stage = 'guard'
    $environmentChanged = $true
    $namesToScrub = @([Environment]::GetEnvironmentVariables('Process').Keys |
        ForEach-Object { [string] $_ } |
        Where-Object {
            $name = $_
            @($scrubPrefixes | Where-Object { $name.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0
        } | Sort-Object -Unique)
    foreach ($name in $namesToScrub) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    }
    $effectiveEnvironment.cleared_names = $namesToScrub
    $env:PATH = "$executableDirectory;$(Split-Path -Parent $cudartPath);$priorPath"
    $env:LLAMACPP_QWEN38_ROW_INVARIANCE_GUARD = 'EXCLUSIVE_GPU'
    $stage = 'runtime_pre_guard_recheck'
    $null = Assert-CurrentRuntimeIdentity $executablePath $guardPath $executableDirectory $cudartPath
    $stage = 'guard'
    $guardStarted = $true
    try {
        $guardResult = & $guardPath `
            -Executable $executablePath `
            -TaskArguments @('--cuda', "--$Mode") `
            -TaskWorkingDirectory $executableDirectory `
            -GpuIndex $GpuIndex `
            -MinFreeRamGiB $MinFreeRamGiB `
            -MinFreeVramMiB $MinFreeVramMiB `
            -MaxUsedVramMiB $MaxUsedVramMiB `
            -MaxRuntimeSeconds $MaxRuntimeSeconds `
            -StdoutPath $stdoutPath `
            -StderrPath $stderrPath
    } catch {
        $guardFailure = $_.Exception
        throw
    }
    if (-not $guardResult -or $guardResult.Status -cne 'completed' -or [int] $guardResult.ExitCode -ne 0) {
        throw 'Exclusive GPU guard returned an invalid completion result.'
    }
    if ([int] $guardResult.GpuIndex -ne $GpuIndex -or
            [string]::IsNullOrWhiteSpace([string] $guardResult.GpuUuid)) {
        throw 'Exclusive GPU guard returned the wrong GPU index or an empty GPU UUID.'
    }

    $stage = 'runtime_recheck'
    $null = Assert-CurrentRuntimeIdentity $executablePath $guardPath $executableDirectory $cudartPath

    $stage = 'validation'
    $records = @()
    foreach ($line in Get-Content -LiteralPath $stdoutPath) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $records += $line | ConvertFrom-Json
    }
    Assert-Records $records $Mode
    $validation = 'passed'
    $stage = 'completed'
} catch {
    if ($stage -ceq 'validation') { $validation = 'failed' }
    $failure = $_.Exception
} finally {
    if ($environmentChanged) {
        [Environment]::SetEnvironmentVariable('PATH', $priorPath, 'Process')
        $currentNames = @([Environment]::GetEnvironmentVariables('Process').Keys |
            ForEach-Object { [string] $_ } |
            Where-Object {
                $name = $_
                @($scrubPrefixes | Where-Object { $name.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0
            })
        foreach ($name in $currentNames) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
        foreach ($name in @($savedEnvironment.Keys)) {
            [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
        }
        if ($null -eq $priorSentinel) {
            Remove-Item Env:LLAMACPP_QWEN38_ROW_INVARIANCE_GUARD -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable('LLAMACPP_QWEN38_ROW_INVARIANCE_GUARD', $priorSentinel, 'Process')
        }
    }
    try {
        $postflight = Get-Postflight $workspaceRoot
    } catch {
        $postflight = [ordered]@{
            relevant_processes = -1
            process_details = @()
            port_18136_free = $false
            checked_lease_paths = @()
            remaining_gpu_leases = -1
            proof_error = $_.Exception.Message
        }
        if (-not $failure) {
            $failure = [InvalidOperationException]::new('Postflight proof could not be collected.', $_.Exception)
            $stage = 'postflight_failed'
        }
    }
    if (($postflight.relevant_processes -ne 0 -or -not $postflight.port_18136_free -or
            $postflight.remaining_gpu_leases -ne 0) -and -not $failure) {
        $failure = [InvalidOperationException]::new('Postflight proof failed after the guarded diagnostic.')
        $stage = 'postflight_failed'
    }
    $receipt = [ordered]@{
        schema = 'ai-loader-qwen38-row-invariance-guard-receipt/v1'
        created_at = [DateTimeOffset]::UtcNow.ToString('o')
        stage = $stage
        mode = $Mode
        guard_started = $guardStarted
        guard_status = if ($guardFailure) { 'failed' } elseif ($guardResult) { 'completed' } else { 'not_started' }
        diagnostic_validation = $validation
        error = if ($failure) { $failure.Message } else { $null }
        wrapper_sha256 = $wrapperSha256
        executable = if ($executablePath) { [ordered]@{ path = $executablePath; sha256 = $executableSha256 } } else { $null }
        guard_sha256 = $guardSha256
        runtime = $runtime
        environment = $effectiveEnvironment
        guard = $guardResult
        postflight = $postflight
        stdout_sha256 = if (Test-Path -LiteralPath $stdoutPath -PathType Leaf) { Get-Sha256 $stdoutPath } else { $null }
        stderr_sha256 = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) { Get-Sha256 $stderrPath } else { $null }
    }
    if (-not $Force -and (Test-Path -LiteralPath $receiptTargetPath -PathType Leaf)) {
        $receiptTargetPath = Join-Path $outputRoot (
            "$Mode.failure-$([DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')).guard-receipt.json")
    }
    Write-AtomicJson $receiptTargetPath $receipt
    if ($mutexHeld) { try { $mutex.ReleaseMutex() } catch {} }
    $mutex.Dispose()
}

if ($failure) { throw $failure }
Write-Host "Qwen3.8 row-invariance $Mode diagnostic passed: $stdoutPath"
Write-Host "Guard receipt and postflight proof: $receiptTargetPath"
