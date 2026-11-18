<#
Prepare a fresh r5 project from the current committed checkout: -Variant m8 (MCOLS=8, 150 MHz,
the author's r150o placement recipe; the default) or m4 (MCOLS=4, 133.33 MHz, r4's recipe).
  .\tools\build_r5_timing.ps1 -Mode Prepare
  .\tools\build_r5_timing.ps1 -Mode Synth
  .\tools\build_r5_timing.ps1 -Mode Build [-Variant m4] [-NoFence]
  .\tools\build_r5_timing.ps1 -Mode Impl -OutDir build\vivado-r5-timing-<rev>
-NoFence (m8): leave the MXU's DSP columns' free sites to the placer (otpu_mxu_place.tcl's fence).
-Threads N: general.maxThreads in the runs (default 8; 0 leaves Vivado's default, 2 on Windows).
-CoreMhz (m8): another core clock than 150 MHz (core_clk2x twice it; core_clock.tcl needs an even
MMCM divider: 141.67 = 850 / 6, 137.5 = 825 / 6), e.g. 137.5; default OutDir
build\vivado-r5-m8-<MHz>-<rev>. The CFG_HASH does not change (the clock is not in the Config).
Prepare creates the project only. Synth runs IP/core synthesis and structural gates.
Build runs synthesis, implementation and writes a bitstream only if setup/hold pass.
Impl reruns implementation only, on an existing project's finished synthesis (OutDir required):
for implementation changes (constraints, directives, margins) on the same RTL. It refuses when
anything synthesized changed between the project's SOURCE_COMMIT.txt and HEAD (only docs/,
opentpu/, tests/, tools/, sim/, boards/ypcb-00338/vivado/ and the .tcl constraints may differ),
moves the previous reports to reports_prev_<time>, and records HEAD in IMPL_COMMIT.txt.
No fetch, checkout, merge or programming.
#>
[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Synth', 'Build', 'Impl')][string]$Mode = 'Prepare',
    [ValidateSet('m8', 'm4')][string]$Variant = 'm8',
    [int]$Jobs = 8,
    [string]$Vivado = 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat',
    [string]$OutDir,
    [switch]$NoFence,
    [string]$CoreMhz = '',
    [int]$Threads = 8,
    [switch]$TernaryGroupScales,
    [switch]$VectorCompare,
    [switch]$BinaryWeights,
    [switch]$BinaryGroupScales,
    [switch]$UnsignedBytes,
    [switch]$R8,                # r8's units, including VOP BFLY
    [switch]$Plan
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$revision = (& git -C $repoRoot rev-parse --short=8 HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot read repository revision' }
$sourceCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
if (-not $OutDir) {
    $OutDir = Join-Path $repoRoot $(if ($Variant -eq 'm8' -and $CoreMhz) { "build\vivado-r5-m8-$CoreMhz-$revision" }
                                    elseif ($Variant -eq 'm8') { "build\vivado-r5-timing-$revision" }
                                    else { "build\vivado-r5-$Variant-$revision" })
    if ($TernaryGroupScales) { $OutDir += '-tgsc' }
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
if ($Jobs -lt 1) { throw 'Jobs must be positive' }
if (-not (Test-Path -LiteralPath $Vivado)) { throw "Vivado not found: $Vivado" }
$buildEnv = @{
    OTPU_SEQ_WC_GEN = '1'; OTPU_TPAIR_GEN = '1'; OTPU_TQUAD_GEN = '1'; OTPU_TFN_GEN = '1'
    OTPU_TGSC_GEN = $(if ($TernaryGroupScales) { '1' } else { '0' })
    OTPU_VCMP_GEN = $(if ($VectorCompare) { '1' } else { '0' })
    OTPU_TBIN_GEN = $(if ($BinaryWeights) { '1' } else { '0' })
    OTPU_TBGSC_GEN = $(if ($BinaryGroupScales) { '1' } else { '0' })
    OTPU_VU8_GEN = $(if ($UnsignedBytes) { '1' } else { '0' })
    # TMM GSC block-major layout (GRQ): part of the GSC adapter, reported with TGSC
    OTPU_TGRQ_GEN = $(if ($TernaryGroupScales) { '1' } else { '0' })
    OTPU_VRAND_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_TCQ_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_TATAN_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_QST4_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_LDX_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_TD5_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_VBFLY_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_VAFF_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_QST6_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_VCSWP_GEN = $(if ($R8) { '1' } else { '0' })
    OTPU_TGQ4_GEN = $(if ($R8) { '1' } else { '0' })
    TRACE_DEPTH = $(if ($env:TRACE_DEPTH) { $env:TRACE_DEPTH } else { '16384' }); COLL_GATHER = '0'; TMEM_WORDS = '131072'; PCIE_GEN = '2'
    OTPU_FAST = '0'; OTPU_SETUP_MARGIN = '0.15'
    # the memory margins (margins.tcl) from the caller's environment: none unless set (r8_build_ddr.cmd)
    OTPU_SYS_MARGIN = $(if ($env:OTPU_SYS_MARGIN) { $env:OTPU_SYS_MARGIN } else { '' })
    OTPU_WR_MARGIN = $(if ($env:OTPU_WR_MARGIN) { $env:OTPU_WR_MARGIN } else { '' })
    OTPU_UC2_MARGIN = $(if ($env:OTPU_UC2_MARGIN) { $env:OTPU_UC2_MARGIN } else { '' })
    # the MXU's columns (otpu_mxu_place.tcl): r5's 3..10 / weight RAMs 2 unless the caller sets them;
    # OTPU_DSP_PROHIBIT_COLS keeps every other DSP off those DSP48 columns (an experiment)
    OTPU_MXU_DSP_COLS = $(if ($env:OTPU_MXU_DSP_COLS) { $env:OTPU_MXU_DSP_COLS } else { '3 4 5 6 7 8 9 10' })
    OTPU_MXU_BRAM_COLS = $(if ($env:OTPU_MXU_BRAM_COLS) { $env:OTPU_MXU_BRAM_COLS } else { '3 4 5 6 7 8 9 10' })
    OTPU_MXU_WRAM_COL = $(if ($env:OTPU_MXU_WRAM_COL) { $env:OTPU_MXU_WRAM_COL } else { '2' }); OTPU_PLACE_DIRECTIVE = 'ExtraPostPlacementOpt'
    OTPU_DSP_PROHIBIT_COLS = $(if ($env:OTPU_DSP_PROHIBIT_COLS) { $env:OTPU_DSP_PROHIBIT_COLS } else { '' })
    # otpu_mem_ch's write register (create_project.tcl): 0 builds r5's write path
    OTPU_MEMCH_WREG = $(if ($env:OTPU_MEMCH_WREG) { $env:OTPU_MEMCH_WREG } else { '1' })
    OTPU_SIGNOFF_STAGE2 = $(if ($env:OTPU_SIGNOFF_STAGE2) { $env:OTPU_SIGNOFF_STAGE2 } else { '' })
    OTPU_MXU_PLACE = '1'; OTPU_DDR_PHY_PLACE = '1'
    # the PHY keep-out (otpu_ddr_phy_place.tcl) and its width from the caller's environment: off unless set
    OTPU_DDR_PHY_KEEPOUT = $(if ($env:OTPU_DDR_PHY_KEEPOUT) { $env:OTPU_DDR_PHY_KEEPOUT } else { '0' })
    OTPU_DDR_PHY_COLS = $(if ($env:OTPU_DDR_PHY_COLS) { $env:OTPU_DDR_PHY_COLS } else { '' })
    OTPU_MXU_FENCE = $(if ($NoFence) { '0' } else { '1' })
    # r5-m8 db36f818 (after phys_opt -0.503 ns, TMEM's pumped RAM outputs -> the units): the
    # clk2x -> core checks at 0.45 ns of margin while placing and routing (removed for the
    # bitstream), 8 threads, the phys_opt passes at AggressiveExplore, the router at Explore
    OTPU_X2C_MARGIN = '0.45'; OTPU_MAX_THREADS = $(if ($Threads -gt 0) { "$Threads" } else { '' })
    OTPU_PHYSOPT_DIRECTIVE = 'AggressiveExplore'; OTPU_ROUTE_DIRECTIVE = 'Explore'
    OTPU_POSTROUTE_PHYSOPT_DIRECTIVE = 'AggressiveExplore'
    OTPU_TMEM_PBLOCK = ''; OTPU_XDMA_PBLOCK = ''; OTPU_STRICT_R5 = '1'
    # the VPU's pblock from the caller's environment (r8_build_ddr.cmd): none unless set
    OTPU_VPU_PBLOCK = $(if ($env:OTPU_VPU_PBLOCK) { $env:OTPU_VPU_PBLOCK } else { '' })
    # incremental implementation from an earlier routed checkpoint (r8_build_incr.cmd): none unless set
    OTPU_INCR_DCP = $(if ($env:OTPU_INCR_DCP) { $env:OTPU_INCR_DCP } else { '' })
    OTPU_INCR_DIRECTIVE = $(if ($env:OTPU_INCR_DIRECTIVE) { $env:OTPU_INCR_DIRECTIVE } else { '' })
    # the LiteDRAM core's read-to-write turnaround (litedram_rtw.tcl, r8_build_rtw4.cmd): 3 unless set
    OTPU_LITEDRAM_RTW = $(if ($env:OTPU_LITEDRAM_RTW) { $env:OTPU_LITEDRAM_RTW } else { '' })
    OTPU_STAGE_REPORT_DIR = (Join-Path $OutDir 'reports\stages')
}
# capabilities left out of the build (OTPU_DROP_FEATURES, e.g. "TCQ TD5 TBIN TBGSC" or
# "TRACE"): their OTPU_<f>_GEN 0, so CAPS2 clears them and the host refuses their programs; the
# MXU leaves TBIN / TCQ / TD5's weight decode out (otpu_mxu HAS_BIN / HAS_CQ / HAS_T5: area).
# TRACE: TRACE_DEPTH 0 (no trace buffer: -32 BRAM36 at 16384). Not set: every r8 capability.
if ($env:OTPU_DROP_FEATURES) {
    foreach ($f in ($env:OTPU_DROP_FEATURES -split '[ ,;]+' | Where-Object { $_ })) {
        $f = $f.ToUpperInvariant()
        if ($f -eq 'TRACE') { $buildEnv.TRACE_DEPTH = '0'; continue }
        $k = "OTPU_${f}_GEN"
        if (-not $buildEnv.ContainsKey($k)) { throw "OTPU_DROP_FEATURES: no capability '$f'" }
        $buildEnv[$k] = '0'
    }
    if ($buildEnv.OTPU_TBIN_GEN -eq '0' -and $buildEnv.OTPU_TBGSC_GEN -ne '0') {
        throw 'OTPU_DROP_FEATURES: TBIN needs TBGSC left out too'
    }
    Write-Host "=== capabilities left out: $env:OTPU_DROP_FEATURES"
}
# m4: r4's recipe (bb27096f: WNS +0.031 / WHS +0.018 at 133.33 MHz) with r5's units -- the
# default placer, no MXU floorplan, no extra setup margin, 64K TMEM words
$v = if ($Variant -eq 'm8') {
    @{ mcols = 8; mhz = '150'; hash = '3b7f244e'; tmem = '131072'; clk2x = 300 }
} else {
    @{ mcols = 4; mhz = '133.33'; hash = '603d5a6f'; tmem = '65536'; clk2x = 266.67 }
}
if ($CoreMhz) {
    $cm = 0.0
    if ($Variant -ne 'm8') { throw '-CoreMhz is for -Variant m8' }
    if (-not [double]::TryParse($CoreMhz, [Globalization.NumberStyles]::Float,
                                [Globalization.CultureInfo]::InvariantCulture, [ref]$cm) -or
        $cm -lt 100 -or $cm -gt 150) { throw "-CoreMhz '$CoreMhz': 100 to 150" }
    $v.mhz = $CoreMhz; $v.clk2x = [math]::Round(2 * $cm, 2)
}
if ($Variant -eq 'm4') {
    $buildEnv.TMEM_WORDS = $v.tmem
    foreach ($k in 'OTPU_SETUP_MARGIN', 'OTPU_MXU_DSP_COLS', 'OTPU_MXU_BRAM_COLS', 'OTPU_MXU_WRAM_COL',
                   'OTPU_PLACE_DIRECTIVE', 'OTPU_MXU_PLACE', 'OTPU_DDR_PHY_PLACE', 'OTPU_MXU_FENCE',
                   'OTPU_X2C_MARGIN', 'OTPU_PHYSOPT_DIRECTIVE', 'OTPU_ROUTE_DIRECTIVE',
                   'OTPU_POSTROUTE_PHYSOPT_DIRECTIVE') {
        $buildEnv[$k] = ''
    }
}
if ($BinaryGroupScales -and (-not $BinaryWeights -or -not $TernaryGroupScales)) { throw 'Binary group scales require BIN and GSC' }
if ($R8 -and -not ($TernaryGroupScales -and $BinaryWeights -and $BinaryGroupScales)) { throw 'r8 (TMM CQ) builds on GSC, BIN and BIN GSC' }
if ($TernaryGroupScales -or $VectorCompare -or $BinaryWeights -or $BinaryGroupScales -or $UnsignedBytes) {
    if ($Variant -ne 'm8') { throw 'The native Bonsai 2 recipe is m8' }
    $gsc = [int][bool]$TernaryGroupScales
    $vcmp = [int][bool]$VectorCompare
    $tbin = [int][bool]$BinaryWeights
    $tbgsc = [int][bool]$BinaryGroupScales
    $vu8 = [int][bool]$UnsignedBytes
    $rb8 = [int][bool]$R8         # (not $r8: PowerShell names are case-blind, $r8 IS the switch)
    # CFG_HASH from the capabilities the build gets ($buildEnv: the switches less OTPU_DROP_FEATURES)
    $capList = @()
    foreach ($c in 'TGSC','VCMP','TBIN','TBGSC','VU8','TGRQ','VRAND','TCQ','TATAN','QST4','LDX','TD5','VBFLY','VAFF','QST6','VCSWP','TGQ4') {
        $capList += ('{0}=bool({1})' -f $c, [int]($buildEnv["OTPU_${c}_GEN"] -eq '1'))
    }
    $caps = $capList -join ', '
    $v.hash = (& python -c "import sys,dataclasses; sys.path.insert(0, r'$repoRoot'); from tools.bonsai2_validate import configuration; from opentpu.target import arch_hash; print(f'{arch_hash(dataclasses.replace(configuration(), $caps)):08x}')").Trim()
    if ($LASTEXITCODE -ne 0 -or $v.hash -notmatch '^[0-9a-f]{8}$') { throw 'Cannot calculate CFG_HASH' }
}
$buildLabel = if ($VectorCompare -or $BinaryWeights) { 'feature-enabled experimental' } else { 'r5 timing' }
Write-Output ("${buildLabel}: revision=$revision mode=$Mode variant=$Variant MCOLS=$($v.mcols) " +
              "core=$($v.mhz)MHz clk2x=$($v.clk2x)MHz hash=$($v.hash)")
Write-Output "Output: $OutDir"
if ($Plan) { $buildEnv.GetEnumerator() | Sort-Object Name | Format-Table -AutoSize; return }
& git -C $repoRoot diff HEAD --quiet --
if ($LASTEXITCODE -ne 0) { throw 'Commit tracked source changes before preparing a release project' }
$untrackedSources = @(& git -C $repoRoot ls-files --others --exclude-standard -- rtl/ boards/ opentpu/ sim/ targets/ tools/)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect untracked source files' }
if ($untrackedSources.Count -gt 0) {
    throw ("Commit or explicitly remove untracked source files before a release project: " + ($untrackedSources -join ', '))
}
$synthCommit = $sourceCommit
if ($Mode -eq 'Impl') {
    if (-not $PSBoundParameters.ContainsKey('OutDir')) { throw 'Impl needs -OutDir (an existing project)' }
    if (-not (Test-Path -LiteralPath (Join-Path $OutDir 'otpu.xpr'))) { throw "No project in $OutDir" }
    $manifest0 = Join-Path $OutDir 'SOURCE_COMMIT.txt'
    if (-not (Test-Path -LiteralPath $manifest0)) { throw "No SOURCE_COMMIT.txt in $OutDir" }
    $synthCommit = (Get-Content -LiteralPath $manifest0 -Raw).Trim()
    $changed = @(& git -C $repoRoot diff --name-only $synthCommit HEAD --)
    if ($LASTEXITCODE -ne 0) { throw "Cannot compare $synthCommit with HEAD" }
    $synthChanged = @($changed | Where-Object {
        $_ -notmatch '^(docs|opentpu|tests|tools|sim)/' -and
        $_ -notmatch '^boards/ypcb-00338/vivado/(build|margins|impl_directives|create_project|userclk1_reroute)\.tcl$' -and
        $_ -notmatch '^boards/ypcb-00338/constraints/[^/]+\.tcl$' -and
        $_ -notmatch '^[^/]+\.md$' })
    if ($synthChanged.Count) {
        throw ("Synthesized sources changed since $synthCommit (run -Mode Build): " + ($synthChanged -join ', '))
    }
    if (Get-Process -Name vivado -ErrorAction SilentlyContinue) {
        throw 'A Vivado process is running: let it finish (or stop it) before rerunning implementation'
    }
    $buildEnv.OTPU_REGEN_MARGIN = '1'
    Write-Output ("Impl: synthesis of $synthCommit reused; implementation from HEAD $sourceCommit " +
                  "($($changed.Count) files differ, none synthesized)")
}
$savedEnv = @{}
try {
    foreach ($key in $buildEnv.Keys) {
        $savedEnv[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $buildEnv[$key], 'Process')
    }
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    $manifest = Join-Path $OutDir 'SOURCE_COMMIT.txt'
    if (Test-Path -LiteralPath (Join-Path $OutDir 'otpu.xpr')) {
        if (-not (Test-Path -LiteralPath $manifest) -or
            (Get-Content -LiteralPath $manifest -Raw).Trim() -ne $synthCommit) {
            throw 'Existing project belongs to a different or unknown source revision; choose a new OutDir'
        }
    } elseif ($Mode -eq 'Impl') {
        throw "No project in $OutDir"
    } else {
        & $Vivado -mode batch -nojournal -log (Join-Path $OutDir 'create_project.log') `
            -source (Join-Path $repoRoot 'boards\ypcb-00338\vivado\create_project.tcl') `
            -tclargs 1066 $OutDir $v.mcols $v.mhz $revision 2 8 16 1 systolic 1 $v.hash $v.mcols 3 0
        if ($LASTEXITCODE -ne 0) { throw 'Vivado project creation failed' }
        Set-Content -LiteralPath $manifest -Value $sourceCommit -Encoding ascii
    }
    if ($Mode -eq 'Impl') {
        $rep = Join-Path $OutDir 'reports'
        if (Test-Path -LiteralPath $rep) {
            $prev = Join-Path $OutDir ('reports_prev_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
            Move-Item -LiteralPath $rep -Destination $prev
            Write-Output "Previous reports moved to $prev"
        }
        Set-Content -LiteralPath (Join-Path $OutDir 'IMPL_COMMIT.txt') -Value $sourceCommit -Encoding ascii
    }
    if ($Mode -ne 'Prepare') {
        $flowMode = if ($Mode -eq 'Synth') { 'synth' } elseif ($Mode -eq 'Impl') { 'impl' } else { 'full' }
        & $Vivado -mode batch -nojournal -log (Join-Path $OutDir "$flowMode.log") `
            -source (Join-Path $repoRoot 'boards\ypcb-00338\vivado\build.tcl') `
            -tclargs $OutDir $Jobs $flowMode
        if ($LASTEXITCODE -ne 0) { throw "Vivado $flowMode failed; see $OutDir" }
    }
    $metadata = [ordered]@{
        source_commit = $synthCommit; impl_commit = $sourceCommit; build_id = $revision
        part = 'xc7k480tffg1156-2'
        cfg_hash = $v.hash; mode = $Mode; variant = $Variant; core_mhz = $v.mhz
        clk2x_mhz = $v.clk2x; mcols = $v.mcols; mxu_e = $v.mcols; lanes = 8; act_rows = 16
        tmem_words = [int]$v.tmem
        environment = $buildEnv
    }
    if ($Mode -eq 'Build' -or $Mode -eq 'Impl') {
        $metadata['bitstream_sha256'] = (Get-FileHash -Algorithm SHA256 -LiteralPath `
            (Join-Path $OutDir 'otpu.bit')).Hash.ToLowerInvariant()
    }
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath `
        (Join-Path $OutDir 'build_manifest.json') -Encoding utf8
    Write-Output "Ready: $OutDir"
} finally {
    foreach ($key in $savedEnv.Keys) {
        [Environment]::SetEnvironmentVariable($key, $savedEnv[$key], 'Process')
    }
}
