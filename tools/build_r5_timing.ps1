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
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
if ($Jobs -lt 1) { throw 'Jobs must be positive' }
if (-not (Test-Path -LiteralPath $Vivado)) { throw "Vivado not found: $Vivado" }
$buildEnv = @{
    OTPU_SEQ_WC_GEN = '1'; OTPU_TPAIR_GEN = '1'; OTPU_TQUAD_GEN = '1'; OTPU_TFN_GEN = '1'
    TRACE_DEPTH = '16384'; COLL_GATHER = '0'; TMEM_WORDS = '131072'; PCIE_GEN = '2'
    OTPU_FAST = '0'; OTPU_SETUP_MARGIN = '0.15'; OTPU_SYS_MARGIN = ''
    OTPU_MXU_DSP_COLS = '3 4 5 6 7 8 9 10'; OTPU_MXU_BRAM_COLS = '3 4 5 6 7 8 9 10'
    OTPU_MXU_WRAM_COL = '2'; OTPU_PLACE_DIRECTIVE = 'ExtraPostPlacementOpt'
    OTPU_MXU_PLACE = '1'; OTPU_DDR_PHY_PLACE = '1'
    OTPU_MXU_FENCE = $(if ($NoFence) { '0' } else { '1' })
    # r5-m8 db36f818 (after phys_opt -0.503 ns, TMEM's pumped RAM outputs -> the units): the
    # clk2x -> core checks at 0.45 ns of margin while placing and routing (removed for the
    # bitstream), 8 threads, the phys_opt passes at AggressiveExplore, the router at Explore
    OTPU_X2C_MARGIN = '0.45'; OTPU_MAX_THREADS = $(if ($Threads -gt 0) { "$Threads" } else { '' })
    OTPU_PHYSOPT_DIRECTIVE = 'AggressiveExplore'; OTPU_ROUTE_DIRECTIVE = 'Explore'
    OTPU_POSTROUTE_PHYSOPT_DIRECTIVE = 'AggressiveExplore'
    OTPU_TMEM_PBLOCK = ''; OTPU_XDMA_PBLOCK = ''; OTPU_STRICT_R5 = '1'
    OTPU_STAGE_REPORT_DIR = (Join-Path $OutDir 'reports\stages')
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
Write-Output ("r5 timing: revision=$revision mode=$Mode variant=$Variant MCOLS=$($v.mcols) " +
              "core=$($v.mhz)MHz clk2x=$($v.clk2x)MHz hash=$($v.hash)")
Write-Output "Output: $OutDir"
if ($Plan) { $buildEnv.GetEnumerator() | Sort-Object Name | Format-Table -AutoSize; return }
& git -C $repoRoot diff HEAD --quiet --
if ($LASTEXITCODE -ne 0) { throw 'Commit tracked source changes before preparing a release project' }
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
