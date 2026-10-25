<#
Prepare a fresh r5 project from the current committed checkout: -Variant m8 (MCOLS=8, 150 MHz,
the author's r150o placement recipe; the default) or m4 (MCOLS=4, 133.33 MHz, r4's recipe).
  .\tools\build_r5_timing.ps1 -Mode Prepare
  .\tools\build_r5_timing.ps1 -Mode Synth
  .\tools\build_r5_timing.ps1 -Mode Build [-Variant m4] [-NoFence]
-NoFence (m8): leave the MXU's DSP columns' free sites to the placer (otpu_mxu_place.tcl's fence).
Prepare creates the project only. Synth runs IP/core synthesis and structural gates.
Build runs synthesis, implementation and writes a bitstream only if setup/hold pass.
No fetch, checkout, merge, programming or reuse of an older RTL checkpoint.
#>
[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Synth', 'Build')][string]$Mode = 'Prepare',
    [ValidateSet('m8', 'm4')][string]$Variant = 'm8',
    [int]$Jobs = 8,
    [string]$Vivado = 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat',
    [string]$OutDir,
    [switch]$NoFence,
    [switch]$Plan
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$revision = (& git -C $repoRoot rev-parse --short=8 HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot read repository revision' }
$sourceCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
if (-not $OutDir) {
    $OutDir = Join-Path $repoRoot $(if ($Variant -eq 'm8') { "build\vivado-r5-timing-$revision" }
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
if ($Variant -eq 'm4') {
    $buildEnv.TMEM_WORDS = $v.tmem
    foreach ($k in 'OTPU_SETUP_MARGIN', 'OTPU_MXU_DSP_COLS', 'OTPU_MXU_BRAM_COLS', 'OTPU_MXU_WRAM_COL',
                   'OTPU_PLACE_DIRECTIVE', 'OTPU_MXU_PLACE', 'OTPU_DDR_PHY_PLACE', 'OTPU_MXU_FENCE') {
        $buildEnv[$k] = ''
    }
}
Write-Output ("r5 timing: revision=$revision mode=$Mode variant=$Variant MCOLS=$($v.mcols) " +
              "core=$($v.mhz)MHz clk2x=$($v.clk2x)MHz hash=$($v.hash)")
Write-Output "Output: $OutDir"
if ($Plan) { $buildEnv.GetEnumerator() | Sort-Object Name | Format-Table -AutoSize; return }
& git -C $repoRoot diff HEAD --quiet --
if ($LASTEXITCODE -ne 0) { throw 'Commit tracked source changes before preparing a release project' }
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
            (Get-Content -LiteralPath $manifest -Raw).Trim() -ne $sourceCommit) {
            throw 'Existing project belongs to a different or unknown source revision; choose a new OutDir'
        }
    } else {
        & $Vivado -mode batch -nojournal -log (Join-Path $OutDir 'create_project.log') `
            -source (Join-Path $repoRoot 'boards\ypcb-00338\vivado\create_project.tcl') `
            -tclargs 1066 $OutDir $v.mcols $v.mhz $revision 2 8 16 1 systolic 1 $v.hash $v.mcols 3 0
        if ($LASTEXITCODE -ne 0) { throw 'Vivado project creation failed' }
        Set-Content -LiteralPath $manifest -Value $sourceCommit -Encoding ascii
    }
    if ($Mode -ne 'Prepare') {
        $flowMode = if ($Mode -eq 'Synth') { 'synth' } else { 'full' }
        & $Vivado -mode batch -nojournal -log (Join-Path $OutDir "$flowMode.log") `
            -source (Join-Path $repoRoot 'boards\ypcb-00338\vivado\build.tcl') `
            -tclargs $OutDir $Jobs $flowMode
        if ($LASTEXITCODE -ne 0) { throw "Vivado $flowMode failed; see $OutDir" }
    }
    $metadata = [ordered]@{
        source_commit = $sourceCommit; build_id = $revision; part = 'xc7k480tffg1156-2'
        cfg_hash = $v.hash; mode = $Mode; variant = $Variant; core_mhz = $v.mhz
        clk2x_mhz = $v.clk2x; mcols = $v.mcols; mxu_e = $v.mcols; lanes = 8; act_rows = 16
        tmem_words = [int]$v.tmem
        environment = $buildEnv
    }
    if ($Mode -eq 'Build') {
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
