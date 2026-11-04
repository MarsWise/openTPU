<#
Finish a routed r5 checkpoint (e.g. tools\replay_r5_locality.ps1 -Mode Route's routed\design.dcp)
into reports and, only if setup and hold are met at the real clocks, a bitstream:
tools\vivado\r5_finalize.tcl (the source build's setup margins removed, a post-route phys_opt
while setup fails, the PCIe userclk1 ECO, build.tcl's reports and r5 gate).
  .\tools\finalize_r5.ps1 -Routed C:\myOpenTPU\r5-locality-...\routed\design.dcp
Writes a fresh -OutDir (default C:\myOpenTPU\r5-final-<source>-<time>) with build_manifest.json.
The routed checkpoint and the source build are only read. Never programs a device.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Routed,
    [string]$SourceBuild,
    [string]$OutDir,
    [ValidateSet('auto', 'no')][string]$PhysOpt = 'auto',
    [string]$Vivado = 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $SourceBuild) { $SourceBuild = Join-Path $repoRoot 'build\vivado-r5-timing-db36f818' }
$SourceBuild = [IO.Path]::GetFullPath($SourceBuild)
$Routed = [IO.Path]::GetFullPath($Routed)
if (-not (Test-Path -LiteralPath $Routed -PathType Leaf)) { throw "Routed checkpoint not found: $Routed" }
$sourceCommit = (Get-Content -LiteralPath (Join-Path $SourceBuild 'SOURCE_COMMIT.txt') -Raw).Trim()
if ($sourceCommit -notmatch '^[0-9a-fA-F]{40}$') { throw 'Invalid source build commit' }
$margin = Join-Path $SourceBuild 'otpu_margin.tcl'
$marginArg = if (Test-Path -LiteralPath $margin) { $margin } else { '-' }
if (-not $OutDir) {
    $OutDir = Join-Path (Split-Path $repoRoot -Parent) ("r5-final-{0}-{1}" -f $sourceCommit.Substring(0, 8),
                                                       (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (Test-Path -LiteralPath $OutDir) { throw "Output already exists; choose a fresh OutDir: $OutDir" }
if (-not (Test-Path -LiteralPath $Vivado -PathType Leaf)) { throw "Vivado not found: $Vivado" }
$head = (& git -C $repoRoot rev-parse HEAD).Trim()
Write-Output "Finalize $Routed"
Write-Output "Source build $SourceBuild ($sourceCommit), margins $marginArg, post-route phys_opt $PhysOpt"
Write-Output "Output: $OutDir"
if ($DryRun) { Write-Output 'Dry run complete; Vivado not launched'; return }
New-Item -ItemType Directory -Path $OutDir | Out-Null
$script = Join-Path $PSScriptRoot 'vivado\r5_finalize.tcl'
$meta = [ordered]@{
    source_commit = $sourceCommit; finalize_commit = $head; source_build = $SourceBuild
    routed = $Routed; routed_sha256 = (Get-FileHash -LiteralPath $Routed -Algorithm SHA256).Hash
    margins = $marginArg; physopt = $PhysOpt
    finalize_script_sha256 = (Get-FileHash -LiteralPath $script -Algorithm SHA256).Hash
    status = 'running'
}
$replay = Join-Path (Split-Path (Split-Path $Routed -Parent) -Parent) 'replay_manifest.json'
if (Test-Path -LiteralPath $replay) { $meta['replay_manifest'] = (Get-Content -LiteralPath $replay -Raw | ConvertFrom-Json) }
$manifest = Join-Path $OutDir 'build_manifest.json'
$meta | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8
Push-Location -LiteralPath $OutDir
try {
    & $Vivado -mode batch -notrace -log (Join-Path $OutDir 'finalize.log') -journal (Join-Path $OutDir 'finalize.jou') `
        -source $script -tclargs $Routed $OutDir $marginArg $PhysOpt
    $rc = $LASTEXITCODE
} finally { Pop-Location }
$bit = Join-Path $OutDir 'otpu.bit'
$meta.status = if ($rc -eq 0 -and (Test-Path -LiteralPath $bit)) { 'bitstream' } else { 'no bitstream' }
$meta['exit_code'] = $rc
if (Test-Path -LiteralPath $bit) {
    $meta['bitstream_sha256'] = (Get-FileHash -LiteralPath $bit -Algorithm SHA256).Hash.ToLowerInvariant()
}
$meta | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8
$sum = Join-Path $OutDir 'reports\SUMMARY.txt'
if (Test-Path -LiteralPath $sum) { Get-Content -LiteralPath $sum }
if ($rc -ne 0) { throw "Finalize: no bitstream (exit $rc); reports in $OutDir" }
Write-Output "Bitstream: $bit ($($meta.bitstream_sha256))"
