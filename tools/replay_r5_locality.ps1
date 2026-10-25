<#
Experimental placement from an existing post-opt DCP, with endpoint-guided locality.
  .\tools\replay_r5_locality.ps1 -Mode Place
  .\tools\replay_r5_locality.ps1 -Mode Place -Scope None   # same directives, no new floorplan
  .\tools\replay_r5_locality.ps1 -Mode Route -Checkpoint C:\...\physopt\design.dcp
Default: soft pblocks, both RMW and TMEM. No synthesis, project reset or bitstream.
Every run writes a new directory outside the source build. -DryRun launches nothing.
#>
[CmdletBinding()]
param(
    [ValidateSet('Plan', 'Place', 'Route')][string]$Mode = 'Place',
    [ValidateSet('Both', 'Rmw', 'Tmem', 'None')][string]$Scope = 'Both',
    [ValidateSet('Soft', 'Hard')][string]$Strength = 'Soft',
    [ValidateRange(1, 8)][int]$Threads = 8,
    [string]$SourceBuild,
    [string]$AnalysisDir = 'C:\myOpenTPU\r5_db36_analysis',
    [string]$Checkpoint,
    [string]$OutDir,
    [string]$Python = 'python',
    [string]$Vivado = 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $SourceBuild) { $SourceBuild = Join-Path $repoRoot 'build\vivado-r5-timing-db36f818' }
$SourceBuild = [IO.Path]::GetFullPath($SourceBuild).TrimEnd('\', '/')
$AnalysisDir = [IO.Path]::GetFullPath($AnalysisDir)
$sourceCommit = (Get-Content -LiteralPath (Join-Path $SourceBuild 'SOURCE_COMMIT.txt') -Raw).Trim()
if ($sourceCommit -notmatch '^[0-9a-fA-F]{40}$') { throw 'Invalid source build commit' }
if ($Mode -eq 'Route' -and -not $Checkpoint) {
    throw 'Route requires -Checkpoint pointing to the saved physopt/design.dcp of the new experiment'
}
if ($Mode -eq 'Route' -and ($PSBoundParameters.ContainsKey('Scope') -or $PSBoundParameters.ContainsKey('Strength'))) {
    throw 'Route inherits constraints from -Checkpoint; do not specify Scope or Strength'
}
if (-not $Checkpoint) { $Checkpoint = Join-Path $SourceBuild 'otpu.runs\impl_1\otpu_fpga_top_ld_opt.dcp' }
$Checkpoint = [IO.Path]::GetFullPath($Checkpoint)
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw "Checkpoint not found: $Checkpoint" }
if (-not (Test-Path -LiteralPath $Vivado -PathType Leaf)) { throw "Vivado not found: $Vivado" }
if ($Mode -ne 'Route') {
    $refDcp = [IO.Path]::GetFullPath((Join-Path $SourceBuild 'reports\stages\routed\design.dcp'))
    $summary = Get-Content -LiteralPath (Join-Path $AnalysisDir 'summary.txt') -Raw
    $firstLine = ($summary -split '\r?\n')[0]
    if (-not $firstLine.StartsWith('checkpoint ') -or
        [IO.Path]::GetFullPath($firstLine.Substring(11)) -ne $refDcp) {
        throw 'Path analysis belongs to a different reference checkpoint'
    }
    $paths = Join-Path $AnalysisDir 'paths.csv'
    if (-not (Test-Path -LiteralPath $paths -PathType Leaf)) { throw 'paths.csv not found' }
}
if (-not $OutDir) {
    $tag = "{0}-{1}-{2}-{3}-{4}" -f $sourceCommit.Substring(0, 8), $Scope.ToLowerInvariant(),
        $Strength.ToLowerInvariant(), $Mode.ToLowerInvariant(), (Get-Date -Format 'yyyyMMdd-HHmmss')
    $OutDir = Join-Path (Split-Path $repoRoot -Parent) "r5-locality-$tag"
}
$OutDir = [IO.Path]::GetFullPath($OutDir).TrimEnd('\', '/')
if ($OutDir -eq $SourceBuild -or $OutDir.StartsWith($SourceBuild + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must be outside the preserved source build'
}
if (Test-Path -LiteralPath $OutDir) { throw "Output already exists; choose a fresh OutDir: $OutDir" }
$script = Join-Path $PSScriptRoot 'vivado\r5_locality_replay.tcl'
Write-Output "Replay $Mode / $Scope / $Strength; source $sourceCommit"
Write-Output "Input:  $Checkpoint"
Write-Output "Output: $OutDir"
if ($DryRun) { Write-Output 'Dry run complete; Vivado not launched'; return }

New-Item -ItemType Directory -Path $OutDir | Out-Null
$targets = Join-Path $OutDir 'targets.tsv'
$metadata = [ordered]@{
    source_commit = $sourceCommit; source_build = $SourceBuild
    checkpoint = $Checkpoint; checkpoint_sha256 = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash
    mode = $Mode; scope = $(if ($Mode -eq 'Route') { 'inherited' } else { $Scope })
    strength = $(if ($Mode -eq 'Route') { 'inherited' } else { $Strength }); threads = $Threads
    place_directive = 'ExtraPostPlacementOpt'; physopt_directive = 'Explore'
    route_directive = 'NoTimingRelaxation'; timing_margins = 'inherited unchanged from checkpoint'
    status = 'prepared'
    replay_script_sha256 = (Get-FileHash -LiteralPath $script -Algorithm SHA256).Hash
    constraint_script_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'vivado\r5_locality_constraints.tcl') -Algorithm SHA256).Hash
}
if ($Mode -ne 'Route') {
    & $Python (Join-Path $PSScriptRoot 'r5_locality_targets.py') --paths $paths --output $targets `
        --summary (Join-Path $OutDir 'paths_summary.json') --scope $Scope.ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'Locality target selection failed' }
    $metadata['paths_sha256'] = (Get-FileHash -LiteralPath $paths -Algorithm SHA256).Hash
}
$manifest = Join-Path $OutDir 'replay_manifest.json'
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifest -Encoding utf8
Push-Location -LiteralPath $OutDir
try {
    & $Vivado -mode batch -notrace -log (Join-Path $OutDir 'replay.log') -journal (Join-Path $OutDir 'replay.jou') `
        -source $script -tclargs $Checkpoint $OutDir $targets $Mode.ToLowerInvariant() `
        $Strength.ToLowerInvariant() $Threads
    $result = $LASTEXITCODE
    $metadata.status = if ($result -eq 0) { 'completed' } else { 'failed' }
    $metadata['exit_code'] = $result
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifest -Encoding utf8
    if ($result -ne 0) { throw "Replay failed; preserved logs/checkpoints: $OutDir" }
} finally { Pop-Location }
Write-Output "Replay complete: $OutDir"
