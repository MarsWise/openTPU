<# Build the experimental r8 m8 design (the r7 base + VOP RAND/BFLY, TMM CQ, TFN atan, QST Q4, LDX, TMM T5) at 133.33 MHz. Never programs FPGA.
   Prepare creates the Vivado project; Synth runs synthesis and structural
   checks; Build runs implementation and emits a bitstream only after the
   existing setup/hold gates pass. A successful build does not establish
   complete model coverage or remove the need to inspect CDC/DRC reports.
#>
[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Synth', 'Build', 'Impl')][string]$Mode = 'Build',
    [int]$Jobs = 4,
    [int]$Threads = 4,
    [string]$Vivado = 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat',
    [string]$OutDir = '',
    [switch]$Plan
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$safeRoot = $root.Replace('\', '/')
$revision = (& git -c "safe.directory=$safeRoot" -C $root rev-parse --short=8 HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot read r8 revision' }
if (-not $OutDir) { $OutDir = Join-Path $root "build\vivado-r8-m8-133.33-$revision" }
Write-Warning 'Experimental r8: full FPGA-only model coverage and hardware timing are unverified.'

# Give this process and its children a single scoped safe.directory entry;
# do not alter the user's global Git configuration or overwrite existing entries.
$oldCount = $env:GIT_CONFIG_COUNT
$count = if ($null -eq $oldCount -or $oldCount -eq '') { 0 } else { [int]$oldCount }
$keyName = "GIT_CONFIG_KEY_$count"
$valueName = "GIT_CONFIG_VALUE_$count"
$oldKey = [Environment]::GetEnvironmentVariable($keyName, 'Process')
$oldValue = [Environment]::GetEnvironmentVariable($valueName, 'Process')
try {
    [Environment]::SetEnvironmentVariable($keyName, 'safe.directory', 'Process')
    [Environment]::SetEnvironmentVariable($valueName, $safeRoot, 'Process')
    $env:GIT_CONFIG_COUNT = [string]($count + 1)
    $args0 = @{
        Mode = $Mode; Variant = 'm8'; CoreMhz = '133.33'; Jobs = $Jobs; Threads = $Threads
        Vivado = $Vivado; OutDir = $OutDir; TernaryGroupScales = $true
        VectorCompare = $true; BinaryWeights = $true; BinaryGroupScales = $true
        UnsignedBytes = $true; R8 = $true; Plan = $Plan
    }
    & (Join-Path $PSScriptRoot 'build_r5_timing.ps1') @args0
    if ($LASTEXITCODE -ne 0) { throw "r8 build failed: $LASTEXITCODE" }
} finally {
    [Environment]::SetEnvironmentVariable($keyName, $oldKey, 'Process')
    [Environment]::SetEnvironmentVariable($valueName, $oldValue, 'Process')
    [Environment]::SetEnvironmentVariable('GIT_CONFIG_COUNT', $oldCount, 'Process')
}
