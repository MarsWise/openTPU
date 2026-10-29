<# Experimental r6 build. Does not program the FPGA.
   Run from the r6 checkout after committing all tracked sources:
     powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\build_r6_133.ps1
   Modes: Prepare, Synth, Build. Output is revision-specific and independent of r5.
   A successful timing build is not proof of real-model or no-host-compute coverage.
#>
[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Synth', 'Build')][string]$Mode = 'Build',
    [int]$Jobs = 8,
    [int]$Threads = 8,
    [string]$Vivado = 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat',
    [string]$OutDir = ''
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$rev = (& git -C $root rev-parse --short=8 HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot read r6 revision' }
if (-not $OutDir) { $OutDir = Join-Path $root "build\vivado-r6-m8-133.33-$rev" }
Write-Warning 'Experimental r6: BIN PAIR/QUAD and full FPGA-only model coverage are not complete.'
& (Join-Path $PSScriptRoot 'build_r5_timing.ps1') -Mode $Mode -Variant m8 -CoreMhz '133.33' `
    -TernaryGroupScales -VectorCompare -BinaryWeights -BinaryGroupScales -Jobs $Jobs -Threads $Threads `
    -Vivado $Vivado -OutDir $OutDir
if ($LASTEXITCODE -ne 0) { throw 'r6 build failed' }
