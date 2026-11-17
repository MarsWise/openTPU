# Claude: reads r8_ddr_diag.log and r8_ddr_check.log (this folder) and says whether DRAM was clean:
# every ECC counter 0 / 0, no run with ECC errors, no wrong word, no failure, judged from the point the
# taps were centred under load (loadcal, which every run on an r8 card now gets) to the end; errors at
# the calibration's own taps before it are counted, not judged. Exit 0 = clean,
# 1 = not clean (the offending lines are printed), 2 = a log is missing. Reads files only.
param([string[]]$Logs = @('r8_ddr_diag.log', 'r8_ddr_check.log'))
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$bad = @()
foreach ($l in $Logs) {
    $p = Join-Path $here $l
    if (-not (Test-Path $p)) { Write-Host "missing: $p"; exit 2 }
    $n = 0; $on = $false; $pre = 0
    foreach ($line in Get-Content $p) {
        $n++
        if ($line -match '######## .*(selfcal \+ loadcal|calibration: selfcal \+ (ternary|int8|fp4))') { $on = $true }
        $why = $null
        if ($line -match 'another claude run is in progress|SYNC FAILED') { $why = 'not run' }
        elseif ($line -match 'Traceback|RuntimeError|RESULT FAIL|ECC_DED True' -and $line -notmatch '\] WARNING|WARNING \d') { $why = 'failure' }
        elseif ($line -match 'ECC \[\((\d+), (\d+)\), \((\d+), (\d+)\)\]' -and
                ([int]$Matches[1] + [int]$Matches[2] + [int]$Matches[3] + [int]$Matches[4]) -gt 0) { $why = 'ECC counters' }
        elseif ($line -match 'ECC (during the reads )?ch0 (\d+) / (\d+), ch1 (\d+) / (\d+)' -and
                ([int]$Matches[2] + [int]$Matches[3] + [int]$Matches[4] + [int]$Matches[5]) -gt 0) { $why = 'ECC counters' }
        elseif ($line -match 'ECC counters meanwhile.*ch0 (\d+) / (\d+), ch1 (\d+) / (\d+)' -and
                ([int]$Matches[1] + [int]$Matches[2] + [int]$Matches[3] + [int]$Matches[4]) -gt 0) { $why = 'ECC counters' }
        elseif ($line -match '(\d+) with ECC errors' -and [int]$Matches[1] -gt 0) { $why = 'runs with ECC errors' }
        elseif ($line -match 'wrong in read 1 / read 2 / both (\d+) / (\d+)' -and ([int]$Matches[1] + [int]$Matches[2]) -gt 0) { $why = 'wrong words' }
        elseif ($line -match '(\d+) lane-beats wrong' -and [int]$Matches[1] -gt 0) { $why = 'BIST lanes' }
        if ($why -and $why -ne 'not run' -and -not $on) { $pre++; continue }
        if ($why) { $bad += "${l}:${n}: [$why] $line" }
    }
    if (-not $on) { $bad += "${l}: no loadcal section (the run did not get that far)" }
    Write-Host "${l}: $pre lines with errors at the calibration's own taps (before loadcal; not judged)"
}
if ($bad.Count) {
    Write-Host "=== DRAM NOT CLEAN ($($bad.Count) lines):"
    $bad | Select-Object -First 40 | ForEach-Object { Write-Host "   $_" }
    exit 1
}
Write-Host "=== DRAM CLEAN: every ECC counter 0 / 0 in $($Logs -join ', ')"
exit 0
