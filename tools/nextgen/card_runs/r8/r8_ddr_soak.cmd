@echo off
rem Claude: DDR3 soak of one build, ~35 min: load it (asks yes), PCIe (sudo password), r8_ddr_check with
rem loadcal under load, 4 passes (BitNet steps, the int8 run_rows, vLLM), then the gate (every ECC counter
rem 0 / 0 after loadcal). No other test. Log copied to runs\soak_<build>_<time>\.
rem   r8_ddr_soak.cmd 2c800a8f [parallel]
setlocal
if "%~1"=="" (echo usage: r8_ddr_soak.cmd ^<build^> [parallel] & pause & exit /b 1)
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%t"
set "OUT=%~dp0runs\soak_%~1_%TS%"
mkdir "%OUT%" || (echo cannot create %OUT% & pause & exit /b 1)
call "%~dp0r8_stop.cmd" check
if errorlevel 1 (echo The card is busy or the host is unreachable: wait, or r8_stop.cmd. & pause & exit /b 1)
call "%~dp0r8_load_newest.cmd" %1 %2
if errorlevel 1 (echo NOT LOADED: stopping. & pause & exit /b 1)
call "%~dp0r8_restore_pcie.cmd"
if errorlevel 1 (echo PCIe restore failed: stopping. & pause & exit /b 1)
set PASSES=4
set "KINDS=ternary:4"
call "%~dp0r8_ddr_check.cmd" < nul
copy /y "%~dp0r8_ddr_check.log" "%OUT%\" >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0r8_ddr_gate.ps1" -Logs r8_ddr_check.log > "%OUT%\gate.txt"
type "%OUT%\gate.txt"
echo === logs: %OUT%
pause
