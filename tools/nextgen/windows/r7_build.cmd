@echo off
rem Claude: build r7 (m8, 133.33 MHz; TGSC + GRQ, VCMP, TBIN, TBGSC, VU8) with Vivado in
rem C:\myOpenTPU\openTPU-r7: tools\build_r7_133.ps1. Run r7_sync.cmd first.
rem   r7_build.cmd            full build (several hours); otpu.bit only if setup/hold pass
rem   r7_build.cmd Synth      synthesis + structural checks only (about an hour): try this first
rem   r7_build.cmd Prepare    the Vivado project only
rem NEVER programs the FPGA: loading the bitstream is a separate step you do (volatile JTAG).
setlocal
set MODE=%~1
if "%MODE%"=="" set MODE=Build
cd /d C:\myOpenTPU\openTPU-r7 || (echo no C:\myOpenTPU\openTPU-r7 & pause & exit /b 1)
findstr /c:"TGRQ" rtl\shell\otpu_ctrl.sv >nul || (echo the tree has no GRQ: run r7_sync.cmd & pause & exit /b 1)
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo a Vivado process is running: stop & pause & exit /b 1)
for /f %%h in ('git rev-parse --short=8 HEAD') do set REV=%%h
git log --oneline -1
set LOG=%~dp0r7_build_%MODE%_%REV%.log
echo === %MODE% of %REV%; log %LOG%; output build\vivado-r7-m8-133.33-%REV%
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& '.\tools\build_r7_133.ps1' -Mode %MODE% *>&1 | Tee-Object -FilePath '%LOG%'"
set RC=%errorlevel%
echo === exit %RC%
set OUTD=build\vivado-r7-m8-133.33-%REV%
if exist "%OUTD%\reports\SUMMARY.txt" type "%OUTD%\reports\SUMMARY.txt"
for /r "%OUTD%" %%f in (*.bit) do certutil -hashfile "%%f" SHA256
echo (nothing was loaded into the FPGA)
pause
exit /b %RC%
