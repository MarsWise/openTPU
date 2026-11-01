@echo off
rem Claude: build r8 (m8, 133.33 MHz; r7 + VRAND, TCQ, TATAN, QST4, LDX, TD5) with Vivado in
rem C:\myOpenTPU\openTPU-r7: tools\build_r8_133.ps1. Run r7_sync.cmd first.
rem   r8_build.cmd            full build (several hours); otpu.bit only if setup/hold pass
rem   r8_build.cmd Synth      synthesis + structural checks only (about an hour): try this first
rem   r8_build.cmd Prepare    the Vivado project only
rem NEVER programs the FPGA: loading the bitstream is a separate step you do (volatile JTAG).
setlocal
set MODE=%~1
if "%MODE%"=="" set MODE=Build
cd /d C:\myOpenTPU\openTPU-r7 || (echo no C:\myOpenTPU\openTPU-r7 & pause & exit /b 1)
findstr /c:"TD5" rtl\shell\otpu_ctrl.sv >nul || (echo the tree has no r8 units: run r7_sync.cmd & pause & exit /b 1)
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo a Vivado process is running: stop & pause & exit /b 1)
for /f %%h in ('git rev-parse --short^=8 HEAD') do set REV=%%h
git log --oneline -1
set LOG=%~dp0r8_build_%MODE%_%REV%.log
echo === %MODE% of %REV%; log %LOG%; output build\vivado-r8-m8-133.33-%REV%
echo === the python the build uses (CFG_HASH):
where python
python -c "import sys, numpy; print(sys.executable, sys.version.split()[0], 'numpy', numpy.__version__)"
if errorlevel 1 goto nopy
python -c "import sys,dataclasses; sys.path.insert(0,'.'); from tools.bonsai2_validate import configuration; from opentpu.target import arch_hash; print('CFG_HASH check', format(arch_hash(dataclasses.replace(configuration(), TGSC=True, VCMP=True, TBIN=True, TBGSC=True, VU8=True, TGRQ=True, VRAND=True, TCQ=True, TATAN=True, QST4=True, LDX=True, TD5=True)), '08x'))"
if errorlevel 1 goto nohash
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { & '.\tools\build_r8_133.ps1' -Mode %MODE% } *>&1 | ForEach-Object { $_; Add-Content -LiteralPath '%LOG%' -Value \"$_\" -Encoding UTF8 }; exit $LASTEXITCODE"
set RC=%errorlevel%
echo === exit %RC%
set OUTD=build\vivado-r8-m8-133.33-%REV%
if exist "%OUTD%\reports\SUMMARY.txt" type "%OUTD%\reports\SUMMARY.txt"
for /r "%OUTD%" %%f in (*.bit) do certutil -hashfile "%%f" SHA256
echo (nothing was loaded into the FPGA)
pause
exit /b %RC%
:nopy
echo The python on PATH cannot import numpy: the build computes CFG_HASH with it.
echo Install it for that python: python -m pip install numpy   then run r8_build.cmd again.
pause
exit /b 1
:nohash
echo The CFG_HASH step failed: its traceback is above. Send it to Claude.
pause
exit /b 1
