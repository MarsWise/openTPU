@echo off
setlocal
cd /d "%~dp0"
echo Experimental r7: hardware-only model coverage is not yet complete.
echo This builds a project and bitstream but does not program the FPGA.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_r7_133.ps1" %*
set "exitcode=%errorlevel%"
echo.
echo r7 build exit code: %exitcode%
pause
exit /b %exitcode%
