@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_r6_133.ps1" %*
set "exitcode=%errorlevel%"
echo.
echo r6 build exit code: %exitcode%
pause
exit /b %exitcode%
