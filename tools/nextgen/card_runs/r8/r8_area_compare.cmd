@echo off
rem Claude: the area of a new build against C3 (6dce3d19), from the builds' reports\util_hier.rpt
rem (tools\util_compare.py: the design, the slice's units, the MXU's own logic "(u_mxu)" where the
rem weight decode and the CQ LUT RAMs sit), and both builds' timing (reports\SUMMARY.txt). Reads
rem files only: no Vivado, nothing loaded. Output also in r8_area_<build>.txt here.
rem   r8_area_compare.cmd             the newest build with an otpu.bit against C3
rem   r8_area_compare.cmd 508da1c6    that build
rem   r8_area_compare.cmd 508da1c6 2c800a8f   against another build
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
set "B=%R%\build"
set "WANT=%~1"
set "BASE=%~2"
if not defined BASE set "BASE=6dce3d19"
if not defined WANT for /f "delims=" %%d in ('dir /b /ad /o-d "%B%\vivado-r8-m8-133.33-*" 2^>nul') do if not defined WANT if exist "%B%\%%d\reports\util_hier.rpt" set "WANT=%%d"
set "WANT=%WANT:vivado-r8-m8-133.33-=%"
if not exist "%B%\vivado-r8-m8-133.33-%WANT%\reports\util_hier.rpt" (echo No reports\util_hier.rpt for build "%WANT%": the build has not ended. & pause & exit /b 1)
rem (util_compare.py beside this script: the build checkout stays as the build left it)
set "UC=%~dp0util_compare.py"
if not exist "%UC%" set "UC=%R%\tools\util_compare.py"
if not exist "%UC%" (echo no util_compare.py beside this script or in %R%\tools & pause & exit /b 1)
set "OUT=%~dp0r8_area_%WANT%.txt"
(
echo === area: %BASE% -^> %WANT%
python "%UC%" "%B%\vivado-r8-m8-133.33-%BASE%" "%B%\vivado-r8-m8-133.33-%WANT%"
echo.
echo === timing %BASE%
type "%B%\vivado-r8-m8-133.33-%BASE%\reports\SUMMARY.txt"
echo.
echo === timing %WANT%
type "%B%\vivado-r8-m8-133.33-%WANT%\reports\SUMMARY.txt"
) > "%OUT%" 2>&1
type "%OUT%"
echo.
echo (written to %OUT%; nothing was loaded)
pause
