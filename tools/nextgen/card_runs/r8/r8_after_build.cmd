@echo off
rem Claude: everything after a new r8 build (r8_build_c3.cmd and the like), in order: about 2.5 h of DDR3
rem checks, then every card test (6-8 h), unattended after the two questions in 1 and 2:
rem   1 load the build over JTAG (asks yes)             r8_load_newest.cmd <build>
rem   2 PCIe back on the host (asks the sudo password)   r8_restore_pcie.cmd
rem   3 DDR3 diagnosis (written / read wrong, BIST lanes), ~25 min   r8_ddr_diag.cmd
rem   3b run_rows split by phase under loadcal ternary:4, 3 passes, ~10 min   r8_ddr_split.cmd
rem   4 DDR3 soak, 4 passes under loadcal ternary:4, ~1.7 h      r8_ddr_check.cmd
rem   5 the verdict: every ECC counter 0 / 0 in 3, 3b and 4 (r8_ddr_gate.ps1): written to gate.txt
rem   6 every card test (r8_all_tests.cmd without its DDR3 step): run whatever the verdict, the
rem     results then marked by it (a step with any ECC error is NOT TRUSTED)
rem Logs are copied to runs\after_<build>_<time>\. No rescan or load other than 1 and 2.
rem   r8_after_build.cmd              the newest build with an otpu.bit
rem   r8_after_build.cmd 58fecf08     that build
rem   r8_after_build.cmd 58fecf08 parallel   (load although a Vivado is running)
setlocal
set "B=C:\myOpenTPU\openTPU-r8\build"
set "WANT=%~1"
if not defined WANT for /f "delims=" %%d in ('dir /b /ad /o-d "%B%\vivado-r8-m8-133.33-*" 2^>nul') do if not defined WANT if exist "%B%\%%d\otpu.bit" set "WANT=%%d"
set "WANT=%WANT:vivado-r8-m8-133.33-=%"
if not exist "%B%\vivado-r8-m8-133.33-%WANT%\otpu.bit" (echo No otpu.bit for build "%WANT%": the build has not met timing or not ended. Send its reports\SUMMARY.txt. & pause & exit /b 1)
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%t"
set "OUT=%~dp0runs\after_%WANT%_%TS%"
mkdir "%OUT%" || (echo cannot create %OUT% & pause & exit /b 1)
echo === build %WANT%; logs: %OUT%
call "%~dp0r8_stop.cmd" check
if errorlevel 1 (echo The card is busy or the host is unreachable: wait, or r8_stop.cmd. & pause & exit /b 1)

echo.
echo ================ 1. load %WANT%
call "%~dp0r8_load_newest.cmd" %WANT% %2
if errorlevel 1 (echo NOT LOADED: stopping. Send r8_load.log. & pause & exit /b 1)
copy /y "%~dp0r8_load.log" "%OUT%\" >nul

echo.
echo ================ 2. PCIe
call "%~dp0r8_restore_pcie.cmd"
if errorlevel 1 (echo PCIe restore failed: stopping. & pause & exit /b 1)

echo.
echo ================ 3. r8_ddr_diag
call "%~dp0r8_ddr_diag.cmd" < nul
copy /y "%~dp0r8_ddr_diag.log" "%OUT%\" >nul

echo.
echo ================ 3b. r8_ddr_split (loadcal ternary:4, 3 passes)
set PASSES=3
set KINDS=ternary:4
call "%~dp0r8_ddr_split.cmd" < nul
copy /y "%~dp0r8_ddr_split.log" "%OUT%\" >nul

echo.
echo ================ 4. r8_ddr_check (4 passes, loadcal ternary:4)
set PASSES=4
set KINDS=ternary:4
call "%~dp0r8_ddr_check.cmd" < nul
copy /y "%~dp0r8_ddr_check.log" "%OUT%\" >nul

echo.
echo ================ 5. gate
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0r8_ddr_gate.ps1" > "%OUT%\gate.txt"
set GATE=%errorlevel%
type "%OUT%\gate.txt"
if not "%GATE%"=="0" (echo DRAM NOT CLEAN on %WANT%: the card tests run anyway, their results NOT TRUSTED where ECC moved) else (echo DRAM CLEAN on %WANT%)
echo.
echo ================ 6. every card test (6-8 h; do not close this window, do not press Ctrl+C)
set SKIP_DDR=1
set PASSES=4
set KINDS=ternary:4
call "%~dp0r8_all_tests.cmd"
echo === done. DDR3 verdict: & type "%OUT%\gate.txt" & echo Logs: %OUT% and runs\ (the newest folder of r8_all_tests)
pause
exit /b 0
