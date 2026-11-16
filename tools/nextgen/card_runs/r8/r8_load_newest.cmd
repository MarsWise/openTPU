@echo off
rem Claude: load an r8 bitstream of C:\myOpenTPU\openTPU-r8\build into the FPGA over JTAG, volatile:
rem   r8_load_newest.cmd             the newest build that has an otpu.bit
rem   r8_load_newest.cmd 1a77396a    that build (the folder vivado-r8-m8-133.33-1a77396a)
rem   ... parallel                   as the last word: although a Vivado build is running
rem Shows the build's SUMMARY.txt, then r8_load.cmd asks "yes" before loading.
rem After it: r8_restore_pcie.cmd, then r8_all_tests.cmd.
setlocal
set "B=C:\myOpenTPU\openTPU-r8\build"
set "BIT="
set "WANT=%~1"
set "PAR=%~2"
if /i "%WANT%"=="parallel" (set "WANT=" & set "PAR=parallel")
if defined WANT (
  if exist "%B%\vivado-r8-m8-133.33-%WANT%\otpu.bit" (set "DIR=%B%\vivado-r8-m8-133.33-%WANT%") else (echo No otpu.bit in %B%\vivado-r8-m8-133.33-%WANT% : that build has not met timing or has not ended & pause & exit /b 1)
) else (
  for /f "delims=" %%d in ('dir /b /ad /o-d "%B%\vivado-r8-m8-133.33-*" 2^>nul') do if not defined DIR if exist "%B%\%%d\otpu.bit" set "DIR=%B%\%%d"
)
if not defined DIR (echo No otpu.bit under %B%\vivado-r8-m8-133.33-* & pause & exit /b 1)
set "BIT=%DIR%\otpu.bit"
echo === build: %DIR%
if exist "%DIR%\SOURCE_COMMIT.txt" type "%DIR%\SOURCE_COMMIT.txt"
if exist "%DIR%\reports\SUMMARY.txt" type "%DIR%\reports\SUMMARY.txt"
call "%~dp0r8_load.cmd" "%BIT%" %PAR%
exit /b %errorlevel%
