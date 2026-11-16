@echo off
rem Claude: load the NEWEST r8 bitstream of C:\myOpenTPU\openTPU-r8\build (the last build that met timing)
rem into the FPGA over JTAG, volatile. Shows which build it is, then r8_load.cmd asks "yes" before loading.
rem After it: r8_restore_pcie.cmd, then r8_all_tests.cmd. Not while a Vivado build is running.
setlocal
set "B=C:\myOpenTPU\openTPU-r8\build"
set "BIT="
for /f "delims=" %%d in ('dir /b /ad /o-d "%B%\vivado-r8-m8-133.33-*" 2^>nul') do if not defined BIT if exist "%B%\%%d\otpu.bit" set "BIT=%B%\%%d\otpu.bit" & set "DIR=%B%\%%d"
if not defined BIT (echo No otpu.bit under %B%\vivado-r8-m8-133.33-* & pause & exit /b 1)
echo === newest build with a bitstream: %DIR%
if exist "%DIR%\SOURCE_COMMIT.txt" type "%DIR%\SOURCE_COMMIT.txt"
if exist "%DIR%\reports\SUMMARY.txt" type "%DIR%\reports\SUMMARY.txt"
call "%~dp0r8_load.cmd" "%BIT%"
exit /b %errorlevel%
