@echo off
rem Claude: read-only placement reports (what sits beside the DDR3 PHYs) of r5 45b1c1d3, r8 2c800a8f and
rem r8 + C2 c969ca4c, 1a77396a: opens each routed checkpoint in Vivado, dumps its placement to place_<name>\ here (tools/place_report.py reads it). About 5 minutes
rem each. Nothing is built, changed or loaded. Not while a Vivado build is running (memory).
setlocal
set "VIV=C:\Xilinx\Vivado\2024.2\bin\vivado.bat"
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it. & pause & exit /b 1)
call :one r5_45b1c1d3 "C:\myOpenTPU\openTPU.worktrees\pasted-text-processing\build\vivado-r5-m8-133.33-45b1c1d3-tgsc\otpu.runs\impl_1\otpu_fpga_top_ld_routed.dcp"
call :one r8_2c800a8f "C:\myOpenTPU\openTPU-r8\build\vivado-r8-m8-133.33-2c800a8f\reports\signoff.dcp"
call :one r8_c969ca4c "C:\myOpenTPU\openTPU-r8\build\vivado-r8-m8-133.33-c969ca4c\reports\signoff.dcp"
if exist "C:\myOpenTPU\openTPU-r8\build\vivado-r8-m8-133.33-1a77396a\reports\signoff.dcp" call :one r8_1a77396a "C:\myOpenTPU\openTPU-r8\build\vivado-r8-m8-133.33-1a77396a\reports\signoff.dcp"
echo.
dir /b /ad "%~dp0place_*"
pause
exit /b 0
:one
if not exist "%~2" (echo no checkpoint %~2 & exit /b 0)
set "D=%~2"
set "T=%~dp0place_%~1"
call "%VIV%" -mode batch -nojournal -log "%~dp0place_%~1.log" -source "%~dp0r8_place_report.tcl" -tclargs "%D:\=/%" "%T:\=/%"
findstr /c:"PLACE_DUMP" "%~dp0place_%~1.log" >nul && (echo %~1: ok) || (echo %~1: FAILED, see place_%~1.log)
exit /b 0
