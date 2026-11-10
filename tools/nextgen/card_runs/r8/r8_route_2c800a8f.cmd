@echo off
rem Claude: finish the 2c800a8f build. Its impl_1 met timing at route (WNS +0.030, WHS +0.013
rem with the margins) but Vivado exited while writing the routed checkpoint: no otpu.bit.
rem This routes again from impl_1's saved physopt checkpoint (about 3 hours), removes the margins,
rem writes reports\SUMMARY.txt and otpu.bit (only if setup / hold pass) BEFORE any checkpoint.
rem Updates the checkout first (fast-forward: host code + tools\vivado\r8_route_resume.tcl; the RTL
rem and the build folder are untouched). Nothing is loaded into the FPGA.
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
set "OUT=%R%\build\vivado-r8-m8-133.33-2c800a8f"
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: close it first. & pause & exit /b 1)
if not exist "%OUT%\otpu.runs\impl_1\otpu_fpga_top_ld_physopt.dcp" (echo No physopt checkpoint in %OUT% & pause & exit /b 1)
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
git -C "%R%" diff --quiet 2c800a8f HEAD -- rtl boards || (echo RTL or board files differ from 2c800a8f: stop & goto :err)
if not exist "%R%\tools\vivado\r8_route_resume.tcl" (echo no r8_route_resume.tcl & goto :err)
git -C "%R%" log --oneline -1
if exist "%OUT%\otpu.bit" ren "%OUT%\otpu.bit" otpu_old.bit
call C:\Xilinx\Vivado\2024.2\bin\vivado.bat -mode batch -nojournal -log "%OUT%\route_resume.log" -source "%R%\tools\vivado\r8_route_resume.tcl" -tclargs "%OUT%" 12
echo === exit %errorlevel%
if exist "%OUT%\reports\SUMMARY.txt" type "%OUT%\reports\SUMMARY.txt"
if exist "%OUT%\otpu.bit" certutil -hashfile "%OUT%\otpu.bit" SHA256
echo (nothing was loaded into the FPGA)
pause
exit /b 0
:err
echo failed
pause
exit /b 1
