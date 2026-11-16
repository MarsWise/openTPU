@echo off
rem Claude: sign off a build whose route ended a few tens of ps short, without building again (about an hour):
rem   r8_signoff_resume.cmd            the 1a77396a build
rem   r8_signoff_resume.cmd <folder>   another build folder under C:\myOpenTPU\openTPU-r8\build
rem Opens the routed design, removes the setup margins, post-route phys_opt + re-route at the real constraints
rem (each pass kept only if better), SUMMARY.txt, otpu.bit only if setup and hold pass. Only AFTER the build
rem has ended without an otpu.bit. Updates the checkout first (fast-forward: scripts only). Nothing is loaded.
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
set "N=%~1"
if "%N%"=="" set "N=vivado-r8-m8-133.33-1a77396a"
set "OUT=%R%\build\%N%"
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for the build to end first. & pause & exit /b 1)
if exist "%OUT%\otpu.bit" (echo %OUT%\otpu.bit exists: the build met timing, nothing to do. & pause & exit /b 0)
if not exist "%OUT%\otpu.runs\impl_1\otpu_fpga_top_ld_routed.dcp" if not exist "%OUT%\reports\stages\routed\design.dcp" (echo No routed checkpoint in %OUT% & pause & exit /b 1)
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
if not exist "%R%\tools\vivado\r8_signoff_resume.tcl" (echo no r8_signoff_resume.tcl: checkout not updated & goto :err)
git -C "%R%" log --oneline -1
call C:\Xilinx\Vivado\2024.2\bin\vivado.bat -mode batch -nojournal -log "%OUT%\signoff_resume.log" -source "%R%\tools\vivado\r8_signoff_resume.tcl" -tclargs "%OUT%" 12
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
