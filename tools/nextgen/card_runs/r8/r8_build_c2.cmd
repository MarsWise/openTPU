@echo off
rem Claude: full r8 build with GRQ QUAD (C2: Bonsai's GRQ chunks four quarters an advance; CAPS2 bit20,
rem CFG_HASH a905c3db) and the MXU k0 fix. New RTL: synthesis and implementation again (several hours).
rem Updates C:\myOpenTPU\openTPU-r8 to origin/feature/r7-fpga (fast-forward only), checks the checkout has
rem TGQ4, then r8_build.cmd (otpu.bit only if setup / hold pass). Nothing is loaded into the FPGA.
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it or close it first. & pause & exit /b 1)
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
findstr /c:"cmd_gq4" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no GRQ QUAD: wrong checkout & goto :err)
findstr /c:"TGQ4" "%R%\rtl\shell\otpu_ctrl.sv" >nul || (echo no TGQ4 capability: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
call "%R%\tools\nextgen\windows\r8_build.cmd"
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
