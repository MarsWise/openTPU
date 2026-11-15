@echo off
rem Claude: full r8 build with the MXU fix of c9f51ceb (MM's k0 bit 28: an MM starting at scale block
rem 16..31 mod 32 read block k0 - 16's scales in every r7 build and r8 2c800a8f). New RTL: synthesis and
rem implementation again (Vivado 16 jobs / 16 threads, several hours). Updates C:\myOpenTPU\openTPU-r8 to
rem origin/feature/r7-fpga (fast-forward only), checks the fix is in the checkout, then r8_build.cmd
rem (otpu.bit only if setup / hold pass). Nothing is loaded into the FPGA.
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: close it first. & pause & exit /b 1)
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
git -C "%R%" merge-base --is-ancestor c9f51ceb HEAD || (echo HEAD is not c9f51ceb or later & goto :err)
findstr /c:"1'b0, cmd.w4[27:24]} : cmd.w4[31:24]" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no k0 fix: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
call "%R%\tools\nextgen\windows\r8_build.cmd"
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
