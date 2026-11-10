@echo off
rem Claude: full r8 build of 2c800a8f (5e6293e6 MXU BRAM output register + the PAIR select fix 63cbadcd; Vivado 16 jobs /
rem 16 threads). Updates C:\myOpenTPU\openTPU-r8 to origin/feature/r7-fpga (fast-forward only),
rem checks HEAD is 2c800a8f or later, then runs r8_build.cmd (synthesis + implementation; otpu.bit
rem only if setup/hold pass). New RTL: the old synthesis cannot be reused. Nothing is loaded.
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: close it first. & pause & exit /b 1)
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
git -C "%R%" merge-base --is-ancestor 2c800a8f HEAD || (echo HEAD is not 2c800a8f or later & goto :err)
findstr /c:"pe_qs1" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no OREG: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
call "%R%\tools\nextgen\windows\r8_build.cmd"
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
