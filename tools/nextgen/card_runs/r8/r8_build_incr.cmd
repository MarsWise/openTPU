@echo off
rem Claude: r8 + C2 (the RTL of c969ca4c: MXU k0 fix, GRQ QUAD) built incrementally from 2c800a8f's routed
rem checkpoint: Vivado keeps 2c800a8f's placement and routing wherever the netlist is the same (the VPU,
rem TMEM, quant, memory logic) and places / routes the changed MXU logic. For the DDR3 read errors: the
rem VPU pblock builds broke timing (WNS -2.4 ns after placement) and crashed Vivado; this keeps the r8
rem placement that read the fewest errors. Run it only if 2c800a8f's DDR3 soak was clean (ПОСЛЕ_СБОРКИ.md).
rem Several hours; otpu.bit only if setup / hold pass. Nothing is loaded.
rem   r8_build_incr.cmd [parallel]
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
if "%OTPU_INCR_DCP%"=="" set "OTPU_INCR_DCP=C:\myOpenTPU\openTPU-r8\build\vivado-r8-m8-133.33-2c800a8f\reports\signoff.dcp"
if "%OTPU_INCR_DIRECTIVE%"=="" set OTPU_INCR_DIRECTIVE=TimingClosure
set OTPU_VPU_PBLOCK=
set OTPU_DDR_PHY_KEEPOUT=0
if not exist "%OTPU_INCR_DCP%" (echo No checkpoint %OTPU_INCR_DCP% & pause & exit /b 1)
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_incr.cmd parallel". & pause & exit /b 1))
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
findstr /c:"OTPU_INCR_DCP" "%R%\boards\ypcb-00338\vivado\build.tcl" >nul || (echo no OTPU_INCR_DCP in build.tcl: wrong checkout & goto :err)
findstr /c:"OTPU_INCR_DCP" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not pass OTPU_INCR_DCP: wrong checkout & goto :err)
findstr /c:"cmd_gq4" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no GRQ QUAD: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
echo === incremental from %OTPU_INCR_DCP% (%OTPU_INCR_DIRECTIVE%)
call "%R%\tools\nextgen\windows\r8_build.cmd"
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
