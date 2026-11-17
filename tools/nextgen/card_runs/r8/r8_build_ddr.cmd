@echo off
rem Claude: r8 + C2 (same RTL as c969ca4c, CFG_HASH a905c3db) built again for the DDR3 read errors under
rem the core's load (ddr_diag wr 2026-11-17: the host's data stored and read clean, the core's reads wrong
rem under int8 / ternary load: the read capture, not the write): the PHY keep-out (OTPU_DDR_PHY_KEEPOUT=1:
rem the PHY's first read / write flops in hard pblocks of OTPU_DDR_PHY_COLS slice columns beside each PHY,
rem EXCLUDE_PLACEMENT: no MXU / TMEM / VPU logic there) and a 0.10 ns setup margin on LiteDRAM's sys clock
rem while placing and routing (removed for the bitstream). Several hours; otpu.bit only if setup / hold pass.
rem Run it only if loadcal with +-2 bitslips (r8_ddr_diag.cmd on c969ca4c) leaves errors. Nothing is loaded.
rem   r8_build_ddr.cmd [parallel]      other values: set OTPU_DDR_PHY_COLS / OTPU_SYS_MARGIN / OTPU_WR_MARGIN before it
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
if "%OTPU_SYS_MARGIN%"=="" set OTPU_SYS_MARGIN=0.10
if "%OTPU_DDR_PHY_KEEPOUT%"=="" set OTPU_DDR_PHY_KEEPOUT=1
if "%OTPU_DDR_PHY_COLS%"=="" set OTPU_DDR_PHY_COLS=4
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_ddr.cmd parallel". & pause & exit /b 1))
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
findstr /c:"OTPU_WR_MARGIN" "%R%\boards\ypcb-00338\vivado\margins.tcl" >nul || (echo no OTPU_WR_MARGIN in margins.tcl: wrong checkout & goto :err)
findstr /c:"OTPU_DDR_PHY_KEEPOUT" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not pass the PHY keep-out: wrong checkout & goto :err)
findstr /c:"OTPU_DDR_PHY_KEEPOUT" "%R%\boards\ypcb-00338\constraints\otpu_ddr_phy_place.tcl" >nul || (echo no PHY keep-out in otpu_ddr_phy_place.tcl: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
echo === PHY keep-out %OTPU_DDR_PHY_KEEPOUT% (%OTPU_DDR_PHY_COLS% columns), sys margin %OTPU_SYS_MARGIN% ns, wr margin %OTPU_WR_MARGIN% ns
call "%R%\tools\nextgen\windows\r8_build.cmd"
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
