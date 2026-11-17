@echo off
rem Claude: r8 + C2 (same RTL as c969ca4c, CFG_HASH a905c3db) built again for the DDR3 read errors under
rem the core's load. r5 45b1c1d3 reads clean on the same card; the placement dumps (r8_place_report,
rem 2026-11-17) show r8's VPU beside channel 0's PHY: 63 000 of its 105 000 cells in clock regions
rem X0Y0 / X0Y1 (r5: 17 000), where channel 0 lane 0 reads wrong. So the VPU goes into a hard pblock
rem away from both PHYs' regions (OTPU_VPU_PBLOCK, default X1Y0..X1Y3 and X0Y3..X0Y4), with a 0.10 ns
rem setup margin on LiteDRAM's sys clock while placing and routing (removed for the bitstream).
rem OTPU_DDR_PHY_KEEPOUT=1 adds the 4-column keep-out beside each PHY (off by default here).
rem Several hours; otpu.bit only if setup / hold pass. Nothing is loaded.
rem   r8_build_ddr.cmd [parallel]
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
if "%OTPU_SYS_MARGIN%"=="" set OTPU_SYS_MARGIN=0.10
if "%OTPU_DDR_PHY_KEEPOUT%"=="" set OTPU_DDR_PHY_KEEPOUT=0
if "%OTPU_VPU_PBLOCK%"=="" set "OTPU_VPU_PBLOCK=CLOCKREGION_X1Y0:CLOCKREGION_X1Y3 CLOCKREGION_X0Y3:CLOCKREGION_X0Y4"
if "%OTPU_DDR_PHY_COLS%"=="" set OTPU_DDR_PHY_COLS=4
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_ddr.cmd parallel". & pause & exit /b 1))
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
findstr /c:"OTPU_WR_MARGIN" "%R%\boards\ypcb-00338\vivado\margins.tcl" >nul || (echo no OTPU_WR_MARGIN in margins.tcl: wrong checkout & goto :err)
findstr /c:"OTPU_VPU_PBLOCK" "%R%\boards\ypcb-00338\vivado\create_project.tcl" >nul || (echo no OTPU_VPU_PBLOCK in create_project.tcl: wrong checkout & goto :err)
findstr /c:"OTPU_VPU_PBLOCK" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not pass the VPU pblock: wrong checkout & goto :err)
findstr /c:"OTPU_DDR_PHY_KEEPOUT" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not pass the PHY keep-out: wrong checkout & goto :err)
findstr /c:"OTPU_DDR_PHY_KEEPOUT" "%R%\boards\ypcb-00338\constraints\otpu_ddr_phy_place.tcl" >nul || (echo no PHY keep-out in otpu_ddr_phy_place.tcl: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
echo === VPU pblock {%OTPU_VPU_PBLOCK%}, PHY keep-out %OTPU_DDR_PHY_KEEPOUT% (%OTPU_DDR_PHY_COLS% columns), sys margin %OTPU_SYS_MARGIN% ns, wr margin %OTPU_WR_MARGIN% ns
call "%R%\tools\nextgen\windows\r8_build.cmd"
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
