@echo off
rem Claude: experiment B -- C4 (r8 less T5 / BIN / BIN GSC, rtw 4, trace 4096) with the MXU moved:
rem   its array on DSP48 X1..X8 / block RAM X1..X8, the weight RAMs in X0 (otpu_mxu_place.tcl's
rem   table; r5 and r8 use X3..X10 / X2), and DSP48 X0 closed to every other DSP
rem   (OTPU_DSP_PROHIBIT_COLS=0): the VPU's, QUANT's and the epilogues' DSPs, with the logic around
rem   them, go to X9..X11, away from the DDR3 PHYs' I/O column. If its DDR3 is cleaner than C4's,
rem   the placement is the cause. CFG_HASH 37f51889 (as C4). Output build\vivado-r8-m8-133.33-<rev>-b.
rem   A full build, several hours; otpu.bit only if setup / hold pass. Nothing is loaded.
rem   "r8_build_b.cmd plan" first: no Vivado.
rem   r8_build_b.cmd [parallel | plan]
rem CLOSED 2026-11-19, never built: r5x (c42beb65, now the feature branch's RTL) has r5's DDR3
rem   behaviour on the card with the MXU where r5 had it: the cause lies with r8's three timing commits
rem   (5e6293e6, 63cbadcd, 800ae9d0) or the placement they bring. The MXU columns (OTPU_MXU_*_COLS, OTPU_DSP_PROHIBIT_COLS) stay as knobs.
echo Experiment B is closed (r5x settled the DDR3 question): nothing built.
pause
exit /b 0
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
set OTPU_INCR_DCP=
set OTPU_INCR_DIRECTIVE=
set OTPU_VPU_PBLOCK=
set OTPU_DDR_PHY_KEEPOUT=0
set OTPU_LITEDRAM_RTW=4
set TRACE_DEPTH=4096
set OTPU_UC2_MARGIN=0.10
set OTPU_SYS_MARGIN=0.10
set OTPU_WR_MARGIN=0.05
set "OTPU_DROP_FEATURES=TD5 TBIN TBGSC"
set OTPU_MEMCH_WREG=1
set "OTPU_MXU_DSP_COLS=1 2 3 4 5 6 7 8"
set "OTPU_MXU_BRAM_COLS=1 2 3 4 5 6 7 8"
set OTPU_MXU_WRAM_COL=0
set OTPU_DSP_PROHIBIT_COLS=0
set OTPU_BUILD_TAG=b
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_b.cmd parallel". & pause & exit /b 1))
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
findstr /c:"OTPU_UC2_MARGIN" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not pass OTPU_UC2_MARGIN: wrong checkout & goto :err)
findstr /c:"lw_we_q" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no replicated LUT write registers: wrong checkout & goto :err)
findstr /c:"cmd_gq4" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no GRQ QUAD: wrong checkout & goto :err)
findstr /c:"HAS_T5" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no format parameters: wrong checkout & goto :err)
findstr /c:"OTPU_DROP_FEATURES" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not take OTPU_DROP_FEATURES: wrong checkout & goto :err)
findstr /c:"OTPU_LITEDRAM_RTW" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not pass OTPU_LITEDRAM_RTW: wrong checkout & goto :err)
if not exist "%R%\boards\ypcb-00338\vivado\litedram_rtw.tcl" (echo no litedram_rtw.tcl: wrong checkout & goto :err)
findstr /c:"WREG" "%R%\rtl\shell\otpu_mem_ch.sv" >nul || (echo otpu_mem_ch has no WREG: wrong checkout & goto :err)
findstr /c:"OTPU_BUILD_TAG" "%R%\tools\build_r8_133.ps1" >nul || (echo build_r8_133.ps1 takes no OTPU_BUILD_TAG: wrong checkout & goto :err)
findstr /c:"OTPU_DSP_PROHIBIT_COLS" "%R%\boards\ypcb-00338\constraints\otpu_mxu_place.tcl" >nul || (echo otpu_mxu_place.tcl takes no OTPU_DSP_PROHIBIT_COLS: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
echo === expected: hash=37f51889 (as C4)
echo === B: C4 with the MXU on DSP X1..X8 / RAM X1..X8 + X0, DSP X0 closed to the rest
if /i "%~1"=="plan" (call "%R%\tools\nextgen\windows\r8_build.cmd" Plan) else (call "%R%\tools\nextgen\windows\r8_build.cmd")
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
