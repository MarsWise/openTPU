@echo off
rem Claude: experiment A -- "r5 plus the new instructions": r8's RTL (VCMP, VU8, GRQ, VRAND, CQ,
rem   TATAN, QST4, LDX, T5, BFLY, AFF, QST6, CSWP, GRQ QUAD; no BIN / BIN GSC) with r5's memory path:
rem   otpu_mem_ch without r8's write-data register (OTPU_MEMCH_WREG=0, as 45b1c1d3), LiteDRAM's
rem   read-to-write as r5 (rtw 3), the trace as r5 (16384). The P&R margins stay (they change no
rem   logic). If its DDR3 is clean where C3 / C4 are not, the write register is the cause.
rem   CFG_HASH 432e34c2. Output build\vivado-r8-m8-133.33-<rev>-a. A full build, several hours;
rem   otpu.bit only if setup / hold pass (the register was added for a -1.20 ns path: it may not
rem   close). Nothing is loaded. "r8_build_a.cmd plan" first: no Vivado, prints the hash.
rem   r8_build_a.cmd [parallel | plan]
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
set OTPU_INCR_DCP=
set OTPU_INCR_DIRECTIVE=
set OTPU_VPU_PBLOCK=
set OTPU_DDR_PHY_KEEPOUT=0
set OTPU_LITEDRAM_RTW=3
set TRACE_DEPTH=16384
set OTPU_UC2_MARGIN=0.10
set OTPU_SYS_MARGIN=0.10
set OTPU_WR_MARGIN=0.05
set "OTPU_DROP_FEATURES=TBIN TBGSC"
set OTPU_MEMCH_WREG=0
set OTPU_MXU_DSP_COLS=
set OTPU_MXU_BRAM_COLS=
set OTPU_MXU_WRAM_COL=
set OTPU_DSP_PROHIBIT_COLS=
set OTPU_BUILD_TAG=a
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_a.cmd parallel". & pause & exit /b 1))
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
echo === expected: hash=432e34c2 (r8 less TBIN TBGSC)
echo === A: r8 RTL less BIN, r5's memory path (no write register, rtw 3), trace 16384
if /i "%~1"=="plan" (call "%R%\tools\nextgen\windows\r8_build.cmd" Plan) else (call "%R%\tools\nextgen\windows\r8_build.cmd")
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
