@echo off
rem Claude: r5x less the TFN atan table -- r5x's build (r5's pipeline, WREG 0, rtw 3, trace
rem   16384, margins 0.10 / 0.10 / 0.05, no BIN / BIN GSC) from feature/r7-fpga, with TATAN off:
rem   otpu_se_comp HAS_ATAN 0 (each lane's TFN ROM keeps tables 0-7, 2048 entries instead of
rem   2304), CAPS2 bit12 clear, the host refuses VOP atan (no model uses it). CFG_HASH 852be9dc.
rem   Output build\vivado-r8-m8-133.33-<rev>-noatan. A full build, several hours; otpu.bit only
rem   if setup / hold pass (r5x closed at WNS +0.002: a rebuild may not). Nothing is loaded.
rem   "r8_build_noatan.cmd plan" first: no Vivado, prints the hash.
rem   r8_build_noatan.cmd [parallel | plan]
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
set "OTPU_DROP_FEATURES=TBIN TBGSC TATAN"
set OTPU_MEMCH_WREG=0
set OTPU_MXU_DSP_COLS=
set OTPU_MXU_BRAM_COLS=
set OTPU_MXU_WRAM_COL=
set OTPU_DSP_PROHIBIT_COLS=
set OTPU_BUILD_TAG=noatan
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_noatan.cmd parallel". & pause & exit /b 1))
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
findstr /c:"HAS_ATAN" "%R%\rtl\vpu\otpu_se_comp.sv" >nul || (echo otpu_se_comp has no HAS_ATAN gate: wrong checkout & goto :err)
findstr /c:".HAS_TATAN(TATAN)) u_slice" "%R%\rtl\shell\otpu_board.sv" >nul || (echo the board does not pass TATAN to the slices: wrong checkout & goto :err)
findstr /c:"OREG" "%R%\rtl\tech\generic\otpu_tech_ram.sv" >nul && (echo the checkout has the FIFO output register: not r5x's pipeline & goto :err)
findstr /c:"OTPU_BUILD_TAG" "%R%\tools\build_r8_133.ps1" >nul || (echo build_r8_133.ps1 takes no OTPU_BUILD_TAG: wrong checkout & goto :err)
findstr /c:"OTPU_DSP_PROHIBIT_COLS" "%R%\boards\ypcb-00338\constraints\otpu_mxu_place.tcl" >nul || (echo otpu_mxu_place.tcl takes no OTPU_DSP_PROHIBIT_COLS: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
echo === expected: hash=852be9dc (r8 less TBIN TBGSC TATAN)
echo === r5x less atan: r5's pipeline + the r8 instructions less BIN / BIN GSC / atan
if /i "%~1"=="plan" (call "%R%\tools\nextgen\windows\r8_build.cmd" Plan) else (call "%R%\tools\nextgen\windows\r8_build.cmd")
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
