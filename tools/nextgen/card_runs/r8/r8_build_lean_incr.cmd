@echo off
rem Claude: r5x lean, incremental -- r8_build_lean.cmd's build implemented incrementally from
rem   r5x's routed checkpoint (c42beb65, timing met at WNS +0.002): the MXU's netlist is r5x's, so
rem   its placement (the weight FIFO -> weight decode path, -0.124 ns in the plain lean build) is
rem   reused; the VPU's changes are placed anew. OTPU_INCR_DCP / OTPU_INCR_DIRECTIVE override.
rem   ---- r8_build_lean.cmd: r5x lean -- r5x's build (r5's pipeline, WREG 0, rtw 3, trace 16384, margins 0.10 /
rem   0.10 / 0.05, no BIN / BIN GSC) from feature/r7-fpga with four area changes: TMM T5 off
rem   (otpu_mxu HAS_TD5 0: its trit decode leaves the weight-decode path; CAPS2 bit15 clear),
rem   TFN atan off (otpu_se_comp HAS_ATAN 0, CAPS2 bit12 clear, VOP atan refused by the host),
rem   one TFN coefficient ROM for each two VPU lanes (8 copies -> 4), and one VOP RAND generator
rem   (otpu_vpu RAND_W 1: RAND one column a cycle, the same numbers). CFG_HASH add73610.
rem   Output build\vivado-r8-m8-133.33-<rev>-lean. A full build, several hours; otpu.bit only if
rem   setup / hold pass (r5x closed at WNS +0.002: a rebuild may not). Nothing is loaded.
rem   "r8_build_lean.cmd plan" first: no Vivado, prints the hash.
rem   ---- usage: r8_build_lean_incr.cmd [parallel | plan | impl]  (impl: implementation only, reusing the finished synthesis of the same checkout, after a crash past synthesis)
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
if "%OTPU_INCR_DCP%"=="" set "OTPU_INCR_DCP=C:\myOpenTPU\openTPU-r5x\build\vivado-r8-m8-133.33-c42beb65\otpu.runs\impl_1\otpu_fpga_top_ld_routed.dcp"
if "%OTPU_INCR_DIRECTIVE%"=="" set OTPU_INCR_DIRECTIVE=TimingClosure
if not exist "%OTPU_INCR_DCP%" (echo No checkpoint %OTPU_INCR_DCP% & pause & exit /b 1)
echo === incremental from %OTPU_INCR_DCP% (%OTPU_INCR_DIRECTIVE%)
set OTPU_VPU_PBLOCK=
set OTPU_DDR_PHY_KEEPOUT=0
set OTPU_LITEDRAM_RTW=3
set TRACE_DEPTH=16384
set OTPU_UC2_MARGIN=0.10
set OTPU_SYS_MARGIN=0.10
set OTPU_WR_MARGIN=0.05
set "OTPU_DROP_FEATURES=TBIN TBGSC TD5 TATAN"
set OTPU_MEMCH_WREG=0
set OTPU_MXU_DSP_COLS=
set OTPU_MXU_BRAM_COLS=
set OTPU_MXU_WRAM_COL=
set OTPU_DSP_PROHIBIT_COLS=
set OTPU_BUILD_TAG=lean-i
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_lean_incr.cmd parallel". & pause & exit /b 1))
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
findstr /c:"begin : g_rom" "%R%\rtl\vpu\otpu_se_comp.sv" >nul || (echo otpu_se_comp has no shared TFN ROM: wrong checkout & goto :err)
findstr /c:"parameter int RAND_W = 1" "%R%\rtl\vpu\otpu_vpu.sv" >nul || (echo otpu_vpu has no RAND_W: wrong checkout & goto :err)
findstr /c:".HAS_TATAN(TATAN)) u_slice" "%R%\rtl\shell\otpu_board.sv" >nul || (echo the board does not pass TATAN to the slices: wrong checkout & goto :err)
findstr /c:"OREG" "%R%\rtl\tech\generic\otpu_tech_ram.sv" >nul && (echo the checkout has the FIFO output register: not r5x's pipeline & goto :err)
findstr /c:"OTPU_BUILD_TAG" "%R%\tools\build_r8_133.ps1" >nul || (echo build_r8_133.ps1 takes no OTPU_BUILD_TAG: wrong checkout & goto :err)
findstr /c:"OTPU_DSP_PROHIBIT_COLS" "%R%\boards\ypcb-00338\constraints\otpu_mxu_place.tcl" >nul || (echo otpu_mxu_place.tcl takes no OTPU_DSP_PROHIBIT_COLS: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
echo === expected: hash=add73610 (r8 less TBIN TBGSC TD5 TATAN)
echo === r5x lean: less BIN / BIN GSC / T5 / atan, TFN ROM a lane pair, one RAND generator
if /i "%~1"=="plan" (call "%R%\tools\nextgen\windows\r8_build.cmd" Plan) else if /i "%~1"=="impl" (call "%R%\tools\nextgen\windows\r8_build.cmd" Impl) else (call "%R%\tools\nextgen\windows\r8_build.cmd")
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
