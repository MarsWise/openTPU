@echo off
rem Claude: r5x lean + GRQ QUAD of 2 rows (TGQ4M) + GRQ QUAD drain overlap (GQ4_PIPE), incremental --
rem   r8_build_lean.cmd's build (CFG_HASH 81dc3958) implemented incrementally from the routed checkpoint
rem   of ed482f26-lean (the firmware on the card: timing met at WNS +0.019): its placement of everything the
rem   two MXU changes leave alone is reused; the MXU's command decode / pop control is placed anew. For when
rem   the full build (r8_build_lean.cmd) misses timing: its route stood at WNS -0.215 ns on paths outside the
rem   change (TMEM's pumped BRAM addresses, DMA, VPU), a placement's luck. OTPU_INCR_DCP / OTPU_INCR_DIRECTIVE
rem   override. Output build\vivado-r8-m8-133.33-<rev>-lean-i2; otpu.bit only if setup / hold pass. Nothing
rem   is loaded. Do not start it while another Vivado runs (it checks).
rem   usage: r8_build_lean_incr2.cmd [parallel | plan | impl]
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
if "%OTPU_INCR_DCP%"=="" set "OTPU_INCR_DCP=C:\myOpenTPU\openTPU-r8\build\vivado-r8-m8-133.33-ed482f26-lean\otpu.runs\impl_1\otpu_fpga_top_ld_routed.dcp"
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
set OTPU_BUILD_TAG=lean-i2
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_lean_incr2.cmd parallel". & pause & exit /b 1))
rem impl reuses this checkout's synthesis: the checkout stays at its commit (no update)
if /i "%~1"=="impl" goto :checks
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
:checks
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
findstr /c:"parameter bit PIPE = 1'b1" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo otpu_tmm_gsc has no PIPE: wrong checkout & goto :err)
findstr /c:"GQ4_PIPE" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no GRQ QUAD drain overlap: wrong checkout & goto :err)
findstr /c:"gq4_run" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no GRQ QUAD of 2 rows: wrong checkout & goto :err)
findstr /c:".HAS_TATAN(TATAN)) u_slice" "%R%\rtl\shell\otpu_board.sv" >nul || (echo the board does not pass TATAN to the slices: wrong checkout & goto :err)
findstr /c:"OREG" "%R%\rtl\tech\generic\otpu_tech_ram.sv" >nul && (echo the checkout has the FIFO output register: not r5x's pipeline & goto :err)
findstr /c:"OTPU_BUILD_TAG" "%R%\tools\build_r8_133.ps1" >nul || (echo build_r8_133.ps1 takes no OTPU_BUILD_TAG: wrong checkout & goto :err)
findstr /c:"OTPU_DSP_PROHIBIT_COLS" "%R%\boards\ypcb-00338\constraints\otpu_mxu_place.tcl" >nul || (echo otpu_mxu_place.tcl takes no OTPU_DSP_PROHIBIT_COLS: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
echo === expected: hash=81dc3958 (r8 less TBIN TBGSC TD5 TATAN, with TGQ4M)
echo === r5x lean: less BIN / BIN GSC / T5 / atan, TFN ROM a lane pair, one RAND generator
if /i "%~1"=="plan" (call "%R%\tools\nextgen\windows\r8_build.cmd" Plan) else if /i "%~1"=="impl" (call "%R%\tools\nextgen\windows\r8_build.cmd" Impl) else (call "%R%\tools\nextgen\windows\r8_build.cmd")
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
