@echo off
rem Claude: r8 C3 -- r8 + C2 with the same function, built for timing and DDR3 (a full build, several hours):
rem   the LiteDRAM read-to-write turnaround 4 cycles (OTPU_LITEDRAM_RTW=4, litedram_rtw.tcl: the DDR3 read
rem   errors where stores interleave with the MXUs weight reads, r8_ddr_split 2026-11-17);
rem   the trace buffer 4096 records (TRACE_DEPTH, was 16384: -24 BRAM36; profiling stays);
rem   the TMM GSC adapters LUT write registers replicated (max_fanout 128, 1,500-1,800 loads);
rem   placement / route margins on the clocks a1ba144b left short at sign-off: userclk2 0.10 ns
rem   (OTPU_UC2_MARGIN: its one failing XDMA path), LiteDRAM sys 0.10 (OTPU_SYS_MARGIN: +0.009 there),
rem   writes 0.05 (OTPU_WR_MARGIN), on top of the core 0.15 / x2c 0.45 the recipe has; all removed for
rem   the bitstream. CFG_HASH unchanged (a905c3db). otpu.bit only if setup / hold pass. Nothing is loaded.
rem   r8_build_c3.cmd [parallel]
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
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_c3.cmd parallel". & pause & exit /b 1))
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
findstr /c:"OTPU_UC2_MARGIN" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not pass OTPU_UC2_MARGIN: wrong checkout & goto :err)
findstr /c:"lw_we_q" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no replicated LUT write registers: wrong checkout & goto :err)
findstr /c:"cmd_gq4" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no GRQ QUAD: wrong checkout & goto :err)
findstr /c:"OTPU_LITEDRAM_RTW" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not pass OTPU_LITEDRAM_RTW: wrong checkout & goto :err)
if not exist "%R%\boards\ypcb-00338\vivado\litedram_rtw.tcl" (echo no litedram_rtw.tcl: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
echo === C3: LiteDRAM rtw 4, TRACE_DEPTH 4096, margins uc2 0.10 / sys 0.10 / wr 0.05 (full build)
call "%R%\tools\nextgen\windows\r8_build.cmd"
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
