@echo off
rem Claude: r5x + the MXU FIFO output register -- the timing experiment. r5x (c42beb65, the current
rem   firmware) closed at WNS +0.002 ns; its three worst paths start at the MXU weight FIFO's block
rem   RAM clock-to-out (no output register) into the weight decode. This build adds r8's MXU timing
rem   pair: 5e6293e6 (the FIFOs read through DOB_REG, one Sp stage) with 63cbadcd, its correctness
rem   half (the PEs' select copies a register later; 5e6293e6 alone fails the select check); NOT the
rem   sequencer T stage (800ae9d0). Everything else as r5x: r5's write path, rtw 3, trace
rem   16384, no BIN / BIN GSC; CFG_HASH 432e34c2. The local branch exp/r5x-oreg (commit bb5af80d)
rem   from r5x_oreg.bundle beside this script, in the checkout C:\myOpenTPU\openTPU-r5x (the one
rem   r8_build_r5x.cmd made; nothing pushed). otpu.bit only if setup / hold pass; then copied to
rem   openTPU-r8\build\vivado-r8-m8-133.33-bb5af80d: r8_after_build.cmd bb5af80d, whose first
rem   minutes (the loadcal windows) answer whether the MXU pair narrows DDR3's read windows.
rem   r8_build_oreg.cmd [parallel | plan]
setlocal
set "R8=C:\myOpenTPU\openTPU-r8"
set "R=C:\myOpenTPU\openTPU-r5x"
set "SHA=bb5af80dac92bb63bfe2bef353887e519af5e694"
set "SHORT=bb5af80d"
if not exist "%~dp0r5x_oreg.bundle" (echo no r5x_oreg.bundle beside this script & goto :err)
if not exist "%R%\.git" (
  echo === a local clone of %R8% in %R%
  git clone -q --no-checkout "%R8%" "%R%" || goto :err
)
rem the bundle's branch into FETCH_HEAD (a fetch into a checked-out branch is refused)
git -C "%R%" fetch -q "%~dp0r5x_oreg.bundle" exp/r5x-oreg || goto :err
rem a clone made with --no-checkout has an empty index (every file reads as deleted): no local work
set "FRESH=1"
for /f %%f in ('git -C "%R%" ls-files') do set "FRESH="
if defined FRESH (
  git -C "%R%" checkout -q -f -B exp/r5x-oreg FETCH_HEAD || goto :err
) else (
  git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
  git -C "%R%" checkout -q -B exp/r5x-oreg FETCH_HEAD || goto :err
)
for /f %%h in ('git -C "%R%" rev-parse HEAD') do if not "%%h"=="%SHA%" (echo %R% is at %%h, not %SHA% & goto :err)
findstr /c:".OREG(1'b1)" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no FIFO output register: not this experiment & goto :err)
findstr /c:"pe_qs1" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no PE select register 63cbadcd: not this experiment & goto :err)
findstr /c:"t_adv" "%R%\rtl\seq\otpu_seq.sv" >nul && (echo the sequencer has the T stage 800ae9d0: not this experiment & goto :err)
findstr /c:"WREG" "%R%\rtl\shell\otpu_mem_ch.sv" >nul || (echo otpu_mem_ch has no WREG & goto :err)
findstr /c:"cmd_gq4" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no GRQ QUAD & goto :err)
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
set OTPU_BUILD_TAG=
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (if /i not "%~1"=="plan" tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_oreg.cmd parallel". & pause & exit /b 1))
git -C "%R%" log --oneline -1
echo === expected: hash=432e34c2 (r8 less TBIN TBGSC)
echo === r5x + r8's MXU timing pair (5e6293e6 + 63cbadcd): the timing experiment
if /i "%~1"=="plan" (call "%R%\tools\nextgen\windows\r8_build.cmd" Plan & exit /b 0)
call "%R%\tools\nextgen\windows\r8_build.cmd"
set "SRC=%R%\build\vivado-r8-m8-133.33-%SHORT%"
set "DST=%R8%\build\vivado-r8-m8-133.33-%SHORT%"
if not exist "%SRC%\otpu.bit" (echo no otpu.bit: timing not met or the build stopped. Send the end of the log. & pause & exit /b 1)
if not exist "%DST%\reports" mkdir "%DST%\reports"
for %%f in (otpu.bit SOURCE_COMMIT.txt build_manifest.json) do copy /y "%SRC%\%%f" "%DST%\" >nul
for %%f in (SUMMARY.txt util_hier.rpt util.rpt timing_summary.rpt) do copy /y "%SRC%\reports\%%f" "%DST%\reports\" >nul
echo === copied to %DST%: r8_after_build.cmd %SHORT%
certutil -hashfile "%DST%\otpu.bit" SHA256
pause
exit /b 0
:err
echo failed
pause
exit /b 1
