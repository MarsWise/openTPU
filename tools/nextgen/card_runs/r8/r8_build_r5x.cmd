@echo off
rem Claude: r5x -- r5 (45b1c1d3) plus the r8 instructions, built as r5 was: r5's RTL pipeline (no
rem   MXU FIFO output register 5e6293e6 / 63cbadcd, no sequencer T stage 800ae9d0), r5's memory
rem   write path (OTPU_MEMCH_WREG=0), rtw 3, trace 16384; the instructions VCMP, VU8, GRQ, VRAND,
rem   CQ, TATAN, QST4, LDX, T5, BFLY, AFF, QST6, CSWP, GRQ QUAD (no BIN / BIN GSC), the MM k0 fix.
rem   The sources: the local branch exp/r5x (commit c42beb65) from r5x.bundle beside this script,
rem   in its own checkout C:\myOpenTPU\openTPU-r5x (a local clone of openTPU-r8; nothing pushed).
rem   CFG_HASH 432e34c2 (as A). A full build, several hours; otpu.bit only if setup / hold pass
rem   (r8 added those pipeline stages for timing: it may not close). After a bitstream its otpu.bit,
rem   manifest and reports are copied to openTPU-r8\build\vivado-r8-m8-133.33-c42beb65, so
rem   r8_after_build.cmd c42beb65 takes it. Nothing is loaded.
rem   r8_build_r5x.cmd [parallel | plan]
setlocal
set "R8=C:\myOpenTPU\openTPU-r8"
set "R=C:\myOpenTPU\openTPU-r5x"
set "SHA=c42beb65c1196c4413af1802d4ac64d91adc4a14"
set "SHORT=c42beb65"
if not exist "%~dp0r5x.bundle" (echo no r5x.bundle beside this script & goto :err)
if not exist "%R%\.git" (
  echo === a local clone of %R8% in %R%
  git clone -q --no-checkout "%R8%" "%R%" || goto :err
)
git -C "%R%" fetch -q "%~dp0r5x.bundle" +exp/r5x:refs/heads/exp/r5x || goto :err
rem a clone made with --no-checkout has an empty index (every file reads as deleted): no local work
set "FRESH=1"
for /f %%f in ('git -C "%R%" ls-files') do set "FRESH="
if defined FRESH (
  git -C "%R%" checkout -q -f exp/r5x || goto :err
) else (
  git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
  git -C "%R%" checkout -q exp/r5x || goto :err
)
for /f %%h in ('git -C "%R%" rev-parse HEAD') do if not "%%h"=="%SHA%" (echo %R% is at %%h, not %SHA% & goto :err)
findstr /c:"OREG" "%R%\rtl\tech\generic\otpu_tech_ram.sv" >nul && (echo the checkout has the FIFO output register: not r5x & goto :err)
findstr /c:"WREG" "%R%\rtl\shell\otpu_mem_ch.sv" >nul || (echo otpu_mem_ch has no WREG: not r5x & goto :err)
findstr /c:"cmd_gq4" "%R%\rtl\mxu\otpu_mxu.sv" >nul || (echo the MXU has no GRQ QUAD: not r5x & goto :err)
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
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (if /i not "%~1"=="plan" tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_r5x.cmd parallel". & pause & exit /b 1))
git -C "%R%" log --oneline -1
echo === expected: hash=432e34c2 (r8 less TBIN TBGSC)
echo === r5x: r5's pipeline + the r8 instructions, r5's write path, rtw 3, trace 16384
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
