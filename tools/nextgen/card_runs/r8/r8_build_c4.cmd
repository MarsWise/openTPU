@echo off
rem Claude: r8 C4 -- C3 (rtw 4, trace 4096, margins) with the MXU's unused weight formats left out for
rem   area: OTPU_DROP_FEATURES=TD5 TBIN TBGSC (otpu_mxu HAS_T5 / HAS_BIN 0: the D T5 trit tables and the
rem   BIN decode are not built; CAPS2 bits 6, 7, 15 clear, so the host refuses those programs). No model
rem   on the card uses them. TCQ stays: the 6-bit KV cache (kv6 / k6v6), Needle CQ4, Cactus CQ weights.
rem   CFG_HASH changes (computed from the capabilities built). A full build, several hours; otpu.bit
rem   only if setup / hold pass. Nothing is loaded. Run "r8_build_c4.cmd plan" first: no Vivado, it
rem   prints the configuration and the CFG_HASH.
rem   r8_build_c4.cmd [parallel | plan]
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
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_c4.cmd parallel". & pause & exit /b 1))
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
git -C "%R%" log --oneline -1
echo === expected: hash=37f51889 (the C3 capabilities a905c3db less TD5 TBIN TBGSC; opentpu.target arch_hash)
echo === C4: C3 less TD5 TBIN TBGSC (area), LiteDRAM rtw 4, TRACE_DEPTH 4096, margins uc2 0.10 / sys 0.10 / wr 0.05
if /i "%~1"=="plan" (call "%R%\tools\nextgen\windows\r8_build.cmd" Plan) else (call "%R%\tools\nextgen\windows\r8_build.cmd")
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
