@echo off
rem Claude: r8 + C2 built again with setup margins on the memory controllers while placing and routing
rem (removed for the reports and the bitstream, as the core's): OTPU_SYS_MARGIN 0.10 ns (LiteDRAM's sys
rem clock, c969ca4c closed at +0.003) and OTPU_WR_MARGIN 0.15 ns (sys -> the write clocks, channel 0 at
rem +0.029). Same RTL as c969ca4c (CFG_HASH a905c3db); several hours; otpu.bit only if setup / hold pass.
rem ONLY after the DDR diagnosis says the slack is the cause (see ПОСЛЕ_ТЕСТОВ.md). Nothing is loaded.
rem   r8_build_ddr.cmd [parallel]      other values: set OTPU_SYS_MARGIN / OTPU_WR_MARGIN before it
setlocal
set "R=C:\myOpenTPU\openTPU-r8"
if "%OTPU_SYS_MARGIN%"=="" set OTPU_SYS_MARGIN=0.10
if "%OTPU_WR_MARGIN%"=="" set OTPU_WR_MARGIN=0.15
if /i "%~1"=="parallel" (set OTPU_ALLOW_PARALLEL=1) else (tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running: wait for it, or run "r8_build_ddr.cmd parallel". & pause & exit /b 1))
git -C "%R%" fetch origin feature/r7-fpga || goto :err
git -C "%R%" status --porcelain --untracked-files=no | findstr . >nul && (echo %R% has local changes: not updated & goto :err)
git -C "%R%" checkout -q feature/r7-fpga && git -C "%R%" merge -q --ff-only origin/feature/r7-fpga || goto :err
findstr /c:"OTPU_WR_MARGIN" "%R%\boards\ypcb-00338\vivado\margins.tcl" >nul || (echo no OTPU_WR_MARGIN in margins.tcl: wrong checkout & goto :err)
findstr /c:"OTPU_WR_MARGIN" "%R%\tools\build_r5_timing.ps1" >nul || (echo build_r5_timing.ps1 does not pass the memory margins: wrong checkout & goto :err)
git -C "%R%" log --oneline -1
echo === memory margins: sys %OTPU_SYS_MARGIN% ns, wr %OTPU_WR_MARGIN% ns
call "%R%\tools\nextgen\windows\r8_build.cmd"
exit /b %errorlevel%
:err
echo failed
pause
exit /b 1
