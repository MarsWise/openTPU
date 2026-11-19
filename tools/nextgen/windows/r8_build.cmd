@echo off
rem Claude: build r8 (m8, 133.33 MHz; r7 + VRAND, TCQ, TATAN, QST4, LDX, TD5, VBFLY, VAFF, QST6, VCSWP) with Vivado in
rem the checkout containing this script: tools\build_r8_133.ps1.
rem   r8_build.cmd            full build (several hours); otpu.bit only if setup/hold pass
rem   r8_build.cmd Synth      synthesis + structural checks only (about an hour): try this first
rem   r8_build.cmd Impl       implementation only, reusing this checkout's finished synthesis (after a crash in place/route)
rem   r8_build.cmd Prepare    the Vivado project only
rem   r8_build.cmd Plan       inspect settings without creating a project
rem NEVER programs the FPGA: loading the bitstream is a separate step you do (volatile JTAG).
rem Defaults (tools\build_r8_133.ps1): r5x's -- OTPU_MEMCH_WREG 0, OTPU_LITEDRAM_RTW 3, TRACE_DEPTH 16384,
rem   OTPU_DROP_FEATURES "TBIN TBGSC" (none: every capability), margins uc2 0.10 / sys 0.10 / wr 0.05;
rem   CFG_HASH 432e34c2. A variable set before the call wins.
setlocal
set "MODE=%~1"
if "%MODE%"=="" set "MODE=Build"
set "PLAN="
if /i "%MODE%"=="Plan" (set "MODE=Build" & set "PLAN=-Plan")
cd /d "%~dp0..\..\.." || (echo Cannot locate the checkout containing this script & pause & exit /b 1)
findstr /c:"VAFF" rtl\shell\otpu_ctrl.sv >nul || (echo This checkout has no VAFF capability & pause & exit /b 1)
findstr /c:"QST6" rtl\shell\otpu_ctrl.sv >nul || (echo This checkout has no QST6 capability & pause & exit /b 1)
findstr /c:"VCSWP" rtl\shell\otpu_ctrl.sv >nul || (echo This checkout has no VCSWP capability & pause & exit /b 1)
findstr /c:"TGQ4" rtl\shell\otpu_ctrl.sv >nul || (echo This checkout has no TGQ4 capability & pause & exit /b 1)
rem OTPU_ALLOW_PARALLEL=1: start while another Vivado runs (a build in its last steps: memory for two)
if not defined PLAN if not "%OTPU_ALLOW_PARALLEL%"=="1" (
    tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo a Vivado process is running: stop & pause & exit /b 1)
)
for /f %%h in ('git rev-parse --short^=8 HEAD') do set REV=%%h
git log --oneline -1
if not exist "build\logs" mkdir "build\logs" || (echo Cannot create build\logs & exit /b 1)
rem OTPU_BUILD_TAG (e.g. a, b): a second build of one commit in its own directory, <rev>-<tag>
set "TREV=%REV%"
if defined OTPU_BUILD_TAG set "TREV=%REV%-%OTPU_BUILD_TAG%"
set "LOG=%CD%\build\logs\r8_build_%MODE%_%TREV%.log"
set "OTPU_R8_LOG=%LOG%"
echo === %MODE% of %REV%; log %LOG%; output build\vivado-r8-m8-133.33-%TREV%
echo === the python the build uses (CFG_HASH):
where python
python -c "import sys, numpy; print(sys.executable, sys.version.split()[0], 'numpy', numpy.__version__)"
if errorlevel 1 goto nopy
python -c "import sys,dataclasses; sys.path.insert(0,'.'); from tools.bonsai2_validate import configuration; from opentpu.target import arch_hash; import os; d=os.environ.get('OTPU_DROP_FEATURES') or 'TBIN TBGSC'; d='' if d.lower()=='none' else d; off=set(d.upper().replace(',',' ').split()); caps={k: k not in off for k in 'TGSC VCMP TBIN TBGSC VU8 TGRQ VRAND TCQ TATAN QST4 LDX TD5 VBFLY VAFF QST6 VCSWP TGQ4'.split()}; print('CFG_HASH check', format(arch_hash(dataclasses.replace(configuration(), **caps)), '08x'), '(less ' + ' '.join(sorted(off)) + ')' if off else '')"
if errorlevel 1 goto nohash
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { & '.\tools\build_r8_133.ps1' -Mode '%MODE%' %PLAN% *>&1 | Tee-Object -FilePath $env:OTPU_R8_LOG -Append; if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; exit 0 } catch { Write-Error $_; exit 1 }"
set RC=%errorlevel%
echo === exit %RC%
set OUTD=build\vivado-r8-m8-133.33-%TREV%
if not defined PLAN if exist "%OUTD%\reports\SUMMARY.txt" type "%OUTD%\reports\SUMMARY.txt"
if not defined PLAN for /r "%OUTD%" %%f in (*.bit) do certutil -hashfile "%%f" SHA256
echo (nothing was loaded into the FPGA)
if defined PLAN exit /b %RC%
pause
exit /b %RC%
:nopy
echo The python on PATH cannot import numpy: the build computes CFG_HASH with it.
echo Install it for that python: python -m pip install numpy   then run r8_build.cmd again.
pause
exit /b 1
:nohash
echo The CFG_HASH step failed: its traceback is above. Send it to Claude.
pause
exit /b 1
