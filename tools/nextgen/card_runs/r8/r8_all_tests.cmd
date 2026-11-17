@echo off
rem Claude: every card test on the LOADED firmware, one after another, unattended (about 6-9 hours):
rem   1 r8_models      the card's build id and ECC counters (read only)
rem   2 r8_card_tests  instructions == ISA, KV, Diffusion, SmolVLA, YOLOX, vLLM, Bonsai
rem   3 r8_ddr_check   the DDR3 soak, 4 passes under each calibration (criterion: ECC 0 / 0 in every pass)
rem   4 r8_needle      Cactus Needle int8 / CQ4
rem   5 r8_llm         LFM3, MiniCPM5-2B, Granite 4.2 3B against the fp32 reference
rem   6 r8_speed       chat speed: Qwen3, Qwen3.5-0.8B / 2B, LFM2.5, BitNet, Bonsai
rem   7 r8_ttft        time to first token, cold / warm program cache, 4-bit Qwen3
rem The DDR3 step under the loadcal kind only (KINDS=ternary:4; the no-loadcal phase fails by
rem design): set KINDS before running for another set. Takes sync_next.bundle first, if it is there.
rem SKIP_DDR=1: without step 3 (r8_after_build.cmd ran the DDR3 steps already).
rem The logs are copied to runs\<date-time>\ beside this file. No bitstream load, no rescan.
rem Do not press Ctrl+C in this window (it interrupts the step; its host side goes on): to copy, select
rem with the mouse and press Enter.
rem Before it: r8_load_newest.cmd (or r8_load.cmd), then r8_restore_pcie.cmd.
setlocal
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%t"
set "OUT=%~dp0runs\%TS%"
mkdir "%OUT%" || (echo cannot create %OUT% & pause & exit /b 1)
set PASSES=4
if not defined KINDS set KINDS=ternary:4
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
echo === logs: %OUT%
call :step r8_models r8_models.log
call :step r8_card_tests r8_card_tests.log
if exist "%~dp0r8_card_tests.tgz" copy /y "%~dp0r8_card_tests.tgz" "%OUT%\" >nul
if not defined SKIP_DDR call :step r8_ddr_check r8_ddr_check.log
call :step r8_needle r8_needle.log
call :step "r8_llm lfm3" r8_llm_lfm3.log
call :step "r8_llm minicpm5" r8_llm_minicpm5.log
call :step "r8_llm granite" r8_llm_granite.log
call :step r8_speed r8_speed.log
call :step r8_ttft r8_ttft.log
echo.
echo === ALL DONE. Results by step:
type "%OUT%\steps.txt"
echo === logs: %OUT%
pause
exit /b 0
:step
echo.
echo ================ %date% %time% %~1
rem a run still holding the card (a step whose window was interrupted: its host side goes on): wait
rem for it, up to an hour, instead of every later step failing "another run is using the card"
set /a W=0
:wait
call "%~dp0r8_stop.cmd" check >nul 2>&1
if not errorlevel 3 goto go
if %W% geq 60 (echo the card stayed busy for an hour: %~1 skipped & echo %~1: busy, skipped >> "%OUT%\steps.txt" & exit /b 0)
if %W%==0 echo the card is busy (an earlier run on the host): waiting, a minute at a time
set /a W+=1
timeout /t 60 /nobreak >nul
goto wait
:go
for /f "tokens=1,*" %%a in ("%~1") do call "%~dp0%%a.cmd" %%b < nul
echo %~1: exit %errorlevel% >> "%OUT%\steps.txt"
if exist "%~dp0%~2" copy /y "%~dp0%~2" "%OUT%\" >nul
exit /b 0
