@echo off
rem Claude: every card test on the LOADED firmware, one after another, unattended (about 12-16 hours):
rem   1 r8_models      the card's build id and ECC counters (read only)
rem   2 r8_card_tests  instructions == ISA, KV, Diffusion, SmolVLA, YOLOX, vLLM, Bonsai
rem   3 r8_ddr_check   the DDR3 soak, 4 passes under each calibration (criterion: ECC 0 / 0 in every pass)
rem   4 r8_needle      Cactus Needle int8 / CQ4
rem   5 r8_llm         LFM3, MiniCPM5-2B, Granite 4.2 3B against the fp32 reference
rem   6 r8_speed       chat speed: Qwen3, Qwen3.5-0.8B / 2B, LFM2.5, BitNet, Bonsai
rem   7 r8_ttft        time to first token, cold / warm program cache, 4-bit Qwen3
rem   8 r8_profile     where a decode token's cycles go (Bonsai, BitNet, Qwen3.5)
rem   9 r8_bonsai_mtp  Bonsai 2 4B plain vs MTP (2-row verify on GRQ QUAD where CAPS2 has TGQ4M)
rem  10 r8_mtp         Qwen3.5-0.8B / 2B plain vs MTP, int8 and fp4
rem  11 r8_llm         LFM3 fp4, Ouro-1.4B, Dory 4B, Maple-Preview (ternary MoE)
rem  12 r8_more_models ReasonLite, LFM2.5 thinking, FunctionGemma, Qwen3.5-4B fp4 (otpu-chat --strict-fpga)
rem  13 r8_paddleocr   PaddleOCR-VL 1.6
rem  14 r8_kokoro      Kokoro-82M text to speech, all on the device (two phrases, WAVs)
rem r8_all_tests.cmd new: steps 1 and 8-14 alone (the models not yet run on the card).
rem The DDR3 step under the loadcal kind only (KINDS=ternary:4; the no-loadcal phase fails by
rem design): set KINDS before running for another set. Takes sync_next.bundle first, if it is there.
rem SKIP_DDR=1: without step 3 (r8_after_build.cmd ran the DDR3 steps already).
rem The logs are copied to runs\<date-time>\ beside this file. No bitstream load, no rescan.
rem Do not press Ctrl+C in this window (it interrupts the step; its host side goes on): to copy, select
rem with the mouse and press Enter.
rem Before it: r8_load_newest.cmd (or r8_load.cmd), then r8_restore_pcie.cmd.
setlocal
if /i "%~1"=="new" set ONLY=new
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%t"
set "OUT=%~dp0runs\%TS%"
mkdir "%OUT%" || (echo cannot create %OUT% & pause & exit /b 1)
set PASSES=4
if not defined KINDS set KINDS=ternary:4
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
echo === logs: %OUT%
call :step r8_models r8_models.log
if /i "%ONLY%"=="new" goto newsteps
call :step r8_card_tests r8_card_tests.log
if exist "%~dp0r8_card_tests.tgz" copy /y "%~dp0r8_card_tests.tgz" "%OUT%\" >nul
if not defined SKIP_DDR call :step r8_ddr_check r8_ddr_check.log
call :step r8_needle r8_needle.log
call :step "r8_llm lfm3" r8_llm_lfm3.log
call :step "r8_llm minicpm5" r8_llm_minicpm5.log
call :step "r8_llm granite" r8_llm_granite.log
call :step r8_speed r8_speed.log
call :step r8_ttft r8_ttft.log
:newsteps
call :step r8_profile r8_profile.log
call :step r8_bonsai_mtp r8_bonsai_mtp.log
call :step r8_mtp r8_mtp.log
call :step "r8_llm lfm3-fp4" r8_llm_lfm3-fp4.log
call :step "r8_llm ouro" r8_llm_ouro.log
call :step "r8_llm dory" r8_llm_dory.log
call :step "r8_llm maple" r8_llm_maple.log
call :step r8_more_models r8_more_models.log
call :step r8_paddleocr r8_paddleocr.log
call :step r8_kokoro r8_kokoro.log
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
