@echo off
rem Claude: a real LLM that ran on the ISA simulator only, on the card:  r8_llm.cmd lfm3 | lfm3-fp4 | minicpm5 | granite | maple | ouro | dory
rem maple: Maple-Preview's TQ2_0 GGUF already on the host (r8_fetch_models); a ternary MoE, the experts streamed.
rem Copies the weights to the host once (lfm3 8 GB, minicpm5 1.5 GB, granite 2.1 GB) and the host tree, runs
rem tools\llm_card.py: logits against the fp32 reference, greedy tokens, tok/s, ECC. Saves r8_llm_<model>.log here.
rem No bitstream load, no rescan. Not during another run on the card.
setlocal
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set M=%~1
if "%M%"=="" set M=lfm3
set WD=%M%
if /i "%M%"=="maple" goto maple
if /i "%M%"=="lfm3" (set "SRC=C:\myOpenTPU\_claude\lfm3_text" & set "PROBE=index.json")
if /i "%M%"=="lfm3-fp4" (set "SRC=C:\myOpenTPU\_claude\lfm3_text" & set "PROBE=index.json" & set "WD=lfm3")
if /i "%M%"=="minicpm5" (set "SRC=C:\ai-models\small\MiniCPM5_2B" & set "PROBE=MiniCPM5-2B-Q4_K_M.gguf")
if /i "%M%"=="granite" (set "SRC=C:\ai-models\small\Granite_4.2_3B" & set "PROBE=granite-4.2-3b-Q4_K_M.gguf")
if /i "%M%"=="ouro" (set "SRC=C:\ai-models\ouro-1.4b-thinking" & set "PROBE=model.safetensors")
if /i "%M%"=="dory" (set "SRC=C:\ai-models\dory-4b" & set "PROBE=NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")
if not defined SRC (echo model: lfm3, lfm3-fp4, minicpm5, granite, maple, ouro or dory & pause & exit /b 2)
if not exist "%SRC%\%PROBE%" (echo No %SRC%\%PROBE% & pause & exit /b 1)
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
goto copy
:maple
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
goto run
:copy
ssh %O% %H% "mkdir -p ~/llm_card/%WD%; test -f ~/llm_card/%WD%/.copied" && goto run
echo Copying %SRC% to the host (once)...
if /i "%WD%"=="lfm3" (scp %O% "%SRC%\*.bin" "%SRC%\*.json" %H%:llm_card/%WD%/ || goto fail) else if /i "%WD%"=="ouro" (scp %O% "%SRC%\model.safetensors" "%SRC%\config.json" "%SRC%\tokenizer.json" %H%:llm_card/%WD%/ || goto fail) else (scp %O% "%SRC%\%PROBE%" %H%:llm_card/%WD%/ || goto fail)
ssh %O% %H% "touch ~/llm_card/%WD%/.copied" || goto fail
:run
scp %O% "%~dp0sync.bundle" "%~dp0r8_llm.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_llm.sh > /tmp/r8_llm_exec.sh && bash -o pipefail -c 'bash /tmp/r8_llm_exec.sh %M% 2>&1 | tee /tmp/r8_llm.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_llm.log "%~dp0r8_llm_%M%.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_llm_%M%.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host or to copy (is it on? ping 192.168.0.157)
pause
exit /b 1
