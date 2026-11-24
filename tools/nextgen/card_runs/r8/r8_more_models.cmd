@echo off
rem Claude: models that ran only on the ISA simulator, on the card (otpu-chat --strict-fpga, greedy, 64 tokens):
rem   r8_more_models.cmd [reasonlite lfm25-thinking functiongemma qwen35-4b]
rem Copies each model once from C:\ai-models to the host's ~/llm_card/<name> (*.safetensors, *.json, tokenizer files;
rem no .py), Qwen3.5-4B Qwenputer is 18 GB. Saves r8_more_models.log here. No bitstream load, no rescan.
rem Not during another run on the card.
setlocal enabledelayedexpansion
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set A=C:\ai-models
set LIST=%*
if "%LIST%"=="" set LIST=reasonlite lfm25-thinking functiongemma qwen35-4b
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
for %%m in (%LIST%) do (
  set "SRC="
  if /i "%%m"=="reasonlite" set "SRC=%A%\hf-specialized-2026\reasonlite"
  if /i "%%m"=="lfm25-thinking" set "SRC=%A%\hf-specialized-2026\lfm25-thinking"
  if /i "%%m"=="functiongemma" set "SRC=%A%\agents\FunctionGemma_270M"
  if /i "%%m"=="qwen35-4b" set "SRC=%A%\coding\Qwen3.5_4B_Qwenputer"
  if not defined SRC (echo unknown model %%m & pause & exit /b 2)
  ssh %O% %H% "mkdir -p ~/llm_card/%%m; test -f ~/llm_card/%%m/.copied" || (
    echo Copying !SRC! to the host once...
    scp %O% "!SRC!\*.safetensors" "!SRC!\*.json" %H%:llm_card/%%m/ || goto fail
    if exist "!SRC!\tokenizer.model" scp %O% "!SRC!\tokenizer.model" %H%:llm_card/%%m/
    if exist "!SRC!\merges.txt" scp %O% "!SRC!\merges.txt" %H%:llm_card/%%m/
    if exist "!SRC!\chat_template.jinja" scp %O% "!SRC!\chat_template.jinja" %H%:llm_card/%%m/
    ssh %O% %H% "touch ~/llm_card/%%m/.copied" || goto fail
  )
)
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
scp %O% "%~dp0sync.bundle" "%~dp0r8_more_models.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_more_models.sh > /tmp/r8_more_models_exec.sh && bash -o pipefail -c 'bash /tmp/r8_more_models_exec.sh %LIST% 2>&1 | tee /tmp/r8_more_models.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_more_models.log "%~dp0r8_more_models.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_more_models.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host or to copy (is it on? ping 192.168.0.157)
pause
exit /b 1
