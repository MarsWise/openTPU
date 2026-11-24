@echo off
rem Claude: the new models onto the card's host (no card use: it may run beside card tests).
rem   r8_fetch_models.cmd            Ivonar Nano 349M (95 MB) and Maple-Preview 20B-A1B (TQ2_0, 5.91 GiB, ~50 min)
rem   r8_fetch_models.cmd ivonar     one of them
rem Also copies Bonsai 2 4B's MTP GGUF (1.27 GB) from C:\ai-models over the LAN, once.
rem At most 2 MB/s from Hugging Face, resumable (run again after a break), sha256 checked. Log: r8_fetch_models.log.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set B=C:\ai-models\ternary\prism-ml--Ternary-Bonsai-2-4B-gguf
set BF=Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf
if exist "%B%\%BF%" (
  ssh %O% %H% "mkdir -p ~/openTPU_BitNET/models/bonsai2-4b-mtp; test -f ~/openTPU_BitNET/models/bonsai2-4b-mtp/%BF%" || scp %O% "%B%\%BF%" "%B%\SHA256SUMS" "%B%\README.md" %H%:openTPU_BitNET/models/bonsai2-4b-mtp/ || goto fail
) else (echo No %B%\%BF%: Bonsai MTP not copied)
scp %O% "%~dp0r8_fetch_models.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_fetch_models.sh > /tmp/r8_fetch_models_exec.sh && bash -o pipefail -c 'bash /tmp/r8_fetch_models_exec.sh %* 2>&1 | tee /tmp/r8_fetch_models.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_fetch_models.log "%~dp0r8_fetch_models.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_fetch_models.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
