@echo off
rem Claude: MTP decoding on the card: r8_mtp.cmd [model ...] (default qwen35 qwen35-2b; bonsai2-4b on firmware with
rem CAPS2 TGQ4M). Three prompts greedy per model: plain and --mtp, int8 and fp4 layers; the replies must be equal;
rem decode tok/s and MTP acceptance, ECC per run. Takes sync_next.bundle first. ~25 min for the two Qwen3.5 models.
rem No bitstream load, no rescan. Not during another run on the card. Log: r8_mtp.log.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
scp %O% "%~dp0sync.bundle" "%~dp0r8_mtp.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_mtp.sh > /tmp/r8_mtp_exec.sh && bash -o pipefail -c 'FP4=1 bash /tmp/r8_mtp_exec.sh %* 2>&1 | tee /tmp/r8_mtp.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_mtp.log "%~dp0r8_mtp.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_mtp.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
