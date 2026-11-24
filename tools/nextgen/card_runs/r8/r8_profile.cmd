@echo off
rem Claude: where a decode token's cycles go, on the loaded firmware:  r8_profile.cmd [bonsai bonsai-grq bitnet qwen35 qwen35-2b]
rem The DRAM calibration first, then per model 64 greedy tokens with the card's counters (tools/decode_profile.py
rem --device-counters): Mcycles, DRAM bytes, MXU busy / MAC a token. About 30-45 minutes.
rem Saves r8_profile.log and prof_*.json here. No bitstream load, no rescan. Not during another run on the card.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
scp %O% "%~dp0sync.bundle" "%~dp0r8_profile.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_profile.sh > /tmp/r8_profile_exec.sh && bash -o pipefail -c 'bash /tmp/r8_profile_exec.sh %* 2>&1 | tee /tmp/r8_profile.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_profile.log "%~dp0r8_profile.log" || set RUNRC=1
scp %O% "%H%:/tmp/prof_*.json" "%~dp0." || set RUNRC=1
echo.
echo Log: %~dp0r8_profile.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
