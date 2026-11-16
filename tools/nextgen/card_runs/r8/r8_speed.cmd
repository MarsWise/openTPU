@echo off
rem Claude: speed of the LLMs on the host, on the loaded firmware:  r8_speed.cmd [qwen3 qwen35 qwen35-2b lfm2 bitnet bonsai]
rem One fixed prompt, greedy, 128 tokens, --strict-fpga; TTFT, prefill and decode tok/s, ECC per model. About 30-60 minutes.
rem Saves r8_speed.log here. No bitstream load, no rescan. Not during another run on the card.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
scp %O% "%~dp0sync.bundle" "%~dp0r8_speed.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_speed.sh > /tmp/r8_speed_exec.sh && bash -o pipefail -c 'bash /tmp/r8_speed_exec.sh %* 2>&1 | tee /tmp/r8_speed.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_speed.log "%~dp0r8_speed.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_speed.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
