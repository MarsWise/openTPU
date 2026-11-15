@echo off
rem Claude: Cactus Needle (26M) on the card: copies the checkpoint (77 MB, once) and the host tree, runs
rem tools\needle_card.py (int8 and CQ4): card == ISA, the README's tool call, tokens picked on the card, ECC.
rem About 10 minutes; saves r8_needle.log here. No bitstream load, no rescan. Not during another run on the card.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set M=C:\ai-models\cactus\Cactus-Compute--needle
if not exist "%M%\model.safetensors" (echo No %M%\model.safetensors & pause & exit /b 1)
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
ssh %O% %H% "mkdir -p ~/needle; test -f ~/needle/model.safetensors" || scp %O% "%M%\model.safetensors" "%M%\needle-cq4.zip" %H%:needle/ || goto fail
scp %O% "%~dp0sync.bundle" "%~dp0r8_needle.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_needle.sh > /tmp/r8_needle_exec.sh && bash -o pipefail -c 'bash /tmp/r8_needle_exec.sh 2>&1 | tee /tmp/r8_needle.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_needle.log "%~dp0r8_needle.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_needle.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
