@echo off
rem Claude: TTFT (program cache) and 4-bit speed on the LOADED r8 firmware, about 20 minutes:
rem Saves r8_ttft.log here. Writes only ~/otpu-cache on the host.
rem No bitstream load, no rescan, no mode registers. Not during another run on the card.
rem Takes sync_next.bundle (the tree with the tool) as sync.bundle first, if it is there.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
call "%~dp0r8_stop.cmd" check
if errorlevel 1 (echo The card is busy or the host is unreachable: wait, or r8_stop.cmd. & pause & exit /b 1)
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
scp %O% "%~dp0sync.bundle" "%~dp0r8_ttft.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_ttft.sh > /tmp/r8_ttft_exec.sh; bash /tmp/r8_ttft_exec.sh 2>&1 | tee /tmp/r8_ttft.log"
scp %O% %H%:/tmp/r8_ttft.log "%~dp0r8_ttft.log"
echo.
echo Log: %~dp0r8_ttft.log
pause
exit /b 0
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
