@echo off
rem Claude: where Kokoro's and SD's card time goes:  r8_media_profile.cmd
rem Every program of Kokoro phrase 1 and of SD 1.5 (256 px, one step) run on its own, cycles summed by op family.
rem Needs r8_kokoro.cmd and r8_sd.cmd to have run once (the weights on the host). ~15 min. Saves r8_media_profile.log.
rem No bitstream load, no rescan. Not during another run on the card.
setlocal
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
:run
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
scp %O% "%~dp0sync.bundle" "%~dp0r8_media_profile.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_media_profile.sh > /tmp/r8_media_profile_exec.sh && bash -o pipefail -c 'bash /tmp/r8_media_profile_exec.sh 2>&1 | tee /tmp/r8_media_profile.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_media_profile.log "%~dp0." || set RUNRC=1
echo.
echo Log: %~dp0r8_media_profile.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host or to copy (is it on? ping 192.168.0.157)
pause
exit /b 1
