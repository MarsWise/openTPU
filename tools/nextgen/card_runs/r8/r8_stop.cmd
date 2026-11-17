@echo off
rem Claude: shows what is still running on the card (a run left behind by a closed window) and, after "yes", stops it.
rem No bitstream load, no rescan, no register writes. "r8_stop.cmd check" only looks (exit 3 = busy, 0 = free).
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
scp %O% "%~dp0r8_stop.sh" %H%:/tmp/ >nul || (echo FAILED to reach the host & if not "%1"=="check" pause & exit /b 1)
ssh %O% %H% "tr -d '\r' < /tmp/r8_stop.sh > /tmp/r8_stop_exec.sh; bash /tmp/r8_stop_exec.sh"
set RC=%errorlevel%
if "%1"=="check" exit /b %RC%
if not "%RC%"=="3" (pause & exit /b %RC%)
set A=
set /p A=Stop these runs? Type yes: 
if /i not "%A%"=="yes" (echo left running & pause & exit /b 3)
ssh %O% %H% "bash /tmp/r8_stop_exec.sh kill"
pause
