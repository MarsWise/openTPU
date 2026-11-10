@echo off
rem Claude: r8 DDR3: ddr133_check's cases (ECC 0 / 0 on r5 45b1 at 133.33 MHz) on r8 under three calibrations (no loadcal, fp4, ternary). No load, no rescan, no mode registers.
rem About 60-80 minutes; saves r8_ddr_check.log here. Not during another run on the card.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
scp %O% "%~dp0sync.bundle" "%~dp0r8_ddr_check.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_ddr_check.sh > /tmp/r8_ddr_check_exec.sh; bash /tmp/r8_ddr_check_exec.sh 2>&1 | tee /tmp/r8_ddr_check.log"
scp %O% %H%:/tmp/r8_ddr_check.log "%~dp0r8_ddr_check.log"
echo.
echo Log: %~dp0r8_ddr_check.log
pause
exit /b 0
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
