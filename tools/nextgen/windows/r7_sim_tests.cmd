@echo off
rem Claude: the r7 tests on the Linux host's CPU (ISA simulator; RTL too if Verilator is there).
rem No card access. Hours for everything; "r7_sim_tests.cmd isa" runs the ISA tests only (-k "not rtl").
rem Saves r7_sim_tests.log and .tgz here. Not during a card run.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set K=all
if /i "%~1"=="isa" set K=isa
scp %O% "%~dp0sync.bundle" "%~dp0r7_sim_tests.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r7_sim_tests.sh > /tmp/r7_sim_tests_exec.sh; bash /tmp/r7_sim_tests_exec.sh %K% 2>&1 | tee /tmp/r7_sim_tests.log"
scp %O% %H%:/tmp/r7_sim_tests.log "%~dp0r7_sim_tests.log"
scp %O% %H%:/tmp/r7_sim_tests.tgz "%~dp0r7_sim_tests.tgz"
echo.
echo Log: %~dp0r7_sim_tests.log
pause
exit /b 0
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
