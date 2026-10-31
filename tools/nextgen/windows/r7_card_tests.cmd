@echo off
rem Claude: r7 software (feature/r7-fpga) on the LOADED firmware, every model path FPGA-only:
rem Diffusion Policy and SmolVLA resident, YOLOX strict, vLLM sampled + greedy, Bonsai GRQ (if TGRQ).
rem About 3-5 hours; saves r7_card_tests.log and .tgz here. No bitstream load, no rescan.
rem Not during another run on the card.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
scp %O% "%~dp0sync.bundle" "%~dp0r7_card_tests.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r7_card_tests.sh > /tmp/r7_card_tests_exec.sh; bash /tmp/r7_card_tests_exec.sh 2>&1 | tee /tmp/r7_card_tests.log"
scp %O% %H%:/tmp/r7_card_tests.log "%~dp0r7_card_tests.log"
scp %O% %H%:/tmp/r7_card_tests.tgz "%~dp0r7_card_tests.tgz"
echo.
echo Log: %~dp0r7_card_tests.log
pause
exit /b 0
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
