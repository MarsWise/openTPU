@echo off
rem Claude: the r8 firmware (a4138552, CFG_HASH 93e29b18) once LOADED (r8_load.cmd, then r8_restore_pcie.cmd):
rem r8 instructions card == ISA, Qwen3 tiny KV int8/kv6/k6v6 card == ISA, then the r7 model steps
rem (Diffusion, SmolVLA, YOLOX strict FPGA-only, vLLM sampled + greedy, Bonsai GRQ). About 4-6 hours; saves r8_card_tests.log and .tgz here. No bitstream load, no rescan.
rem Not during another run on the card.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
scp %O% "%~dp0sync.bundle" "%~dp0r8_card_tests.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_card_tests.sh > /tmp/r8_card_tests_exec.sh && bash -o pipefail -c 'bash /tmp/r8_card_tests_exec.sh 2>&1 | tee /tmp/r8_card_tests.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_card_tests.log "%~dp0r8_card_tests.log" || set RUNRC=1
scp %O% %H%:/tmp/r8_card_tests.tgz "%~dp0r8_card_tests.tgz" || set RUNRC=1
echo.
echo Log: %~dp0r8_card_tests.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
