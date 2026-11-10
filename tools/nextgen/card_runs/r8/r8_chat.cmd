@echo off
rem Claude: an interactive chat on the r8 card over ssh: r8_chat.cmd [model] [otpu-chat options]
rem   model: bitnet (default) ^| bonsai ^| qwen3 ^| qwen35 ^| qwen35-2b ^| lfm2 ^| reasonlite ^| a checkpoint directory on the host
rem   e.g.  r8_chat.cmd bitnet --head-format int4      r8_chat.cmd qwen35-2b --wformat fp4      r8_chat.cmd qwen3 --prompt "Why is the sky blue?"
rem After r8_load + r8_restore_pcie + one r8_card_tests.cmd (its step 0b calibrates the DDR3). Not during another run. /exit or Ctrl+C to leave.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
scp %O% "%~dp0r8_chat.sh" %H%:/tmp/ || goto fail
ssh -t %O% %H% "tr -d '\r' < /tmp/r8_chat.sh > /tmp/r8_chat_exec.sh; bash /tmp/r8_chat_exec.sh %*"
pause
exit /b 0
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
