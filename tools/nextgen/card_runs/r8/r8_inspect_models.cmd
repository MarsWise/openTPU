@echo off
rem Claude: what the new models' files hold (Maple-Preview GGUF header, Ivonar Nano checkpoint keys), read on the
rem card's host without running their code. No card use: it may run beside card tests. Log: r8_inspect_models.log.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
scp %O% "%~dp0r8_inspect_models.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_inspect_models.sh > /tmp/r8_inspect_models_exec.sh && bash -o pipefail -c 'bash /tmp/r8_inspect_models_exec.sh 2>&1 | tee /tmp/r8_inspect_models.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_inspect_models.log "%~dp0r8_inspect_models.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_inspect_models.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
