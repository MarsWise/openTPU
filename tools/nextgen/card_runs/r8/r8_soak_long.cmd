@echo off
rem Claude: the long DDR3 soak on the loaded firmware (the TZ's ECC 0 / 0 qualification):
rem   r8_soak_long.cmd [hours] [models...]     default 4 h; qwen35-2b bitnet qwen35 lfm2 qwen3
rem One calibration (selfcal + loadcal ternary:4; set KIND=int8:4 or fp4 before the call for another),
rem then the models in rotation with no recalibration until the hours pass, ECC after every run;
rem the table and VERDICT at the end. It runs on the host in the background (closing this window or
rem a dropped connection does not stop it: run this again to keep watching). The log comes back as
rem r8_soak_long.log here. No bitstream load, no rescan, no mode register. Not during another run.
setlocal
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set "K=%KIND%"
if "%K%"=="" set "K=ternary:4"
ssh %O% %H% "pgrep -f '[r]8_soak_long_exec.sh' >/dev/null" && (echo A soak is already running on the host: watching it. & goto watch)
ssh %O% %H% "pgrep -f '[c]laude_exec.sh|[r]8_speed_exec.sh|[r]8_ddr_check' >/dev/null" && (echo Another run is using the card: wait for it to finish first. & pause & exit /b 1)
scp %O% "%~dp0sync.bundle" "%~dp0r8_soak_long.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_soak_long.sh > /tmp/r8_soak_long_exec.sh && rm -f /tmp/r8_soak_long.log && KIND=%K% setsid nohup bash /tmp/r8_soak_long_exec.sh %* > /tmp/r8_soak_long.log 2>&1 < /dev/null &" || goto fail
echo === started on the host; the progress every 2 minutes (Ctrl+C stops the watching, not the soak)
:watch
timeout /t 120 /nobreak >nul
ssh %O% %H% "grep -E '^######## |^   after|SOAKREC|FAILED|Traceback' /tmp/r8_soak_long.log | tail -n 2"
ssh %O% %H% "grep -q '^VERDICT' /tmp/r8_soak_long.log" || (ssh %O% %H% "pgrep -f '[r]8_soak_long_exec.sh' >/dev/null" && goto watch)
scp %O% %H%:/tmp/r8_soak_long.log "%~dp0r8_soak_long.log"
echo.
powershell -NoProfile -Command "Get-Content '%~dp0r8_soak_long.log' | Select-String -Pattern '^model|^qwen|^bitnet|^lfm|^bonsai|^runs with|^  run |^VERDICT|soak done|loadcal'"
echo Log: %~dp0r8_soak_long.log
pause
exit /b 0
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
