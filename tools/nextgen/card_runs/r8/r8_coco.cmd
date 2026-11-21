@echo off
rem Claude: YOLOX-Nano COCO mAP on the loaded firmware:  r8_coco.cmd [images, default 500]
rem The float model's mAP first (host CPU), then the whole graph on the card; ECC before / after.
rem About 10-20 minutes for 500. Log: r8_coco.log here. No bitstream load, no rescan.
setlocal
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
ssh %O% %H% "pgrep -f '[c]laude_exec.sh|[r]8_soak_long_exec.sh|[r]8_speed_exec.sh' >/dev/null" && (echo Another run is using the card: wait for it to finish first. & pause & exit /b 1)
scp %O% "%~dp0sync.bundle" "%~dp0r8_coco.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_coco.sh > /tmp/r8_coco_exec.sh && bash -o pipefail -c 'bash /tmp/r8_coco_exec.sh %1 2>&1 | tee /tmp/r8_coco.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_coco.log "%~dp0r8_coco.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_coco.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
