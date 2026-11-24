@echo off
rem Claude: Bonsai 2 4B with its MTP drafter on the card: three prompts greedy, plain, then --mtp (MTP layer int8, then fp4); the replies must be
rem equal; decode tok/s and MTP acceptance of each, ECC per run. Copies Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf (1.27 GB,
rem once) from C:\ai-models. Takes sync_next.bundle first, if it is there. About 30-45 minutes; saves r8_bonsai_mtp.log here.
rem No bitstream load, no rescan. Not during another run on the card.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set B=C:\ai-models\ternary\prism-ml--Ternary-Bonsai-2-4B-gguf
set BF=Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
ssh %O% %H% "mkdir -p ~/openTPU_BitNET/models/bonsai2-4b-mtp; test -f ~/openTPU_BitNET/models/bonsai2-4b-mtp/%BF%" || scp %O% "%B%\%BF%" %H%:openTPU_BitNET/models/bonsai2-4b-mtp/ || goto fail
scp %O% "%~dp0sync.bundle" "%~dp0r8_bonsai_mtp.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_bonsai_mtp.sh > /tmp/r8_bonsai_mtp_exec.sh && bash -o pipefail -c 'bash /tmp/r8_bonsai_mtp_exec.sh 2>&1 | tee /tmp/r8_bonsai_mtp.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_bonsai_mtp.log "%~dp0r8_bonsai_mtp.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_bonsai_mtp.log
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host (is it on? ping 192.168.0.157)
pause
exit /b 1
