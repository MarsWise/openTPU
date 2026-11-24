@echo off
rem Claude: Kokoro-82M text to speech on the card (every model operation on the device):  r8_kokoro.cmd
rem Copies kokoro-v1_0.pth (312 MB), config.json and voices\af_heart.pt from C:\ai-models\audio\hexgrad--Kokoro-82M
rem to the host once; two phrases; durations, F0 / N and the log-mel spectrogram against the float reference on the
rem host CPU; ECC. Saves r8_kokoro.log and kokoro_1.wav / kokoro_2.wav (and the reference's .ref.wav) here.
rem No bitstream load, no rescan. Not during another run on the card.
setlocal
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set SRC=C:\ai-models\audio\hexgrad--Kokoro-82M
if not exist "%SRC%\kokoro-v1_0.pth" (echo No %SRC%\kokoro-v1_0.pth & pause & exit /b 1)
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
ssh %O% %H% "mkdir -p ~/llm_card/kokoro; test -f ~/llm_card/kokoro/.copied" && goto run
echo Copying Kokoro to the host (once)...
scp %O% "%SRC%\kokoro-v1_0.pth" "%SRC%\config.json" "%SRC%\voices\af_heart.pt" %H%:llm_card/kokoro/ || goto fail
ssh %O% %H% "touch ~/llm_card/kokoro/.copied" || goto fail
:run
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
scp %O% "%~dp0sync.bundle" "%~dp0r8_kokoro.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_kokoro.sh > /tmp/r8_kokoro_exec.sh && bash -o pipefail -c 'bash /tmp/r8_kokoro_exec.sh 2>&1 | tee /tmp/r8_kokoro.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_kokoro.log %H%:/tmp/kokoro_*.wav "%~dp0" || set RUNRC=1
echo.
echo Log: %~dp0r8_kokoro.log   audio: %~dp0kokoro_1.wav, kokoro_2.wav
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host or to copy (is it on? ping 192.168.0.157)
pause
exit /b 1
