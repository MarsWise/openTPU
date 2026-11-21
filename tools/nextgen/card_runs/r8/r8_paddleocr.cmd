@echo off
rem Claude: PaddleOCR-VL 1.6 on the card (VLM: an image read into text on the device), against the
rem fp32 reference on the host CPU:  r8_paddleocr.cmd [image files...]
rem Three rendered lines, plus any images given (copied to the host's /tmp/pocr_img). The first run
rem copies the model (model.safetensors 1.9 GB, tokenizer.json) from C:\ai-models\ocr\PaddleOCR_VL_1.6
rem to the host's models\PaddleOCR_VL_1.6 (no code from the model folder). Log: r8_paddleocr.log here.
rem No bitstream load, no rescan. Not during another run on the card.
setlocal
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set "MD=C:\ai-models\ocr\PaddleOCR_VL_1.6"
ssh %O% %H% "pgrep -f '[c]laude_exec.sh|[r]8_soak_long_exec.sh|[r]8_speed_exec.sh' >/dev/null" && (echo Another run is using the card: wait for it to finish first. & pause & exit /b 1)
ssh %O% %H% "test -f ~/openTPU_BitNET/models/PaddleOCR_VL_1.6/model.safetensors" || (
  echo === copying the model to the host, once ^(1.9 GB^)
  ssh %O% %H% "mkdir -p ~/openTPU_BitNET/models/PaddleOCR_VL_1.6" || goto fail
  scp %O% "%MD%\tokenizer.json" "%MD%\config.json" %H%:openTPU_BitNET/models/PaddleOCR_VL_1.6/ || goto fail
  scp %O% "%MD%\model.safetensors" %H%:openTPU_BitNET/models/PaddleOCR_VL_1.6/model.safetensors.part || goto fail
  ssh %O% %H% "mv ~/openTPU_BitNET/models/PaddleOCR_VL_1.6/model.safetensors.part ~/openTPU_BitNET/models/PaddleOCR_VL_1.6/model.safetensors" || goto fail
)
ssh %O% %H% "rm -rf /tmp/pocr_img && mkdir -p /tmp/pocr_img" || goto fail
:imgs
if "%~1"=="" goto run
scp %O% "%~1" %H%:/tmp/pocr_img/ || goto fail
shift
goto imgs
:run
scp %O% "%~dp0sync.bundle" "%~dp0r8_paddleocr.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_paddleocr.sh > /tmp/r8_paddleocr_exec.sh && bash -o pipefail -c 'bash /tmp/r8_paddleocr_exec.sh 2>&1 | tee /tmp/r8_paddleocr.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_paddleocr.log "%~dp0r8_paddleocr.log" || set RUNRC=1
echo.
echo Log: %~dp0r8_paddleocr.log
pause
exit /b %RUNRC%
:fail
echo FAILED (host reachable? ping 192.168.0.157; model in %MD%?)
pause
exit /b 1
