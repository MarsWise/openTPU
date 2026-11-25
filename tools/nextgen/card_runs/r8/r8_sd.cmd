@echo off
rem Claude: Stable Diffusion 1.5 text to image on the card (every model operation on the device):  r8_sd.cmd [ddim]
rem Copies C:\ai-models\diffusion\sd-v1-5 (text_encoder, unet, vae, tokenizer, scheduler: ~2 GB) to the host once;
rem the host fetches LCM-LoRA (135 MB) and TAESD (10 MB) once at 2 MB/s. 256 px, LCM-LoRA 4 steps + TAESD, then the
rem image against diffusers (fp32, host CPU) from the same latent; "ddim": also 20 DDIM steps (guidance 7.5) + the VAE.
rem Saves r8_sd.log, sd_lcm.png / sd_lcm.ref.png (and sd_256*.png) here. No bitstream load, no rescan.
setlocal
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
set SRC=C:\ai-models\diffusion\sd-v1-5
if not exist "%SRC%\unet\diffusion_pytorch_model.fp16.safetensors" (echo No %SRC%\unet & pause & exit /b 1)
ssh %O% %H% "pgrep -f '[c]laude_exec.sh' >/dev/null" && (echo A measurement run is in progress on the host: wait for it to finish first. & pause & exit /b 1)
ssh %O% %H% "test -f ~/llm_card/sd15/.copied" && goto run
echo Copying SD 1.5 to the host (once, ~2 GB)...
ssh %O% %H% "mkdir -p ~/llm_card/sd15/text_encoder ~/llm_card/sd15/unet ~/llm_card/sd15/vae ~/llm_card/sd15/tokenizer ~/llm_card/sd15/scheduler" || goto fail
scp %O% "%SRC%\text_encoder\model.fp16.safetensors" "%SRC%\text_encoder\config.json" %H%:llm_card/sd15/text_encoder/ || goto fail
scp %O% "%SRC%\unet\diffusion_pytorch_model.fp16.safetensors" "%SRC%\unet\config.json" %H%:llm_card/sd15/unet/ || goto fail
scp %O% "%SRC%\vae\diffusion_pytorch_model.fp16.safetensors" "%SRC%\vae\config.json" %H%:llm_card/sd15/vae/ || goto fail
scp %O% "%SRC%\tokenizer\vocab.json" "%SRC%\tokenizer\merges.txt" %H%:llm_card/sd15/tokenizer/ || goto fail
scp %O% "%SRC%\scheduler\scheduler_config.json" %H%:llm_card/sd15/scheduler/ || goto fail
ssh %O% %H% "touch ~/llm_card/sd15/.copied" || goto fail
:run
if exist "%~dp0sync_next.bundle" (copy /y "%~dp0sync_next.bundle" "%~dp0sync.bundle" >nul & del "%~dp0sync_next.bundle")
scp %O% "%~dp0sync.bundle" "%~dp0r8_sd.sh" %H%:/tmp/ || goto fail
ssh %O% %H% "tr -d '\r' < /tmp/r8_sd.sh > /tmp/r8_sd_exec.sh && bash -o pipefail -c 'bash /tmp/r8_sd_exec.sh %1 2>&1 | tee /tmp/r8_sd.log'"
set RUNRC=%errorlevel%
scp %O% %H%:/tmp/r8_sd.log %H%:/tmp/sd_*.png "%~dp0." || set RUNRC=1
echo.
echo Log: %~dp0r8_sd.log   image: %~dp0sd_lcm.png (reference sd_lcm.ref.png)
pause
exit /b %RUNRC%
:fail
echo FAILED to reach the host or to copy (is it on? ping 192.168.0.157)
pause
exit /b 1
