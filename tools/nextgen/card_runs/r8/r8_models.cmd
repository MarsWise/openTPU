@echo off
rem Claude: read only: the models on the host, the card's identity and ECC counters. No run, no load.
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
set O=-o UserKnownHostsFile="%KH%" -o ConnectTimeout=15
ssh %O% %H% "cd ~/openTPU_BitNET && echo == models && du -sh models/* 2>/dev/null && echo == other checkpoints && ls -d ~/yolox_ref ~/.cache/huggingface/hub/models--* 2>/dev/null; echo == tree && git log --oneline -1 && echo == card && . .venv/bin/activate && python -c \"from opentpu.host.board import Board, XdmaTransport; from opentpu.host import memcal; b = Board(XdmaTransport('/dev/xdma0'), check=False); i = b.info(); print('build', hex(i.get('build_id') or 0), 'core_khz', i.get('core_khz'), 'ECC', memcal.ecc(b.t), 'temp', i.get('temp_c'))\"" > "%~dp0r8_models.log" 2>&1
type "%~dp0r8_models.log"
echo.
echo Saved: %~dp0r8_models.log
pause
