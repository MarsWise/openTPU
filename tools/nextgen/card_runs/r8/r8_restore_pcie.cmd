@echo off
rem Claude: after r8_load.cmd: the host re-reads the card on PCIe (setup_pcie.sh --rescan: removes
rem the endpoint, rescans, loads the XDMA driver), then prints the card's identity (r8: CFG_HASH
rem 0x93e29b18). Enter the host's sudo password in this window when asked.
setlocal
set KH=C:\Users\user\.copilot\session-state\11d00541-3609-4624-ad4e-77febeaa144f\files\linux-host_known_hosts
set H=nvx@192.168.0.157
ssh -tt -o UserKnownHostsFile="%KH%" -o HostKeyAlgorithms=ssh-ed25519 %H% "sudo bash ~/openTPU_BitNET/opentpu/host/setup_pcie.sh --rescan && cd ~/openTPU_BitNET && . .venv/bin/activate && python -c \"from opentpu.host.board import Board, XdmaTransport; i = Board(XdmaTransport('/dev/xdma0'), check=False).info(); print('build', hex(i.get('build_id') or 0), 'core_khz', i.get('core_khz'), 'cfg hash', hex((i.get('cfg') or {}).get('hash', 0)), 'r8' if (i.get('cfg') or {}).get('hash') == 0x93e29b18 else 'NOT r8')\""
set "RC=%errorlevel%"
echo.
echo PCIe restore exit code: %RC%
pause
exit /b %RC%
