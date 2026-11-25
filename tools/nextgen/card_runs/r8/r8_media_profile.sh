#!/bin/bash
# Claude: where Kokoro's and SD's card time goes: every program run on its own (resident.profile_board), the cycles
# summed by op family. Kokoro phrase 1 (graphs A and B), then SD 1.5 at 256 px with one DDIM step (CLIP, two UNet
# evaluations, the VAE). Uses the weights r8_kokoro / r8_sd copied (~/llm_card/kokoro, ~/llm_card/sd15). Zeros over
# DRAM first; ECC before / after. No bitstream load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python; K=$HOME/llm_card/kokoro; S=$HOME/llm_card/sd15
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
export PYTHONPATH=$R:$R/tools
[ -f "$K/conv/kokoro-v1_0.safetensors" ] || { echo "== no converted Kokoro: run r8_kokoro.cmd first"; exit 1; }
[ -f "$S/.copied" ] || [ -f "$S/unet/diffusion_pytorch_model.fp16.safetensors" ] || { echo "== no SD 1.5: run r8_sd.cmd first"; exit 1; }
"$R/.venv/bin/python" -c "
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False); b.scrub(force=True); i = b.info()
print('   scrubbed; build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'temp', i.get('temp_c'))"
rc=0
echo; echo "######## $(date +%H:%M:%S) Kokoro phrase 1, profiled"
timeout 3600 "$P" tools/kokoro_card.py --weights "$K/conv" --config "$K/config.json" --phonemes "həlˈO wˈɜɹld" \
  --out /tmp/kokoro_prof.wav --profile || rc=1
echo; echo "######## $(date +%H:%M:%S) SD 1.5 256 px, 1 step, profiled"
timeout 7200 "$P" tools/sd_card.py --sd "$S" --size 256 --steps 1 --seed 42 --out /tmp/sd_prof.png --profile || rc=1
L=$HOME/llm_card/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors; T=$HOME/llm_card/taesd/diffusion_pytorch_model.safetensors
if [ -f "$L" ] && [ -f "$T" ]; then
  echo; echo "######## $(date +%H:%M:%S) SD 1.5 256 px, LCM-LoRA 4 steps + TAESD, profiled"
  timeout 7200 "$P" tools/sd_card.py --sd "$S" --size 256 --lcm-lora "$L" --taesd "$T" --seed 42 \
    --out /tmp/sd_lcm_prof.png --profile || rc=1
else
  echo; echo "(no LCM-LoRA / TAESD on the host yet: r8_sd.cmd fetches them)"
fi
echo; echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
