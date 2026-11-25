#!/bin/bash
# Claude: Stable Diffusion 1.5 text to image on the card, every model operation on the device (opentpu/sd_graph.py,
# tools/sd_card.py). Default: LCM-LoRA (merged into the UNet as it is read) with 4 LCMScheduler steps, guidance 1 (one
# UNet evaluation a step), the noise drawn on the device, TAESD's decoder; 256 px, then --ref (diffusers fp32 on the
# host CPU from the same latent and noise: cosine, PSNR). "ddim" as the first argument: also the 20-step DDIM run
# with guidance 7.5 and the VAE. LCM-LoRA (openrail++) and TAESD (MIT) are fetched once into ~/llm_card at most
# 2 MB/s, sha256 pinned. Zeros over DRAM first; ECC before / after. No bitstream load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python; S=$HOME/llm_card/sd15
L=$HOME/llm_card/lcm-lora-sdv1-5; T=$HOME/llm_card/taesd
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
rm -f /tmp/sd_*.png

fetch() {   # repo file dir sha256 (or - : a small text file, recorded)
  local f=$3/$2
  mkdir -p "$3"
  if [ -f "$f" ] && { [ "$4" = - ] || echo "$4  $f" | sha256sum -c --status; }; then return 0; fi
  echo "   fetching $1/$2"
  curl -fL --limit-rate 2M -C - --retry 5 --retry-delay 10 -o "$f" "https://huggingface.co/$1/resolve/main/$2" \
    || { echo "   download failed: $1/$2"; return 1; }
  [ "$4" = - ] || echo "$4  $f" | sha256sum -c --status || { echo "   SHA256 MISMATCH: $f"; mv "$f" "$f.bad"; return 1; }
}
fetch latent-consistency/lcm-lora-sdv1-5 pytorch_lora_weights.safetensors "$L" \
  8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4 &&
fetch latent-consistency/lcm-lora-sdv1-5 README.md "$L" - &&
fetch madebyollin/taesd diffusion_pytorch_model.safetensors "$T" \
  db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549 &&
fetch madebyollin/taesd config.json "$T" - &&
fetch madebyollin/taesd README.md "$T" - || { echo "== LCM-LoRA / TAESD not available"; exit 1; }

export PYTHONPATH=$R
"$R/.venv/bin/python" -c "
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False); b.scrub(force=True); i = b.info()
print('   scrubbed; build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'temp', i.get('temp_c'))"
rc=0
PR="${PROMPT:-a photo of a red fox in the snow}"
echo; echo "######## $(date +%H:%M:%S) 256 px, LCM-LoRA ${LCM_STEPS:-4} steps, TAESD, the reference on the host"
timeout 3600 "$P" tools/sd_card.py --sd "$S" --size 256 --lcm-lora "$L/pytorch_lora_weights.safetensors" \
  --taesd "$T/diffusion_pytorch_model.safetensors" --steps ${LCM_STEPS:-4} --prompt "$PR" \
  --seed 42 --out /tmp/sd_lcm.png --ref || rc=1
if [ "${1:-}" = ddim ]; then
  echo; echo "######## $(date +%H:%M:%S) 256 px, DDIM 20 steps, VAE, the reference on the host"
  timeout 7200 "$P" tools/sd_card.py --sd "$S" --size 256 --steps ${STEPS:-20} --prompt "$PR" \
    --seed 42 --out /tmp/sd_256.png --ref || rc=1
fi
echo; echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
