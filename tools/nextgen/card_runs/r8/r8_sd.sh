#!/bin/bash
# Claude: Stable Diffusion 1.5 text to image on the card, every model operation on the device (opentpu/sd_graph.py,
# tools/sd_card.py): CLIP, the initial latent from the seed (VOP RAND), DDIM steps with classifier-free guidance,
# the VAE decoder. 256 px, 20 steps; then the same with --ref (diffusers fp32 on the host CPU from the same initial
# latent: cosine, PSNR). The weights come with the .cmd (~/llm_card/sd15: the diffusers folder's fp16 files).
# Zeros over DRAM first; ECC before / after. No bitstream load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python; S=$HOME/llm_card/sd15
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
export PYTHONPATH=$R
"$R/.venv/bin/python" -c "
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False); b.scrub(force=True); i = b.info()
print('   scrubbed; build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'temp', i.get('temp_c'))"
rc=0
echo; echo "######## $(date +%H:%M:%S) 256 px, 20 steps, the reference on the host"
timeout 7200 "$P" tools/sd_card.py --sd "$S" --size 256 --steps ${STEPS:-20} --prompt "${PROMPT:-a photo of a red fox in the snow}" \
  --seed 42 --out /tmp/sd_256.png --ref || rc=1
echo; echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
