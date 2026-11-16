#!/bin/bash
# Claude: Cactus Needle (26M) on the card: tools/needle_card.py with the int8 weights and with the
# CACT CQ4 export (TMM CQ4). Logits card == ISA bit for bit, the README's tool call token for token,
# every token picked on the card (ARGMAX), ECC before / after. The weights come with the .cmd
# (C:\ai-models\cactus\Cactus-Compute--needle -> ~/needle). Zeros over DRAM first (scrub), as before
# every step of r8_card_tests. No bitstream load, no rescan, no calibration change. ~10 minutes.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; N=$HOME/needle; export OTPU_MM_K0_REPORT=1
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
. .venv/bin/activate
[ -f "$N/model.safetensors" ] || { echo "no $N/model.safetensors"; exit 1; }
if [ -f "$N/needle-cq4.zip" ] && [ ! -f "$N/cq4/encoder_layer_0_attn_q.weights" ]; then
  mkdir -p "$N/cq4" && python -m zipfile -e "$N/needle-cq4.zip" "$N/cq4"
fi
scrub() { python -c "
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False); b.scrub(force=True); i = b.info()
print('   scrubbed; build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'temp', i.get('temp_c'))"; }
rc=0
echo; echo "== $(date +%H:%M:%S) Needle int8"; scrub
timeout 1800 python tools/needle_card.py "$N" || rc=1
if [ -f "$N/cq4/encoder_layer_0_attn_q.weights" ]; then
  echo; echo "== $(date +%H:%M:%S) Needle CQ4 (TMM CQ4, the CQ basis on the VPU)"; scrub
  timeout 1800 python tools/needle_card.py "$N" --cq || rc=1
fi
echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
