#!/bin/bash
# Claude: speed of the LLMs already on the host, on the loaded firmware: one fixed prompt, greedy, 128
# new tokens, --strict-fpga, opentpu.host.chat --prompt (its summary: TTFT, prefill and decode tok/s,
# wall and device). r8_speed.sh [models...]; default: qwen3 qwen35 qwen35-2b lfm2 bitnet bonsai.
# Before each model: zeros over DRAM (scrub), ECC cleared; after: the ECC counts of that model's run
# (nonzero: the run's numbers are not qualified). No bitstream load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
export PYTHONPATH=$R OTPU_MM_K0_REPORT=1
card() { "$R/.venv/bin/python" -c "
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False)
if sys.argv[1] == 'scrub': b.scrub(force=True)
i = b.info()
print('   ' + sys.argv[1] + ': build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'ECC_DED', bool(i['status'] & (1 << 9)), 'temp', i.get('temp_c'))" "$1"; }
[ $# -gt 0 ] || set -- qwen3 qwen35 qwen35-2b lfm2 bitnet bonsai
rc=0
for m in "$@"; do
  case "$m" in bitnet) M=$R/models/bitnet-b1.58-2B-4T ;; bonsai) M=bonsai2-4b ;; *) M=$m ;; esac
  echo; echo "######## $(date +%H:%M:%S) $m ($M)"
  card scrub
  timeout 3600 "$P" -m opentpu.host.chat --model "$M" --backend board --strict-fpga --cap 2048 --plain \
    --greedy --max-new 128 --prompt "Explain in three sentences how a DDR3 memory controller calibrates its read timing." \
    || { echo "== $m FAILED (rc $?)"; rc=1; }
  card after
done
echo; echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
