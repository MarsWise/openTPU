#!/bin/bash
# Claude: the models that ran on the ISA simulator only, on the card, through otpu-chat: one fixed prompt, greedy,
# 64 new tokens, --strict-fpga (no host fallback), TTFT / prefill / decode tok/s, ECC per model. The weights come
# with r8_more_models.cmd (~/llm_card/<name>: safetensors, config and tokenizer files only, no code).
# r8_more_models.sh [reasonlite lfm25-thinking functiongemma qwen35-4b]. Zeros over DRAM before each model.
# No bitstream load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
export PYTHONPATH=$R
card() { "$R/.venv/bin/python" -c "
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False)
if sys.argv[1] == 'scrub': b.scrub(force=True)
i = b.info()
print('   ' + sys.argv[1] + ': build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'ECC_DED', bool(i['status'] & (1 << 9)), 'temp', i.get('temp_c'))" "$1"; }
[ $# -gt 0 ] || set -- reasonlite lfm25-thinking functiongemma qwen35-4b
rc=0
for m in "$@"; do
  M=$HOME/llm_card/$m; X=""
  case "$m" in
    qwen35-4b) X="--wformat fp4" ;;          # 4B: int8 would not leave room for the KV caches in 4 GiB
  esac
  echo; echo "######## $(date +%H:%M:%S) $m ($M) $X"
  [ -f "$M/config.json" ] || { echo "== $m: no $M/config.json (r8_more_models.cmd copies it)"; rc=1; continue; }
  card scrub
  timeout 3600 "$P" -m opentpu.host.chat --model "$M" --backend board --strict-fpga --cap 1024 --plain --greedy \
    --max-new 64 $X --prompt "The capital of France is" || { echo "== $m FAILED (rc $?)"; rc=1; }
  card after
done
echo; echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
