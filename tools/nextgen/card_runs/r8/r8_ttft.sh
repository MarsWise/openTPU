#!/bin/bash
# Claude: steps 1-2 of docs/nextgen/ARCH_OPTIMIZATION_2026-11-17.md on the loaded r8 firmware, ~20 min:
#  1. TTFT: is the host's program / image cache on? (progcache.cache_dir() is None unless ~/otpu-build
#     exists or OTPU_PROG_CACHE names a directory.) Qwen3-0.6B as r8_speed runs it, then twice with the
#     caches in ~/otpu-cache (cold, then warm): TTFT each time.
#  2. 4-bit layers with an int8 head (the author's card: Qwen3 21.6 -> 31.3 tok/s): Qwen3-0.6B,
#     Qwen3.5-0.8B, LFM2.5-230M, with the warm caches. ECC after every run (nonzero: not qualified).
# No bitstream load, no rescan; writes only ~/otpu-cache.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
export PYTHONPATH=$R
echo "== the caches: ~/otpu-build $( [ -d "$HOME/otpu-build" ] && echo exists || echo 'MISSING (progcache / imgcache off by default)' ); OTPU_PROG_CACHE=${OTPU_PROG_CACHE:-unset}"
df -h "$HOME" | tail -1 | awk '{print "   free on " $6 ": " $4 " of " $2}'
"$R/.venv/bin/python" -c "
from opentpu import progcache, imgcache
print('   progcache.cache_dir():', progcache.cache_dir(), '; imgcache:', getattr(imgcache, 'cache_dir', lambda: '?')())"
card() { "$R/.venv/bin/python" -c "
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False)
if sys.argv[1] == 'scrub': b.scrub(force=True)
i = b.info()
print('   ' + sys.argv[1] + ': build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'ECC_DED', bool(i['status'] & (1 << 9)))" "$1"; }
chat() { timeout 3600 "$P" -m opentpu.host.chat --model "$1" --backend board --strict-fpga --cap 2048 --plain \
  --greedy --max-new 128 "${@:2}" --prompt "Explain in three sentences how a DDR3 memory controller calibrates its read timing." 2>&1 \
  | grep -E "^\[TTFT|weights:|FAILED|Error|Traceback" ; }
run() { echo; echo "######## $(date +%H:%M:%S) $*"; card scrub; chat "$@"; card after; }
echo; echo "#### 1. TTFT"
run qwen3
export OTPU_PROG_CACHE=$HOME/otpu-cache/prog OTPU_IMG_CACHE=$HOME/otpu-cache/img
export OTPU_PROG_CACHE_FREE_GB=${FREE_GB:-5} OTPU_IMG_CACHE_FREE_GB=${FREE_GB:-5}
mkdir -p "$OTPU_PROG_CACHE" "$OTPU_IMG_CACHE"
echo "   caches now: $OTPU_PROG_CACHE, $OTPU_IMG_CACHE (free floor ${FREE_GB:-5} GB)"
run qwen3          # cold: fills the caches
run qwen3          # warm
echo; echo "#### 2. 4-bit layers, int8 head (warm caches after the first run of each)"
for m in qwen3 qwen35 lfm2; do
  run "$m" --wformat fp4 --head-format int8
  run "$m" --wformat fp4 --head-format int8
done
du -sh "$HOME/otpu-cache" 2>/dev/null
echo "== $(date +%H:%M:%S) DONE"
