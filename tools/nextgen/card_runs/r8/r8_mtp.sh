#!/bin/bash
# Claude: MTP decoding on the card for any model with an MTP layer (opentpu/llm/mtp.py; the loop
# on the card). r8_mtp.sh [model ...] (default: qwen35 qwen35-2b; bonsai2-4b needs the
# firmware with CAPS2 TGQ4M, its 2-row GRQ QUAD). Per model the same three prompts greedy: plain
# decode, then --mtp; with FP4=1 also both with 4-bit layers (--wformat fp4 --head-format int8).
# The replies must be equal (MTP is exact for greedy); decode tok/s and MTP acceptance of each.
# Before each run: DRAM zeroed, ECC cleared; ECC after (nonzero: not qualified). No bitstream
# load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python; OUT=/tmp/r8_mtp
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
MODELS=("$@"); [ ${#MODELS[@]} = 0 ] && MODELS=(qwen35 qwen35-2b)
HOWS=(plain mtp); [ "${FP4:-0}" = 1 ] && HOWS=(plain mtp plain4 mtp4)
export PYTHONPATH=$R
# the host's program / image caches (r8_ttft: TTFT 30 s cold -> 0.35-0.6 s warm; the same programs)
export OTPU_PROG_CACHE=$HOME/otpu-cache/prog OTPU_IMG_CACHE=$HOME/otpu-cache/img
export OTPU_PROG_CACHE_FREE_GB=5 OTPU_IMG_CACHE_FREE_GB=5; mkdir -p "$OTPU_PROG_CACHE" "$OTPU_IMG_CACHE"
mkdir -p "$OUT"; rm -f "$OUT"/*.txt
card() { "$R/.venv/bin/python" -c "
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False)
if sys.argv[1] == 'scrub': b.scrub(force=True)
i = b.info()
e = memcal.ecc(b.t)
print('   ' + sys.argv[1] + ': build', hex(i.get('build_id') or 0), 'ECC', e, 'ECC_DED', bool(i['status'] & (1 << 9)), 'temp', i.get('temp_c'), 'caps2', {k: v for k, v in (i.get('caps2') or {}).items() if k.startswith('tgq4')})
sys.exit(1 if sys.argv[1] == 'after' and (any(any(x) for x in e) if isinstance(e, (list, tuple)) else bool(e)) else 0)" "$1"; }
PROMPTS=("Explain in three sentences how a DDR3 memory controller calibrates its read timing."
         "Write a Python function that returns the n-th Fibonacci number, with a short docstring."
         "List five differences between a CPU and a GPU.")
rc=0
for M in "${MODELS[@]}"; do
for i in 0 1 2; do
  for how in "${HOWS[@]}"; do
    X=; case $how in mtp*) X=--mtp ;; esac
    case $how in *4) X="$X --wformat fp4 --head-format int8" ;; esac
    echo; echo "######## $(date +%H:%M:%S) $M prompt $i, $how"
    card scrub
    timeout 3600 "$P" -m opentpu.host.chat --model "$M" --backend board --strict-fpga --cap 2048 --plain \
      --greedy --max-new 128 $X --prompt "${PROMPTS[$i]}" 2>&1 | tee "$OUT/${M}_p${i}_$how.txt"
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "== $M prompt $i $how FAILED"; rc=1; }
    card after || { echo "== ECC nonzero: $M prompt $i $how NOT TRUSTED"; rc=1; }
  done
done
done
echo; echo "######## equal replies (greedy: MTP must give plain decode's tokens)"
python3 - "$OUT" "${HOWS[*]}" "${MODELS[@]}" <<'PY' || rc=1
import sys, re, pathlib
d = pathlib.Path(sys.argv[1]); bad = 0
def reply(p):
    lines = p.read_text(errors="replace").splitlines()
    # (after the placement table: its "first token of a reply" row differs with --mtp)
    i = max(k for k, l in enumerate(lines) if l.startswith(("loaded in", "  first token of a reply")))
    j = max(k for k, l in enumerate(lines) if l.startswith("["))
    return "\n".join(l for l in lines[i + 1:j] if not l.startswith("otpu-chat:")), lines[j]
hows = sys.argv[2].split()
for M in sys.argv[3:]:
  for i in range(3):
    print(f"{M} prompt {i}")
    for base, other in (("plain", "mtp"), ("plain4", "mtp4")):
        if other not in hows:
            continue
        try:
            a, la = reply(d / f"{M}_p{i}_{base}.txt")
            b, lb = reply(d / f"{M}_p{i}_{other}.txt")
        except (ValueError, FileNotFoundError) as e:
            print(f"  {other}: no reply ({e})"); bad = 1; continue
        ok = a == b; bad |= not ok
        print(f"  {base:6s}       {la}")
        print(f"  {other:6s} {'EQUAL' if ok else 'DIFF '} {lb}")
sys.exit(bad)
PY
echo; echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
