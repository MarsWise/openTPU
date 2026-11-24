#!/bin/bash
# Claude: Bonsai 2 4B with its MTP drafter on the card (opentpu/llm/mtp.py; the MTP layer from
# models/bonsai2-4b-mtp/*MTP*.gguf, Q8_0 -> int8, the model from the MLX pack). The same prompts
# greedy, plain decode, then --mtp with the MTP layer int8, then fp4 (OTPU_BONSAI_MTP_FORMAT=fp4:
# half the bytes a draft reads): the replies must be equal (MTP is exact for greedy); decode tok/s
# and MTP acceptance of each. Before each run: DRAM zeroed, ECC cleared; ECC after (nonzero:
# not qualified). No bitstream load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python; OUT=/tmp/r8_bonsai_mtp
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
ls "$R"/models/bonsai2-4b-mtp/*MTP*.gguf || { echo "no MTP GGUF in $R/models/bonsai2-4b-mtp"; exit 1; }
export PYTHONPATH=$R
mkdir -p "$OUT"; rm -f "$OUT"/*.txt
card() { "$R/.venv/bin/python" -c "
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False)
if sys.argv[1] == 'scrub': b.scrub(force=True)
i = b.info()
e = memcal.ecc(b.t)
print('   ' + sys.argv[1] + ': build', hex(i.get('build_id') or 0), 'ECC', e, 'ECC_DED', bool(i['status'] & (1 << 9)), 'temp', i.get('temp_c'))
sys.exit(1 if sys.argv[1] == 'after' and (any(any(x) for x in e) if isinstance(e, (list, tuple)) else bool(e)) else 0)" "$1"; }
PROMPTS=("Explain in three sentences how a DDR3 memory controller calibrates its read timing."
         "Write a Python function that returns the n-th Fibonacci number, with a short docstring."
         "List five differences between a CPU and a GPU.")
rc=0
for i in 0 1 2; do
  for how in plain mtp mtp4; do
    X=; [ $how != plain ] && X=--mtp
    export OTPU_BONSAI_MTP_FORMAT=int8; [ $how = mtp4 ] && export OTPU_BONSAI_MTP_FORMAT=fp4
    echo; echo "######## $(date +%H:%M:%S) prompt $i, $how"
    card scrub
    timeout 3600 "$P" -m opentpu.host.chat --model bonsai2-4b --backend board --strict-fpga --cap 2048 --plain \
      --greedy --max-new 128 $X --prompt "${PROMPTS[$i]}" 2>&1 | tee "$OUT/p${i}_$how.txt"
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "== prompt $i $how FAILED"; rc=1; }
    card after || { echo "== ECC nonzero: prompt $i $how NOT TRUSTED"; rc=1; }
  done
done
echo; echo "######## equal replies (greedy: MTP must give plain decode's tokens)"
python3 - "$OUT" <<'PY' || rc=1
import sys, re, pathlib
d = pathlib.Path(sys.argv[1]); bad = 0
def reply(p):
    lines = p.read_text(errors="replace").splitlines()
    i = max(k for k, l in enumerate(lines) if l.startswith("loaded in"))
    j = max(k for k, l in enumerate(lines) if l.startswith("["))
    return "\n".join(l for l in lines[i + 1:j] if not l.startswith("otpu-chat:")), lines[j]
for i in range(3):
    try:
        a, la = reply(d / f"p{i}_plain.txt")
    except (ValueError, FileNotFoundError) as e:
        print(f"prompt {i}: no plain reply ({e})"); bad = 1; continue
    print(f"prompt {i}\n  plain       {la}")
    for how in ("mtp", "mtp4"):
        try:
            b, lb = reply(d / f"p{i}_{how}.txt")
        except (ValueError, FileNotFoundError) as e:
            print(f"  {how}: no reply ({e})"); bad = 1; continue
        ok = a == b; bad |= not ok
        print(f"  {how:5s} {'EQUAL' if ok else 'DIFF '} {lb}")
sys.exit(bad)
PY
echo; echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
