#!/bin/bash
# Claude: BitNet b1.58 2B4T at full speed on r5-m8 (2f2320e2). Decode is DDR3-bound (~850 MB a token:
# 520 MB ternary weights + 330 MB int8 LM head, ~15 of 17 GB/s): 1) the LM head in int8 / fp4 /
# int4 / nvfp4 (fewer head bytes a token; tokens compared with int8's); 2) vLLM with 1, 2, 4 slots
# (several sequences a decode step: M > 1 uses more of the 8 MXU columns per weight read; tokens
# checked against openTPU's own engine); 3) TMM at M = 1, 2 (QUAD) for the MXU side. No load, no rescan.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== a claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T; H=$HOME/hfref/bin/python; V=$HOME/vllmenv312
TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/bitnet-max-$TS; mkdir -p "$OUT"
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature
echo "== $(date +%H:%M:%S) $(git log --oneline -1)"
. .venv/bin/activate
python - <<'PY'
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info(); print("card build", hex(i.get("build_id") or 0), "core_khz", i.get("core_khz"))
b.scrub(force=True); memcal.ecc(b.t, clear=True); print("ECC cleared:", memcal.ecc(b.t))
PY
PROMPT="Explain in a few sentences why the sky is blue."
echo; echo "== $(date +%H:%M:%S) 1. BitNet generate 128 tokens, LM head int8 / fp4 / int4 / nvfp4"
for hf in int8 fp4 int4 nvfp4; do
  echo "-- head $hf"
  PYTHONPATH="$R" timeout 1800 "$H" tools/bitnet_engine.py "$MD" --backend board --tokens 128 --cap 512 \
    --mode generate --head-format $hf --prompt "$PROMPT" --save "$OUT/gen_$hf.npz" 2>&1 \
    | grep -viE "warn|regex|tokenizer you are" | grep -E "decode|tokens/s|card cycles|output|image built|head|Error|Traceback" | tee -a "$OUT/heads.txt"
done
"$H" - "$OUT" <<'PY'
import sys, numpy as np, glob, os
d = sys.argv[1]; ref = None
for hf in ("int8", "fp4", "int4", "nvfp4"):
    f = os.path.join(d, f"gen_{hf}.npz")
    if not os.path.exists(f):
        print(hf, "no result"); continue
    z = np.load(f); t = next((z[k] for k in ("tokens", "ids", "out", "generated") if k in z), None)
    if t is None:
        print(hf, "keys", list(z.keys())); continue
    t = np.asarray(t).ravel()
    if ref is None: ref = t
    n = min(len(t), len(ref)); same = int(np.argmax(t[:n] != ref[:n])) if (t[:n] != ref[:n]).any() else n
    print(f"head {hf}: first {same} of {n} tokens equal to int8's")
PY
echo; echo "== $(date +%H:%M:%S) 2. vLLM: 1, 2, 4 sequences at once (head int8)"
if "$V/bin/python" -c "import vllm" 2>/dev/null; then
  "$HOME/hfref/bin/uv" pip install -q --python "$V/bin/python" --no-deps -e "$R" 2>&1 | tail -1
  for s in 1 2 4; do
    echo "-- slots $s, prompts $s"
    OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots $s \
      --tokens 32 --max-model-len 256 --prompts $s 2>&1 | grep -E "^vLLM|MATCH|RESULT|Error|Traceback|DeviceBusy" | tee -a "$OUT/vllm.txt"
  done
fi
echo; echo "== $(date +%H:%M:%S) 3. TMM QUAD at M = 1 and 2 (4 / 8 of the m8 columns)"
for M in 1 2; do python tools/bench_tmm.py --m $M --k 2560 --n 1024 --mode quad 2>&1 | tail -3 | tee -a "$OUT/tmm.txt"; done
python - <<'PY'
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info(); print("ECC at the end:", memcal.ecc(b.t), "ECC_DED", bool(i["status"] & (1 << 9)), "temp", i.get("temp_c"))
PY
echo "== $(date +%H:%M:%S) DONE $OUT"
