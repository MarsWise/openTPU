#!/bin/bash
# Claude: check of 2026-10-26's two fixes on r5-m8 (2f2320e2). 1) The ECC errors of batched
# decode (channel 0 only, never under host pattern tests or one sequence's decode): the batched
# step stored each row's pick as its own 4-byte write, two rows' picks in one 64-bit ECC word,
# back to back (the ISA simulator's store trace; tout's first beat is on channel 0). Now one DRAM
# beat a row. Expect: ddr_diag steps with 0 errors; BitNet through vLLM at 2 / 4 / 8 slots MATCH
# with ECC 0 / 0. 2) Diffusion Policy pusht's wide U-Net convs (TMEM split at 64K by the stream
# engine's descriptors; K 20,480 over ACT RAM): the linear kernel's tiles in one TMEM part, K in
# ranges. Expect pusht at 10 and 100 steps against float32. No bitstream load, no rescan, no
# calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T; V=$HOME/vllmenv312; LV=$HOME/lrenv312
TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/fix-check-$TS; mkdir -p "$OUT"
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature
echo "== $(date +%H:%M:%S) $(git log --oneline -1)"
. .venv/bin/activate
ecc() { python - "$@" <<'PY'
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
if sys.argv[1:] == ["scrub"]:
    b.scrub(force=True)
elif sys.argv[1:] == ["clear"]:
    memcal.ecc(b.t, clear=True)
i = b.info(); print("   ECC", memcal.ecc(b.t), "ECC_DED", bool(i["status"] & (1 << 9)), "temp", i.get("temp_c"))
PY
}
ecc scrub
echo; echo "== $(date +%H:%M:%S) 1. decode modes, ECC after every run (expect 0 everywhere)"
PYTHONPATH="$R" timeout 3600 "$HOME/hfref/bin/python" tools/ddr_diag.py steps "$MD" --n 32 2>&1 | grep -viE "warn|regex|tokenizer you are" | tail -7 | tee "$OUT/steps.txt"
echo; echo "== $(date +%H:%M:%S) 2. BitNet through vLLM, batched, ECC per case"
"$HOME/hfref/bin/uv" pip install -q --python "$V/bin/python" --no-deps -e "$R" 2>&1 | tail -1
for c in "2 int8" "4 int8" "8 int8" "4 int4"; do
  set -- $c
  ecc clear | tee -a "$OUT/vllm.txt"
  echo "-- slots $1, head $2" | tee -a "$OUT/vllm.txt"
  OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots $1 \
    --tokens 64 --max-model-len 256 --prompts $1 --head-format $2 2>&1 \
    | grep -E "^vLLM|MISMATCH|RESULT|Error|Traceback|DeviceBusy" | tee -a "$OUT/vllm.txt"
  ecc | tee -a "$OUT/vllm.txt"
done
echo; echo "== $(date +%H:%M:%S) 3. Diffusion Policy pusht on the card"
if "$LV/bin/python" -c "import lerobot, diffusers" 2>/dev/null; then
  for st in 10 100; do
    ecc clear
    PYTHONPATH="$R" timeout 7200 "$LV/bin/python" tools/diffusion_run.py --policy lerobot/diffusion_pusht --backend board \
      --steps $st --save "$OUT/diffusion_pusht_$st.npz" 2>&1 | grep -viE "warn" | tail -14 | tee "$OUT/diffusion_pusht_$st.txt"
    ecc
  done
else
  echo "no lerobot / diffusers in $LV"
fi
echo "== $(date +%H:%M:%S) DONE $OUT"
