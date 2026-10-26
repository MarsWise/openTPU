#!/bin/bash
# Claude: BitNet b1.58 2B4T through vLLM on r5-m8 with the decode steps batched over the KV slots
# (worker: BitNet no longer limited to one slot; each step's decode rows go through one run, every
# TMM over all of them, M = slots). DDR3-bound: the weights are read once a step for all rows, so
# the total tokens/s should grow with the slots. Slots 1 / 2 / 4 / 8 with the int8 LM head, 4 with
# int4; each request's greedy tokens checked against openTPU's own engine. No load, no rescan.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== a claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T; V=$HOME/vllmenv312
TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/bitnet-vllm-$TS; mkdir -p "$OUT"
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature
echo "== $(date +%H:%M:%S) $(git log --oneline -1)"
"$V/bin/python" -c "import vllm" 2>/dev/null || { echo "no vLLM in $V"; exit 1; }
"$HOME/hfref/bin/uv" pip install -q --python "$V/bin/python" --no-deps -e "$R" 2>&1 | tail -1
run() {   # slots head
  echo "-- slots $1, prompts $1, head $2"
  OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots $1 \
    --tokens 32 --max-model-len 256 --prompts $1 --head-format $2 2>&1 \
    | grep -E "^vLLM|MATCH|RESULT|Error|Traceback|DeviceBusy|KV slot" | tee -a "$OUT/vllm.txt"
}
for s in 1 2 4 8; do run $s int8; done
run 4 int4
. .venv/bin/activate
python - <<'PY'
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info(); print("card build", hex(i.get("build_id") or 0), "ECC", memcal.ecc(b.t), "ECC_DED", bool(i["status"] & (1 << 9)), "temp", i.get("temp_c"))
PY
echo "== $(date +%H:%M:%S) DONE $OUT"
