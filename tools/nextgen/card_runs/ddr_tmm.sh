#!/bin/bash
# Claude: DDR3 channel 0 under batched BitNet decode on r5-m8 (bitnet_vllm run 2: ECC errors on
# channel 0 only, growing with the load: 2 slots 7 / 1, 4 slots 62 / 17, 8 slots 154 / 48, with
# the read taps loadcal centred in r5a under its fp4 load, channel 0 windows 3-5 taps). Here:
# 1) the read windows under the fp4 load and under a TMM load like the batched steps (4 rows,
# PAIR on all 8 columns), each lane centred under it (loadcal.center: the read IDELAY taps and
# bitslips of the LiteDRAM PHY, what Board.scrub does after every bitstream load; volatile, no
# bitstream change); a lane too narrow under TMM -> back to the fp4 load's centres; 2) BitNet
# through vLLM at 4 and 8 slots again, ECC cleared before and read after each case. No load of
# a bitstream, no rescan. Not during another run.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T; V=$HOME/vllmenv312
TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/ddr-tmm-$TS; mkdir -p "$OUT"
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature
echo "== $(date +%H:%M:%S) $(git log --oneline -1)"
. .venv/bin/activate
ecc() { python - "$@" <<'PY'
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
if sys.argv[1:] == ["clear"]:
    memcal.ecc(b.t, clear=True)
i = b.info(); print("   ECC", memcal.ecc(b.t), "ECC_DED", bool(i["status"] & (1 << 9)), "temp", i.get("temp_c"))
PY
}
echo; echo "== $(date +%H:%M:%S) 1. read windows under the fp4 load, then under TMM (4 rows)"
python - <<'PY' 2>&1 | tee "$OUT/loadcal.txt"
import json, os
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import loadcal, memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
print("ECC before", memcal.ecc(b.t), "temp", b.info().get("temp_c"))
res = {}
for kind in ("fp4", "ternary:4"):
    b.scrub(force=True)                     # zeros: the load reads what it wrote
    os.environ[loadcal.ENV_KIND] = kind
    r = loadcal.center(b)
    res[kind] = r
    for ch in ("0", "1"):
        c = r["channels"][int(ch)] if int(ch) in r["channels"] else r["channels"][ch]
        print(f"{kind} channel {ch}: windows {[x['taps'] for x in c['lanes']]} taps "
              f"{[x['tap'] for x in c['lanes']]} narrow {[m for m, x in enumerate(c['lanes']) if x['narrow']]}")
narrow = any(x["narrow"] for c in res["ternary:4"]["channels"].values() for x in c["lanes"])
if narrow:
    print("a lane too narrow under TMM: the fp4 load's centres again")
    b.scrub(force=True)
    os.environ[loadcal.ENV_KIND] = "fp4"
    loadcal.center(b)
b.scrub(force=True)                         # zeros over the BIST's region, the counters cleared
print("ECC after the calibration", memcal.ecc(b.t), "temp", b.info().get("temp_c"))
json.dump({k: v["channels"] for k, v in res.items()}, open(os.environ.get("OUTJ", "/dev/null"), "w"), default=str)
PY
echo; echo "== $(date +%H:%M:%S) 2. BitNet through vLLM, batched, ECC per case"
"$HOME/hfref/bin/uv" pip install -q --python "$V/bin/python" --no-deps -e "$R" 2>&1 | tail -1
for c in "4 int8" "8 int8" "4 int4"; do
  set -- $c
  ecc clear | tee -a "$OUT/vllm.txt"
  echo "-- slots $1, head $2" | tee -a "$OUT/vllm.txt"
  OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots $1 \
    --tokens 64 --max-model-len 256 --prompts $1 --head-format $2 2>&1 \
    | grep -E "^vLLM|MISMATCH|RESULT|Error|Traceback|DeviceBusy" | tee -a "$OUT/vllm.txt"
  ecc | tee -a "$OUT/vllm.txt"
done
echo "== $(date +%H:%M:%S) DONE $OUT"
