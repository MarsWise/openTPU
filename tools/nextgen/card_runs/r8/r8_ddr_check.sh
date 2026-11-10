#!/bin/bash
# Claude: r8 (2c800a8f, 133.33 MHz) DDR3 under the same cases as ddr133_check on r5 45b1c1d3
# (2026-10-31: ECC 0 / 0 everywhere at 133.33 MHz). Run 1 (18:02): no loadcal / fp4 / ternary:4;
# ternary:4 best (vLLM PASS, steps clean) but run_rows (4 rows, int8 head) still read errors.
# Run 2: loadcal under ternary:4, int8:4 (an int8 head's load) and ternary:8, each after the core
# CPU's selfcal and zeros, the same cases, ECC per case. Volatile only (the PHY's taps, as every bring-up);
# no bitstream load, no rescan, no mode-register change. ~60-80 minutes.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 75; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T; V=$HOME/vllmenv312; H=$HOME/hfref/bin/python
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1)"
. .venv/bin/activate
ecc() { python - "$@" <<'PY'
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
if sys.argv[1:] == ["scrub"]:
    b.scrub(force=True)
i = b.info()
print("   build", hex(i.get("build_id") or 0), "core_khz", i.get("core_khz"), "ECC", memcal.ecc(b.t),
      "ECC_DED", bool(i["status"] & (1 << 9)), "temp", i.get("temp_c"))
PY
}
calib() {   # $1: none | fp4 | ternary:4
  python -m opentpu.host.memcal selfcal 2>&1 | tail -3
  OTPU_LOAD_KIND="${1/none/fp4}" python - "$1" <<'PY'
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import loadcal, memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
b.scrub(force=True)
if sys.argv[1] != "none":
    r = loadcal.center(b, log=print)
    for ch, c in r["channels"].items():
        print("   loadcal", sys.argv[1], "channel", ch, "windows", [x.get("taps") for x in c["lanes"]],
              "taps", [x["tap"] for x in c["lanes"]], "narrow", [m for m, x in enumerate(c["lanes"]) if x["narrow"]])
b.scrub(force=True)
print("   calibration", sys.argv[1], "done; ECC", memcal.ecc(b.t))
PY
}
cases() {
  echo; echo "== $(date +%H:%M:%S) [$1] ddr_diag rmw, twice"
  for p in 1 2; do timeout 1800 python tools/ddr_diag.py rmw --iters 200000 2>&1 | tail -11; done
  ecc scrub
  echo; echo "== $(date +%H:%M:%S) [$1] BitNet decode modes, ECC after every run"
  PYTHONPATH="$R" timeout 3600 "$H" tools/ddr_diag.py steps "$MD" --n 32 2>&1 | grep -viE "warn|regex|tokenizer you are" | tail -7
  ecc scrub
  echo; echo "== $(date +%H:%M:%S) [$1] vLLM BitNet 4 slots"
  OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots 4 \
    --tokens 64 --max-model-len 256 --prompts 4 2>&1 | grep -E "^vLLM|MISMATCH|RESULT|Error|Traceback" | grep -v "Tokenizer not available"
  ecc
}
"$HOME/hfref/bin/uv" pip install -q --python "$V/bin/python" --no-deps -e "$R" 2>&1 | tail -1
for k in ${KINDS:-none ternary:4}; do
  echo; echo "######## $(date +%H:%M:%S) calibration: selfcal + ${k/none/no loadcal}"
  calib "$k"
  ecc
  cases "$k"
done
echo "== $(date +%H:%M:%S) DONE"
