#!/bin/bash
# Claude: where do run_rows' ECC errors come from (run 1, 15:37: the core's reads during the run;
# the host's reads and the stored logits clean) (soak of 2c800a8f, 2026-11-17: ddr_diag rmw and the
# on-card picks clean in 4 passes; run_rows (4 rows, logits to the host) 7 corrected; vLLM one
# uncorrectable)? tools/ddr_diag.py steps, now with run_rows split in phases: the run alone, then
# the host reads the logits three times, the ECC counters after each phase (verdict: the core's reads,
# the host's C2H read path, or stored wrong). Phases now also the rows program WITHOUT the LM head. Per KINDS (default
# "ternary:4 int8:4": loadcal under the BitNet steps' load, then under an int8 LM head's) selfcal +
# loadcal, then the steps PASSES times (default 2). Volatile only; no bitstream load, no rescan, no mode registers. ~25 minutes.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 75; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T; H=$HOME/hfref/bin/python
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1)"
. .venv/bin/activate
for k in ${KINDS:-ternary:4 int8:4}; do
echo; echo "######## $(date +%H:%M:%S) selfcal + loadcal under $k"
python -m opentpu.host.memcal selfcal 2>&1 | tail -3
OTPU_LOAD_KIND=$k python - <<'PY'
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import loadcal, memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
b.scrub(force=True)
r = loadcal.center(b, log=print)
for ch, c in r["channels"].items():
    print("   loadcal channel", ch, "windows", [x.get("taps") for x in c["lanes"]], "taps", [x["tap"] for x in c["lanes"]])
b.scrub(force=True)
i = b.info()
print("== card build", hex(i.get("build_id") or 0), "core_khz", i.get("core_khz"), "temp", i.get("temp_c"), "ECC", memcal.ecc(b.t))
PY
for p in $(seq 1 "${PASSES:-2}"); do
  echo; echo "== $(date +%H:%M:%S) [$k] pass $p: BitNet decode modes + run_rows split"
  PYTHONPATH="$R" timeout 3600 "$H" tools/ddr_diag.py steps "$MD" --n 32 2>&1 | grep -viE "warn|regex|tokenizer you are" | tail -9
done
done
echo "== $(date +%H:%M:%S) DONE"
