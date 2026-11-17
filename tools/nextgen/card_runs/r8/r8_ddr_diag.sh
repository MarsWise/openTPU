#!/bin/bash
# Claude: DDR3 on the loaded firmware (r8, or r5 as the control): is a wrong word written wrong or read
# wrong, and does it take the core's load (tools/ddr_diag.py wr)? After the core CPU's selfcal, then
# after loadcal under ternary:4; the load ternary:4 and int8:4 each. Volatile only (the PHY's read taps,
# as every bring-up); no bitstream load, no rescan, no mode-register change. About 25 minutes.
# loadcal scans each lane's bitslip 0, -1, +1, -2, +2 (a lane calibrated two bit times along: ch1 lane 8).
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 75; }
R=$HOME/openTPU_BitNET
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1)"
. .venv/bin/activate
python - <<'PY'
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info()
print("== card build", hex(i.get("build_id") or 0), "core_khz", i.get("core_khz"), "temp", i.get("temp_c"), "ECC", memcal.ecc(b.t, clear=False))
PY
wr() { for k in ternary:4 int8:4; do
  echo; echo "== $(date +%H:%M:%S) [$1] wr under $k"; timeout 1500 python tools/ddr_diag.py wr --mib "${MIB:-512}" --load "$k" 2>&1 | tail -24
  echo; echo "== $(date +%H:%M:%S) [$1] lanes under $k (BIST: which lanes / bits read wrong)"; timeout 1500 python tools/ddr_diag.py lanes --load "$k" 2>&1 | tail -12
done; }
echo; echo "######## $(date +%H:%M:%S) selfcal, no loadcal"
python -m opentpu.host.memcal selfcal 2>&1 | tail -3
wr selfcal
echo; echo "######## $(date +%H:%M:%S) selfcal + loadcal under ternary:4"
python -m opentpu.host.memcal selfcal 2>&1 | tail -3
OTPU_LOAD_KIND=ternary:4 python - <<'PY'
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import loadcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
b.scrub(force=True)
r = loadcal.center(b, log=print)
for ch, c in r["channels"].items():
    print("   loadcal channel", ch, "windows", [x.get("taps") for x in c["lanes"]], "taps", [x["tap"] for x in c["lanes"]],
          "narrow", [m for m, x in enumerate(c["lanes"]) if x["narrow"]])
b.scrub(force=True)
PY
wr loadcal
echo "== $(date +%H:%M:%S) DONE"
