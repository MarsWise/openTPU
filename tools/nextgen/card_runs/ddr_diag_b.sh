#!/bin/bash
# Claude, DDR3 diagnosis part B on r5-m8: the channels calibrated again from the host
# (otpu-memcal cal --force: CK phase, write latency, read leveling; the core's CPU held after),
# the read taps centred under load again (loadcal fp4), DRAM scrubbed; then part A's pattern
# test under the TMM load and the decode modes again, to compare. Approved 2026-10-26 ("давай
# все проверим"). Volatile: reloading the bitstream (r5_1_jtag + r5_2_pcie) restores the
# power-up calibration. No bitstream load here.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T
cd "$R" || exit 1
. .venv/bin/activate
echo "== $(date +%H:%M:%S) B1. host calibration (cal --force)"
timeout 900 python -m opentpu.host.memcal cal --force 2>&1 | tail -12
python -m opentpu.host.memcal status 2>&1 | tail -6
echo "== $(date +%H:%M:%S) B2. read taps under load (fp4), scrub"
python - <<'PY' 2>&1 | tail -8
import os
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import loadcal, memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
b.scrub(force=True)
os.environ[loadcal.ENV_KIND] = "fp4"
loadcal.center(b)
b.scrub(force=True)
print("ECC", memcal.ecc(b.t), "temp", b.info().get("temp_c"))
PY
echo "== $(date +%H:%M:%S) B3. pattern test under the TMM load"
timeout 1800 python tools/ddr_diag.py pattern --load ternary:4 --mib 512 --passes 2 2>&1 | tail -5
echo "== $(date +%H:%M:%S) B4. decode modes, ECC after every run"
PYTHONPATH="$R" timeout 3600 "$HOME/hfref/bin/python" tools/ddr_diag.py steps "$MD" --n 32 2>&1 | grep -viE "warn|regex|tokenizer you are" | tail -8
echo "== $(date +%H:%M:%S) B DONE"
