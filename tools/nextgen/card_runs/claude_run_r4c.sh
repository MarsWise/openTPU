#!/bin/bash
# Claude, run 4c (v2: the temperature sampler no longer takes the device lock): r4's DRAM channel 0. Run 4b: COCO plain clean (0 / 0), then COCO --wsplit hit
# ECC_DED on channel 0 again (in the end 2,052,213 corrected / 54,084 uncorrectable; channel 1
# 0 / 0); loadcal did not run (its load did not fit a 64K TMEM: fixed in this tree).
# Here, to tell a load-pattern fault from a thermal / time one: the die temperature and both
# channels' ECC counts sampled every 20 s through all of it; loadcal under load; then --wsplit
# FIRST (cold), plain, --wsplit again, each from cleared counters. Nothing is programmed.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/claude-r4c-$TS; mkdir -p "$OUT"
DONE=/tmp/claude_run.done; rm -f $DONE
finish() { [ -n "${SAMPLER:-}" ] && kill $SAMPLER 2>/dev/null; echo "== END $(date -Is)"
           cp /tmp/claude_run.log "$OUT/run.log" 2>/dev/null
           tar czf /tmp/claude_run.tgz -C "$R/runs" "claude-r4c-$TS" 2>/dev/null; touch $DONE; }
trap finish EXIT
log() { echo; echo "== $(date +%H:%M:%S) $*"; }
FAILS=0; check() { if [ "$1" = 0 ]; then echo "CHECK PASS: $2"; else echo "CHECK FAIL: $2"; FAILS=$((FAILS+1)); fi; }
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature \
  && git merge -q --ff-only refs/remotes/claude/feature || { log "SYNC FAILED"; exit 1; }
git log --oneline -1
H=$HOME/hfref/bin/python; Y=$HOME/yolox_ref; CO=$HOME/coco
. .venv/bin/activate
export OTPU_LOAD_CENTER=1
# ECC counts + temperature, optionally clearing the counters (and STATUS ECC_DED with a scrub)
stat() { python - "$1" "${2:-}" <<'PY' 2>&1 | tee -a "$OUT/stat.txt"
import sys, time
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info()
e = memcal.ecc(b.t)
print(f"STAT {time.strftime('%H:%M:%S')} [{sys.argv[1]}] temp {i['temp_c']} C, ECC_DED "
      f"{bool(i['status'] & (1 << 9))}, ch0 corrected/uncorrectable {e[0] if e else None}, "
      f"ch1 {e[1] if e else None}")
if sys.argv[2] == "clear":
    b.scrub(force=True)          # zeros everywhere, STATUS ECC_DED and the counters cleared
    memcal.ecc(b.t, clear=True)
    print("STAT counters cleared (scrub)")
PY
}
# temperature only (one register read), every 20 s, all through the run
python - <<'PY' > "$OUT/temp.txt" 2>&1 &
import time
from opentpu.host.board import XdmaTransport
from opentpu.host import regs as R
t = XdmaTransport("/dev/xdma0", dma=False)   # the register BAR only: no device lock, no DMA
while True:
    v = t.reg_read(R.R_TEMP)
    print(time.strftime("%H:%M:%S"), round(R.temp_c(v), 2) if v & R.TEMP_VALID else None, flush=True)
    time.sleep(20)
PY
SAMPLER=$!

log "1. as run 4b left it; recalibrated from the host; scrub + counters cleared"
stat start clear
python -m opentpu.host.memcal cal --force 2>&1 | tail -12 | tee "$OUT/memcal_cal.txt"
stat after-cal clear

log "2. read taps centred under the core's load (loadcal; the m4 fit fixed)"
python - <<'PY' 2>&1 | tee "$OUT/loadcal.txt"
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import loadcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
b.scrub(force=True)
r = loadcal.center(b, log=print)
print("loadcal:", {k: v for k, v in r.items() if k != "scans"} if isinstance(r, dict) else r)
PY
check ${PIPESTATUS[0]} "loadcal centred the read taps under load"
stat after-loadcal clear
python -m opentpu.host.selftest 2>&1 | tail -16 | tee "$OUT/selftest.txt"; grep -q "ALL PASS" "$OUT/selftest.txt"; check $? "selftest (ECC included)"
stat after-selftest clear

log "3. COCO 500: --wsplit first (cold), plain, --wsplit again; counters cleared before each"
n=0
for ws in "--wsplit" "" "--wsplit"; do
  n=$((n+1)); nm="${ws:-plain}#$n"
  stat "before $nm" clear
  PYTHONPATH="$R" timeout 7200 "$H" tools/yolox_coco.py "$Y" "$Y/yolox_nano.pth" "$CO" --backend board --limit 500 $ws --out "$OUT/coco_${n}.json" 2>&1 | grep -E "RESULT|s/image|Error|Traceback|ECC" | tee -a "$OUT/coco.txt"
  check ${PIPESTATUS[0]} "COCO 500 $nm without an ECC error"
  stat "after $nm"
done

log "4. selftest at the end"
stat end-before clear
python -m opentpu.host.selftest 2>&1 | tail -16 | tee "$OUT/selftest_end.txt"; grep -q "ALL PASS" "$OUT/selftest_end.txt"; check $? "selftest at the end (ECC included)"
log "SUMMARY: $FAILS check(s) failed"; grep -h "CHECK \|^STAT " /tmp/claude_run.log | tee "$OUT/checks.txt"
echo "temperature samples:"; awk 'NR==1||NR%6==0' "$OUT/temp.txt" | tail -40
log "ALL DONE $OUT"
