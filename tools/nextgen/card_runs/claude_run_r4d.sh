#!/bin/bash
# Claude, run 4d: r4's DRAM after run 4c, and COCO --wsplit. Run 4c: loadcal under load cleared
# channel 0 (0 / 0 over three COCO passes, 53-56 C), but it started from the core CPU's stale
# calibration after a host one, so channel 1 lane 5 lost a bit offset (~6e8 corrected a pass);
# and COCO --wsplit gave mAP 0.0000 (r3: 0.2852; the ISA simulator on this tree: right, head
# rel L2 0.0922). Here: the core's CPU recalibrates both channels (a consistent, live baseline),
# loadcal (fixed) under load, then YOLOX --wsplit card vs ISA per op (where it goes wrong, if it
# does), COCO plain and --wsplit; ECC counts + temperature throughout. Nothing is programmed.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/claude-r4d-$TS; mkdir -p "$OUT"
DONE=/tmp/claude_run.done; rm -f $DONE
finish() { [ -n "${SAMPLER:-}" ] && kill $SAMPLER 2>/dev/null; echo "== END $(date -Is)"
           cp /tmp/claude_run.log "$OUT/run.log" 2>/dev/null
           tar czf /tmp/claude_run.tgz -C "$R/runs" "claude-r4d-$TS" 2>/dev/null; touch $DONE; }
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
    b.scrub(force=True)
    memcal.ecc(b.t, clear=True)
    print("STAT counters cleared (scrub)")
PY
}
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

log "1. as run 4c left it; the core's CPU recalibrates both channels; scrub + counters cleared"
stat start clear
python -m opentpu.host.memcal selfcal 2>&1 | tail -8 | tee "$OUT/selfcal.txt"
check ${PIPESTATUS[0]} "the core's CPU recalibrated both channels"
python -m opentpu.host.memcal status 2>&1 | tail -6 | tee "$OUT/memcal_status.txt"
stat after-selfcal clear
python -m opentpu.host.selftest 2>&1 | tail -16 | tee "$OUT/selftest0.txt"; grep -q "ALL PASS" "$OUT/selftest0.txt"; check $? "selftest after the CPU's calibration"
stat after-selftest0

log "2. loadcal under load (from the CPU's live calibration)"
python - <<'PY' 2>&1 | tee "$OUT/loadcal.txt"
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import loadcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
b.scrub(force=True)
r = loadcal.center(b, log=print)
for ch, c in r["channels"].items():
    print("loadcal channel", ch, "taps", [x["tap"] for x in c["lanes"]], "windows",
          [x["taps"] for x in c["lanes"]], "narrow", [m for m, x in enumerate(c["lanes"]) if x["narrow"]],
          "bitslips", c.get("calibrated_bitslips"), "->", c.get("bitslips"))
    print("\n".join(f"  lane {m}: {s}" for m, s in enumerate(c["map"])))
PY
check ${PIPESTATUS[0]} "loadcal centred the read taps under load"
stat after-loadcal clear
python -m opentpu.host.selftest 2>&1 | tail -16 | tee "$OUT/selftest.txt"; grep -q "ALL PASS" "$OUT/selftest.txt"; check $? "selftest (ECC included)"
stat after-selftest clear

log "3. YOLOX --wsplit: card vs ISA per op (dog.jpg 416)"
PYTHONPATH="$R" timeout 3600 "$H" tools/yolox_graph.py "$Y" "$Y/yolox_nano.pth" --image "$Y/assets/dog.jpg" --size 416 --backend board --strict-fpga --wsplit --check-isa --per-op 2>&1 | grep -viE "warn" | tail -40 | tee "$OUT/yolox_wsplit_check.txt"
check ${PIPESTATUS[0]} "YOLOX --wsplit graph card == ISA"
stat after-wsplit-check clear

log "4. COCO 500 plain and --wsplit"
for ws in "" "--wsplit"; do
  nm="${ws:-plain}"
  PYTHONPATH="$R" timeout 7200 "$H" tools/yolox_coco.py "$Y" "$Y/yolox_nano.pth" "$CO" --backend board --limit 500 $ws --out "$OUT/coco_${nm#--}.json" 2>&1 | grep -E "RESULT|s/image|Error|Traceback|ECC" | tee -a "$OUT/coco.txt"
  check ${PIPESTATUS[0]} "COCO 500 $nm ran"
  stat "after $nm" clear
done

log "5. selftest at the end"
python -m opentpu.host.selftest 2>&1 | tail -16 | tee "$OUT/selftest_end.txt"; grep -q "ALL PASS" "$OUT/selftest_end.txt"; check $? "selftest at the end (ECC included)"
log "SUMMARY: $FAILS check(s) failed"; grep -h "CHECK \|^STAT \|RESULT" /tmp/claude_run.log | tee "$OUT/checks.txt"
echo "temperature samples:"; awk 'NR==1||NR%6==0' "$OUT/temp.txt" | tail -30
log "ALL DONE $OUT"
