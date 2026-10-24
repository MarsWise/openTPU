#!/bin/bash
# Claude, run 4e: r4 with both DRAM channels clean (run 4d: the core CPU's calibration + loadcal
# under load: 0 ECC events over COCO plain and --wsplit) and the IMEM fix: run 4d's COCO --wsplit
# mAP 0 was YOLOX's stem program, 4765 instructions loaded into a 4096-instruction IMEM (the host
# did not check; now it refuses, and the graph compiler falls back to the row loop: 1061).
# Here: YOLOX --wsplit card vs ISA per op, COCO --wsplit (r3: 0.2852), SmolVLA base vs float32
# (runs 4 / 4b / 4c read it with DRAM errors), ECC + temperature throughout. Nothing programmed.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/claude-r4e-$TS; mkdir -p "$OUT"
DONE=/tmp/claude_run.done; rm -f $DONE
finish() { [ -n "${SAMPLER:-}" ] && kill $SAMPLER 2>/dev/null; echo "== END $(date -Is)"
           cp /tmp/claude_run.log "$OUT/run.log" 2>/dev/null
           tar czf /tmp/claude_run.tgz -C "$R/runs" "claude-r4e-$TS" 2>/dev/null; touch $DONE; }
trap finish EXIT
log() { echo; echo "== $(date +%H:%M:%S) $*"; }
FAILS=0; check() { if [ "$1" = 0 ]; then echo "CHECK PASS: $2"; else echo "CHECK FAIL: $2"; FAILS=$((FAILS+1)); fi; }
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature \
  && git merge -q --ff-only refs/remotes/claude/feature || { log "SYNC FAILED"; exit 1; }
git log --oneline -1
H=$HOME/hfref/bin/python; LV=$HOME/lrenv312; Y=$HOME/yolox_ref; CO=$HOME/coco
. .venv/bin/activate
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

log "1. the card as run 4d left it (loadcal's taps); selftest"
stat start clear
python -m opentpu.host.selftest 2>&1 | tail -16 | tee "$OUT/selftest.txt"; grep -q "ALL PASS" "$OUT/selftest.txt"; check $? "selftest (ECC included)"
stat after-selftest clear

log "2. YOLOX --wsplit: card vs ISA per op (dog.jpg 416)"
PYTHONPATH="$R" timeout 3600 "$H" tools/yolox_graph.py "$Y" "$Y/yolox_nano.pth" --image "$Y/assets/dog.jpg" --size 416 --backend board --strict-fpga --wsplit --check-isa --per-op 2>&1 | grep -viE "warn" | tail -30 | tee "$OUT/yolox_wsplit_check.txt"
check ${PIPESTATUS[0]} "YOLOX --wsplit graph card == ISA"
stat after-wsplit-check clear

log "3. COCO 500 --wsplit"
PYTHONPATH="$R" timeout 7200 "$H" tools/yolox_coco.py "$Y" "$Y/yolox_nano.pth" "$CO" --backend board --limit 500 --wsplit --out "$OUT/coco_wsplit.json" 2>&1 | grep -E "RESULT|s/image|Error|Traceback|ECC" | tee -a "$OUT/coco.txt"
check ${PIPESTATUS[0]} "COCO 500 --wsplit ran"
grep -q "mAP50-95=0.2[0-9]" "$OUT/coco.txt"; check $? "COCO --wsplit mAP50-95 >= 0.20 (r3: 0.2852)"
stat after-coco-wsplit clear

log "4. SmolVLA base vs float32"
if "$LV/bin/python" -c "import lerobot" 2>/dev/null; then
  PYTHONPATH="$R" timeout 10800 "$LV/bin/python" tools/smolvla_run.py --policy lerobot/smolvla_base --backend board --save "$OUT/smolvla_base.npz" 2>&1 | grep -vi warn | tail -14 | tee "$OUT/smolvla_base.txt"
  check ${PIPESTATUS[0]} "SmolVLA base ran"
  stat after-smolvla
fi

log "5. selftest at the end"
python -m opentpu.host.selftest 2>&1 | tail -16 | tee "$OUT/selftest_end.txt"; grep -q "ALL PASS" "$OUT/selftest_end.txt"; check $? "selftest at the end (ECC included)"
log "SUMMARY: $FAILS check(s) failed"; grep -h "CHECK \|^STAT \|RESULT" /tmp/claude_run.log | tee "$OUT/checks.txt"
echo "temperature samples:"; awk 'NR==1||NR%9==0' "$OUT/temp.txt" | tail -30
log "ALL DONE $OUT"
