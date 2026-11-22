#!/bin/bash
# Claude: YOLOX-Nano COCO val2017 mAP on the loaded firmware (the TZ's CNN metric; r5 45b1c1d3:
# COCO-500 mAP50-95 0.2825 at 0.22 s/image): tools/yolox_coco.py --backend board (the whole graph
# on the card, its decode too; NMS and the COCO scoring on the host), the first LIMIT images
# (default 500), with the float model's mAP on the same images first. ECC before and after.
# No bitstream load, no rescan.   r8_coco.sh [limit]
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; H=$HOME/hfref/bin/python; Y=$HOME/yolox_ref; CO=$HOME/coco; N=${1:-500}
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%T) $(git log --oneline -1 | cut -c1-90); COCO first $N images"
[ -d "$CO/val2017" ] && [ -f "$Y/yolox_nano.pth" ] || { echo "== no COCO in $CO or no YOLOX in $Y"; exit 1; }
export PYTHONPATH=$R OTPU_MM_K0_REPORT=1

# the calibration: the core's calibration, zeros, loadcal under KIND, again until every lane's
# window under the load is 4 taps or more (at most 5 tries; opentpu.host.loadcal calibrate_until)
KIND=${KIND:-ternary:4}
OTPU_LOAD_KIND="$KIND" "$R/.venv/bin/python" -m opentpu.host.loadcal --min-window 4 --tries 5
case $? in 0) ;; 3) echo "   (a lane stays under 4 taps: its words may be wrong; ECC will say)" ;; *) echo "== calibration failed"; exit 1 ;; esac
ecc() { "$R/.venv/bin/python" -c "
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False)
i = b.info()
print('   ' + sys.argv[1] + ': build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'ECC_DED', bool(i['status'] & (1 << 9)), 'temp', i.get('temp_c'))" "$1"; }
OUT=$R/runs/coco-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
echo; echo "######## $(date +%T) float (host CPU, the reference)"
timeout 7200 "$H" tools/yolox_coco.py "$Y" "$Y/yolox_nano.pth" "$CO" --backend float --limit "$N" --out "$OUT/float.json" 2>&1 \
  | grep -E "RESULT|Average Precision|s/image|Error|Traceback" | head -4
ecc before
echo; echo "######## $(date +%T) the card"
timeout 14400 "$H" tools/yolox_coco.py "$Y" "$Y/yolox_nano.pth" "$CO" --backend board --limit "$N" --out "$OUT/board.json" 2>&1 \
  | grep -E "RESULT|Average Precision|s/image|Error|Traceback|ECC" | head -6
rc=${PIPESTATUS[0]}
ecc after
echo "== $(date +%T) DONE rc $rc (detections in $OUT)"
exit $rc
