#!/bin/bash
# Claude: PaddleOCR-VL 1.6 on the card (the TZ's VLM: an image-conditioned answer on the device):
# tools/paddleocr_card.py, rendered text lines and any images in /tmp/pocr_img/, against the fp32
# reference on the host CPU; ECC before and after each phase. The model: models/PaddleOCR_VL_1.6
# (model.safetensors, tokenizer.json; the .cmd copies them once). No bitstream load, no rescan.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python; M=$R/models/PaddleOCR_VL_1.6
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%T) $(git log --oneline -1 | cut -c1-90)"
[ -f "$M/model.safetensors" ] && [ -f "$M/tokenizer.json" ] || { echo "== no model in $M"; exit 1; }
"$P" -c "import torch, tokenizers, PIL; print('   python ok: torch', torch.__version__, 'tokenizers', tokenizers.__version__)" || exit 1
[ -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ] || echo "   (no DejaVu Sans: the rendered lines are skipped; images only)"
args=()
if [ -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ]; then
  args+=(--text "Hello openTPU 2026" --text "FPGA Kintex-7 XC7K480T" --text "Invoice 4417: 23.50 EUR")
fi
for f in /tmp/pocr_img/*; do [ -f "$f" ] && args+=(--image "$f"); done
export PYTHONPATH=$R OTPU_MM_K0_REPORT=1

# the calibration: the core's calibration, zeros, loadcal under KIND, again until every lane's
# window under the load is 4 taps or more (at most 5 tries; opentpu.host.loadcal calibrate_until)
KIND=${KIND:-ternary:4}
OTPU_LOAD_KIND="$KIND" "$R/.venv/bin/python" -m opentpu.host.loadcal --min-window 4 --tries 5
case $? in 0) ;; 3) echo "   (a lane stays under 4 taps: its words may be wrong; ECC will say)" ;; *) echo "== calibration failed"; exit 1 ;; esac
timeout 5400 "$P" tools/paddleocr_card.py "$M" --backend board -n 48 "${args[@]}"
rc=$?
echo "== $(date +%T) DONE rc $rc"
exit $rc
