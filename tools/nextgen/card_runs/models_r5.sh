#!/bin/bash
# Claude: models on r5-m8 (2f2320e2, 137.5 MHz) beyond BitNet. 1) Diffusion Policy (LeRobot): a
# random small one card == ISA, then lerobot/diffusion_pusht (ResNet-18 + 1-D U-Net, DDPM) at 10
# and 100 steps against float32; 2) Qwen3.5-2B (downloaded if missing): upstream's perf method
# (tools/qual/perf.py: 512-token prompt, 64 greedy tokens) in fp4 + int8 head (upstream's card:
# 12.09 tok/s), fp4 + fp4 head, int8 (8.02); a chat answer in fp4 --strict-fpga; the MTP drafter;
# 3) Qwen3-0.6B, Qwen3.5-0.8B, LFM2.5-230M (downloaded if missing) in int8 and fp4 + int8 head
# against upstream's README; 4) SmolVLA tiny card == ISA and smolvla_base vs float32 with the
# activations / norms on the card. No bitstream load, no rescan. Not during another run.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/models-r5-$TS; mkdir -p "$OUT"
finish() { echo "== END $(date -Is)"; cp /tmp/models_r5.log "$OUT/run.log" 2>/dev/null
           tar czf /tmp/models_r5.tgz -C "$R/runs" "models-r5-$TS" 2>/dev/null; }
trap finish EXIT
log() { echo; echo "== $(date +%H:%M:%S) $*"; }
show() { grep -v -i "warn" | grep -vE "^\s*$" ; }
FAILS=0; check() { if [ "$1" = 0 ]; then echo "CHECK PASS: $2"; else echo "CHECK FAIL: $2"; FAILS=$((FAILS+1)); fi; }
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature \
  && git merge -q --ff-only refs/remotes/claude/feature || { log "SYNC FAILED"; exit 1; }
log "tree $(git log --oneline -1)"
H=$HOME/hfref/bin/python; LV=$HOME/lrenv312; UV=$HOME/hfref/bin/uv
. .venv/bin/activate
python - <<'PY' 2>&1 | tee "$OUT/card.txt"
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info(); print("card build", hex(i.get("build_id") or 0), "core_khz", i.get("core_khz"))
b.scrub(force=True); memcal.ecc(b.t, clear=True); print("ECC cleared:", memcal.ecc(b.t))
PY
fetch() {   # HF id, directory under models/
  [ -f "$R/models/$2/config.json" ] && return 0
  echo "downloading $1 -> models/$2"
  "$H" -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1], local_dir=sys.argv[2])" \
    "$1" "$R/models/$2" 2>&1 | tail -2
  [ -f "$R/models/$2/config.json" ]
}

log "1. Diffusion Policy (LeRobot): every conv / linear on the card"
if "$LV/bin/python" -c "import lerobot" 2>/dev/null; then
  "$LV/bin/python" -c "import diffusers" 2>/dev/null || "$UV" pip install -q --python "$LV/bin/python" diffusers 2>&1 | tail -1
  PYTHONPATH="$R" timeout 3600 "$LV/bin/python" tools/diffusion_run.py --tiny --backend board --check-isa 2>&1 | show | tail -14 | tee "$OUT/diffusion_tiny.txt"
  ! grep -q "CARD != ISA" "$OUT/diffusion_tiny.txt" && grep -q "rel L2" "$OUT/diffusion_tiny.txt"; check $? "Diffusion Policy tiny card == ISA"
  for st in 10 100; do
    PYTHONPATH="$R" timeout 7200 "$LV/bin/python" tools/diffusion_run.py --policy lerobot/diffusion_pusht --backend board \
      --steps $st --save "$OUT/diffusion_pusht_$st.npz" 2>&1 | show | tail -16 | tee "$OUT/diffusion_pusht_$st.txt"
    check ${PIPESTATUS[0]} "Diffusion Policy pusht $st steps on the card"
  done
else
  echo "no lerobot in $LV"
fi

log "2. Qwen3.5-2B"
if fetch Qwen/Qwen3.5-2B Qwen3.5-2B; then
  for f in "fp4 int8" "fp4 fp4" "int8 -"; do
    PYTHONPATH="$R" timeout 5400 "$H" tools/qual/perf.py qwen35-2b $f 2>&1 | show | grep -vE "regex|tokenizer you are" | tail -4 | tee -a "$OUT/qwen35_2b_perf.txt"
    check ${PIPESTATUS[0]} "Qwen3.5-2B perf $f"
  done
  PYTHONPATH="$R" timeout 3600 "$H" -m opentpu.host.chat --model qwen35-2b --backend board --greedy --max-new 96 \
    --cap 1024 --wformat fp4 --head-format int8 --strict-fpga \
    --prompt "Explain in a few sentences why the sky is blue." 2>&1 | show | tail -12 | tee "$OUT/qwen35_2b_chat.txt"
  check ${PIPESTATUS[0]} "Qwen3.5-2B chat fp4 --strict-fpga"
  PYTHONPATH="$R" timeout 3600 "$H" -m opentpu.host.chat --model qwen35-2b --backend board --greedy --max-new 96 \
    --cap 1024 --wformat fp4 --head-format int8 --mtp \
    --prompt "Explain in a few sentences why the sky is blue." 2>&1 | show | tail -12 | tee "$OUT/qwen35_2b_mtp.txt"
  check ${PIPESTATUS[0]} "Qwen3.5-2B chat fp4 with the MTP drafter"
else
  echo "Qwen3.5-2B: download failed"; FAILS=$((FAILS+1))
fi

log "3. small LLMs: upstream's perf method, int8 and fp4 + int8 head"
for e in "Qwen/Qwen3-0.6B Qwen3-0.6B qwen3" "Qwen/Qwen3.5-0.8B Qwen3.5-0.8B qwen35" "LiquidAI/LFM2.5-230M LFM2.5-230M lfm2"; do
  set -- $e
  fetch $1 $2 || { echo "$2: download failed"; continue; }
  for f in "int8 -" "fp4 int8"; do
    PYTHONPATH="$R" timeout 3600 "$H" tools/qual/perf.py $3 $f 2>&1 | show | grep -vE "regex|tokenizer you are" | tail -3 | tee -a "$OUT/small_perf.txt"
    check ${PIPESTATUS[0]} "$2 perf $f"
  done
done

log "4. SmolVLA (VLA): tiny card == ISA, smolvla_base with activations / norms on the card"
if "$LV/bin/python" -c "import lerobot" 2>/dev/null; then
  PYTHONPATH="$R" timeout 3600 "$LV/bin/python" tools/smolvla_run.py --tiny --backend board --check-isa --device-rows 2>&1 | show | tail -14 | tee "$OUT/smolvla_tiny_rows.txt"
  check ${PIPESTATUS[0]} "SmolVLA tiny --device-rows card == ISA"
  PYTHONPATH="$R" timeout 10800 "$LV/bin/python" tools/smolvla_run.py --policy lerobot/smolvla_base --backend board --device-rows \
    --save "$OUT/smolvla_base_rows.npz" 2>&1 | show | tail -20 | tee "$OUT/smolvla_base_rows.txt"
  check ${PIPESTATUS[0]} "SmolVLA base on the card"
fi

log "5. the card at the end"
python - <<'PY' 2>&1 | tee "$OUT/card_end.txt"
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info(); print("ECC", memcal.ecc(b.t), "ECC_DED", bool(i["status"] & (1 << 9)), "temp", i.get("temp_c"))
PY
log "SUMMARY: $FAILS check(s) failed"; grep -h "CHECK " /tmp/models_r5.log | tee "$OUT/checks.txt"
log "ALL DONE $OUT"
