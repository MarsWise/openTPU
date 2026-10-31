#!/bin/bash
# Claude: r7 software (MarsWise feature/r7-fpga, 8a4a7ac3 or later) on the LOADED firmware, every model path
# FPGA-only (the host writes inputs, runs programs, reads results). Steps by what the loaded
# firmware has (CAPS2: VU8 -> YOLOX letterbox on the card; TGRQ -> Bonsai GRQ on/off):
#  1 Diffusion Policy resident: tiny card == ISA; lerobot/diffusion_pusht 10 and 100 steps vs float32
#  2 SmolVLA resident: tiny card == ISA; lerobot/smolvla_base vs float32
#  3 YOLOX-Nano strict FPGA-only (VU8) or --strict-fpga --fpga-nms (no VU8)
#  4 vLLM BitNet 4 slots: sampled on the card, greedy, sampled with logprobs (top 5) on the card
#  5 Bonsai 2 4B with OTPU_GRQ=0 / 1 (TGRQ firmware and the model present only)
# ECC cleared before and read after each step. No bitstream load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T; V=$HOME/vllmenv312; LV=$HOME/lrenv312
H=$HOME/hfref/bin/python; Y=$HOME/yolox_ref; BM=$R/models/Ternary-Bonsai-2-4B-mlx-2bit
TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/r7-card-$TS; mkdir -p "$OUT"
finish() { echo; echo "== SUMMARY: $PASSES passed, $FAILS failed, $SKIPS skipped"; echo "== END $(date -Is) ($OUT)"
           cp /tmp/r7_card_tests.log "$OUT/run.log" 2>/dev/null
           tar czf /tmp/r7_card_tests.tgz -C "$R/runs" "r7-card-$TS" 2>/dev/null; }
PASSES=0; FAILS=0; SKIPS=0
trap finish EXIT
log() { echo; echo "== $(date +%H:%M:%S) $*"; }
show() { grep -viE "warn" | grep -vE "^\s*$"; }
pass() { echo "CHECK PASS: $1"; PASSES=$((PASSES+1)); }
fail() { echo "CHECK FAIL: $1"; FAILS=$((FAILS+1)); }
skip() { echo "CHECK SKIP: $1"; SKIPS=$((SKIPS+1)); }
# rel L2 from a tool's "rel L2 x" line below a bound
relok() { python3 - "$1" "$2" <<'PY'
import re, sys
m = re.findall(r"rel L2 ([0-9.]+)", open(sys.argv[1]).read())
sys.exit(0 if m and float(m[-1]) < float(sys.argv[2]) else 1)
PY
}
cd "$R" || exit 1
sync_tree() {   # fast-forward to the bundle; untracked files the branch adds are MOVED aside, not deleted
  git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature || return 1
  local BK="$R/runs/untracked-moved-$TS" f
  for f in $(git diff --name-only --diff-filter=A HEAD refs/remotes/claude/feature); do
    if [ -e "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      mkdir -p "$BK/$(dirname "$f")" && mv "$f" "$BK/$f" && echo "   moved untracked $f -> $BK/$f (the branch adds it)"
    fi
  done
  git merge -q --ff-only refs/remotes/claude/feature
}
sync_tree || { log "SYNC FAILED (not a fast-forward?)"; exit 1; }
log "tree $(git log --oneline -1)"
git merge-base --is-ancestor 8a4a7ac3702bd4dcd4b0cc825d7139dc332fe719 HEAD || { log "the tree lacks r7 8a4a7ac3: stop"; exit 1; }
. .venv/bin/activate
ecc() { python - "$@" <<'PY'
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
if sys.argv[1:] == ["clear"]:
    memcal.ecc(b.t, clear=True)
i = b.info(); e = memcal.ecc(b.t)
print("   ECC", e, "ECC_DED", bool(i["status"] & (1 << 9)), "temp", i.get("temp_c"))
PY
}
log "0. the card"
python - <<'PY' 2>&1 | tee "$OUT/card.txt"
from opentpu.host.board import Board, XdmaTransport, device_config
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info(); c = device_config(i); c2 = i.get("caps2") or {}
print("build", hex(i.get("build_id") or 0), "core_khz", i.get("core_khz"), "caps2", c2)
open("/tmp/r7caps.env", "w").write(f"VU8={int(bool(c2.get('vu8')))}\nTGRQ={int(bool(c2.get('tgrq')))}\n")
PY
. /tmp/r7caps.env
echo "   VU8=$VU8 TGRQ=$TGRQ"
ecc clear

log "1a. Diffusion Policy tiny, resident: card == ISA"
PYTHONPATH="$R" timeout 3600 "$LV/bin/python" tools/diffusion_run.py --tiny --backend board --resident --steps 3 --check-isa 2>&1 | show | tail -6 | tee "$OUT/diffusion_tiny.txt"
grep -q "card == ISA: True" "$OUT/diffusion_tiny.txt" && pass "Diffusion tiny resident card == ISA" || fail "Diffusion tiny resident card == ISA"
ecc
for st in 10 100; do
  log "1b. lerobot/diffusion_pusht resident, $st steps (gate: rel L2 < 0.05 vs float32)"
  ecc clear
  PYTHONPATH="$R" timeout 7200 "$LV/bin/python" tools/diffusion_run.py --policy lerobot/diffusion_pusht --backend board --resident --steps $st 2>&1 | show | tail -6 | tee "$OUT/diffusion_pusht_$st.txt"
  relok "$OUT/diffusion_pusht_$st.txt" 0.05 && pass "Diffusion pusht $st steps resident" || fail "Diffusion pusht $st steps resident"
  ecc
done

log "2a. SmolVLA tiny, resident: card == ISA"
ecc clear
PYTHONPATH="$R" timeout 3600 "$LV/bin/python" tools/smolvla_run.py --tiny --backend board --resident --check-isa 2>&1 | show | tail -6 | tee "$OUT/smolvla_tiny.txt"
grep -q "card == ISA: True" "$OUT/smolvla_tiny.txt" && pass "SmolVLA tiny resident card == ISA" || fail "SmolVLA tiny resident card == ISA"
ecc
log "2b. lerobot/smolvla_base resident (gate: rel L2 < 0.1 vs float32; first run of the real checkpoint resident)"
ecc clear
PYTHONPATH="$R" timeout 10800 "$LV/bin/python" tools/smolvla_run.py --policy lerobot/smolvla_base --backend board --resident 2>&1 | show | tail -10 | tee "$OUT/smolvla_base.txt"
relok "$OUT/smolvla_base.txt" 0.1 && pass "SmolVLA base resident" || fail "SmolVLA base resident"
ecc

log "3. YOLOX-Nano, dog.jpg 416"
ecc clear
if [ "$VU8" = 1 ]; then
  F="--fpga-resize --fpga-nms --strict-fpga-only"; W="strict FPGA-only (letterbox, NMS on the card)"
else
  F="--strict-fpga --fpga-nms"; W="--strict-fpga --fpga-nms (no VU8 in this firmware: letterbox on the host)"
fi
echo "   $W"
PYTHONPATH="$R" timeout 3600 "$H" tools/yolox_graph.py "$Y" "$Y/yolox_nano.pth" --image "$Y/assets/dog.jpg" --size 416 --backend board $F 2>&1 | show | tail -20 | tee "$OUT/yolox.txt"
[ "${PIPESTATUS[0]}" = 0 ] && grep -q "detections" "$OUT/yolox.txt" && pass "YOLOX $W" || fail "YOLOX $W"
ecc

log "4. vLLM BitNet 4 slots: sampled on the card, then greedy"
"$HOME/hfref/bin/uv" pip install -q --python "$V/bin/python" --no-deps -e "$R" 2>&1 | tail -1
ecc clear
OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots 4 --tokens 32 --max-model-len 256 --prompts 4 --temperature 0.8 --top-k 40 --top-p 0.95 2>&1 | grep -E "^vLLM|MATCH|RESULT|Error|Traceback|Unsupported" | tee "$OUT/vllm_sampled.txt"
grep -q "RESULT PASS" "$OUT/vllm_sampled.txt" && pass "vLLM sampled on the card" || fail "vLLM sampled on the card"
ecc
OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots 4 --tokens 32 --max-model-len 256 --prompts 4 2>&1 | grep -E "^vLLM|MISMATCH|RESULT|Error|Traceback" | tee "$OUT/vllm_greedy.txt"
grep -q "RESULT PASS" "$OUT/vllm_greedy.txt" && pass "vLLM greedy" || fail "vLLM greedy"
ecc
ecc clear
OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots 4 --tokens 16 --max-model-len 256 --prompts 4 --temperature 0.8 --top-k 40 --top-p 0.95 --logprobs 5 2>&1 | grep -E "^vLLM|MATCH|logprobs|RESULT|Error|Traceback|Unsupported" | tee "$OUT/vllm_logprobs.txt"
grep -q "RESULT PASS" "$OUT/vllm_logprobs.txt" && pass "vLLM logprobs on the card" || fail "vLLM logprobs on the card"
ecc

log "5. Bonsai 2 4B, TMM GSC with GRQ off / on"
if [ "$TGRQ" != 1 ]; then
  skip "Bonsai GRQ: the loaded firmware has no TGRQ (needs the r7 GRQ build)"
elif [ ! -f "$BM/config.json" ]; then
  skip "Bonsai GRQ: no $BM"
else
  for g in 0 1; do
    ecc clear
    OTPU_GRQ=$g PYTHONPATH="$R" timeout 7200 python tools/bonsai2_validate.py --mode board --model "$BM" --out "$R/build/bonsai2-4b-grq$g" 2>&1 | show | tail -12 | tee "$OUT/bonsai_grq$g.txt"
    [ "${PIPESTATUS[0]}" = 0 ] && pass "Bonsai OTPU_GRQ=$g" || fail "Bonsai OTPU_GRQ=$g"
    ecc
  done
fi
