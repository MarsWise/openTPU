#!/bin/bash
# Claude: r7 software (MarsWise feature/r7-fpga, 8a4a7ac3 or later) on the LOADED firmware, every model path
# FPGA-only (the host writes inputs, runs programs, reads results). Steps by what the loaded
# firmware has (CAPS2: VU8 -> YOLOX letterbox on the card; TGRQ -> Bonsai GRQ on/off):
#  1 Diffusion Policy resident: tiny card == ISA; lerobot/diffusion_pusht 10 and 100 steps vs float32
#  2 SmolVLA resident: tiny card == ISA; lerobot/smolvla_base vs float32
#  3 YOLOX-Nano strict FPGA-only (VU8) or --strict-fpga --fpga-nms (no VU8)
#  4 vLLM BitNet 4 slots: sampled on the card, greedy, sampled with logprobs (top 5) on the card
#  5 Bonsai 2 4B with OTPU_GRQ=0 / 1 (TGRQ firmware and the model present only)
# The harness checks ECC before and after each step and does not directly clear it.
# Individual backends may initialize/scrub memory and clear counters; see their logs.
set -u
set -o pipefail
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 75; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T; V=$HOME/vllmenv312; LV=$HOME/lrenv312
H=$HOME/hfref/bin/python; Y=$HOME/yolox_ref; BM=$R/models/Ternary-Bonsai-2-4B-mlx-2bit
TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/r7-card-$TS; mkdir -p "$OUT"
finish() { echo; echo "== SUMMARY: $PASSES passed, $FAILS failed, $SKIPS skipped"; echo "== END $(date -Is) ($OUT)"
           cp /tmp/r7_card_tests.log "$OUT/run.log" 2>/dev/null
           tar czf /tmp/r7_card_tests.tgz -C "$R/runs" "r7-card-$TS" 2>/dev/null; }
PASSES=0; FAILS=0; SKIPS=0
trap finish EXIT
log() { echo; echo "== $(date +%H:%M:%S) $*"; }
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
sync_tree() {   # Git refuses to overwrite tracked or untracked user data.
  git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature || return 1
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
from opentpu.host.ecc_guard import require_clean_ecc
b = Board(XdmaTransport("/dev/xdma0"), check=False)
if sys.argv[1:] not in ([], ["clear"]):
    raise ValueError("invalid ECC check arguments")
i = b.info(); e = memcal.ecc(b.t)
print("   ECC", e, "ECC_DED", bool(i["status"] & (1 << 9)), "temp", i.get("temp_c"))
require_clean_ecc(e, i["status"])
PY
}
check_ecc() { ecc "$@" || { fail "ECC unavailable, nonzero, or check failed ($*)"; return 1; }; }
log "0. the card"
python - "$OUT/r7caps.env" <<'PY' 2>&1 | tee "$OUT/card.txt"
import sys
from opentpu.host.board import Board, XdmaTransport, device_config
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info(); c = device_config(i); c2 = i.get("caps2") or {}
print("build", hex(i.get("build_id") or 0), "core_khz", i.get("core_khz"), "caps2", c2)
open(sys.argv[1], "w").write(f"VU8={int(bool(c2.get('vu8')))}\nTGRQ={int(bool(c2.get('tgrq')))}\n")
PY
rc=$?; [ "$rc" = 0 ] && [ -f "$OUT/r7caps.env" ] || { fail "card capability read"; exit 1; }
. "$OUT/r7caps.env"
echo "   VU8=$VU8 TGRQ=$TGRQ"
check_ecc clear || exit 1

log "1a. Diffusion Policy tiny, resident: card == ISA"
PYTHONPATH="$R" timeout 3600 "$LV/bin/python" tools/diffusion_run.py --tiny --backend board --resident --steps 3 --check-isa 2>&1 | tee "$OUT/diffusion_tiny.txt"
rc=$?; [ "$rc" = 0 ] && grep -q "card == ISA: True" "$OUT/diffusion_tiny.txt" && pass "Diffusion tiny resident card == ISA" || fail "Diffusion tiny resident card == ISA"
check_ecc || exit 1
for st in 10 100; do
  log "1b. lerobot/diffusion_pusht resident, $st steps (gate: rel L2 < 0.05 vs float32)"
  check_ecc clear || exit 1
  PYTHONPATH="$R" timeout 7200 "$LV/bin/python" tools/diffusion_run.py --policy lerobot/diffusion_pusht --backend board --resident --steps $st 2>&1 | tee "$OUT/diffusion_pusht_$st.txt"
  rc=$?; [ "$rc" = 0 ] && relok "$OUT/diffusion_pusht_$st.txt" 0.05 && pass "Diffusion pusht $st steps resident" || fail "Diffusion pusht $st steps resident"
  check_ecc || exit 1
done

log "2a. SmolVLA tiny, resident: card == ISA"
check_ecc clear || exit 1
PYTHONPATH="$R" timeout 3600 "$LV/bin/python" tools/smolvla_run.py --tiny --backend board --resident --check-isa 2>&1 | tee "$OUT/smolvla_tiny.txt"
rc=$?; [ "$rc" = 0 ] && grep -q "card == ISA: True" "$OUT/smolvla_tiny.txt" && pass "SmolVLA tiny resident card == ISA" || fail "SmolVLA tiny resident card == ISA"
check_ecc || exit 1
log "2b. lerobot/smolvla_base resident (gate: rel L2 < 0.1 vs float32; first run of the real checkpoint resident)"
check_ecc clear || exit 1
PYTHONPATH="$R" timeout 10800 "$LV/bin/python" tools/smolvla_run.py --policy lerobot/smolvla_base --backend board --resident 2>&1 | tee "$OUT/smolvla_base.txt"
rc=$?; [ "$rc" = 0 ] && relok "$OUT/smolvla_base.txt" 0.1 && pass "SmolVLA base resident" || fail "SmolVLA base resident"
check_ecc || exit 1

log "3. YOLOX-Nano, dog.jpg 416"
check_ecc clear || exit 1
if [ "$VU8" = 1 ]; then
  F="--fpga-resize --fpga-nms --strict-fpga-only"; W="strict FPGA-only (letterbox, NMS on the card)"
else
  F="--strict-fpga --fpga-nms"; W="--strict-fpga --fpga-nms (no VU8 in this firmware: letterbox on the host)"
fi
echo "   $W"
PYTHONPATH="$R" timeout 3600 "$H" tools/yolox_graph.py "$Y" "$Y/yolox_nano.pth" --image "$Y/assets/dog.jpg" --size 416 --backend board $F 2>&1 | tee "$OUT/yolox.txt"
rc=$?; [ "$rc" = 0 ] && grep -q "detections" "$OUT/yolox.txt" && pass "YOLOX $W" || fail "YOLOX $W"
check_ecc || exit 1

log "4. vLLM BitNet 4 slots: sampled on the card, then greedy"
"$HOME/hfref/bin/uv" pip install -q --python "$V/bin/python" --no-deps -e "$R" || { fail "vLLM installation"; exit 1; }
check_ecc clear || exit 1
OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots 4 --tokens 32 --max-model-len 256 --prompts 4 --temperature 0.8 --top-k 40 --top-p 0.95 2>&1 | tee "$OUT/vllm_sampled.txt"
rc=$?; [ "$rc" = 0 ] && grep -q "RESULT PASS" "$OUT/vllm_sampled.txt" && pass "vLLM sampled on the card" || fail "vLLM sampled on the card"
check_ecc || exit 1
OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots 4 --tokens 32 --max-model-len 256 --prompts 4 2>&1 | tee "$OUT/vllm_greedy.txt"
rc=$?; [ "$rc" = 0 ] && grep -q "RESULT PASS" "$OUT/vllm_greedy.txt" && pass "vLLM greedy" || fail "vLLM greedy"
check_ecc || exit 1
check_ecc clear || exit 1
OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots 4 --tokens 16 --max-model-len 256 --prompts 4 --temperature 0.8 --top-k 40 --top-p 0.95 --logprobs 5 2>&1 | tee "$OUT/vllm_logprobs.txt"
rc=$?; [ "$rc" = 0 ] && grep -q "RESULT PASS" "$OUT/vllm_logprobs.txt" && pass "vLLM logprobs on the card" || fail "vLLM logprobs on the card"
check_ecc || exit 1

log "5. Bonsai 2 4B, TMM GSC with GRQ off / on"
if [ "$TGRQ" != 1 ]; then
  skip "Bonsai GRQ: the loaded firmware has no TGRQ (needs the r7 GRQ build)"
elif [ ! -f "$BM/config.json" ]; then
  skip "Bonsai GRQ: no $BM"
else
  REF="$OUT/bonsai-reference"
  mkdir -p "$REF"
  if PYTHONPATH="$R" timeout 7200 "$H" tools/bonsai2_validate.py --mode reference --model "$BM" --out "$REF" 2>&1 | tee "$OUT/bonsai_reference.txt" && [ -f "$REF/reference.npz" ]; then
    for g in 0 1; do
      check_ecc clear || exit 1
      dest="$OUT/bonsai-grq$g"; mkdir -p "$dest"
      cp "$REF/reference.npz" "$dest/reference.npz" || { fail "copy Bonsai reference"; exit 1; }
      OTPU_GRQ=$g PYTHONPATH="$R" timeout 7200 python tools/bonsai2_validate.py --mode board --model "$BM" --out "$dest" 2>&1 | tee "$OUT/bonsai_grq$g.txt"
      rc=$?; [ "$rc" = 0 ] && pass "Bonsai OTPU_GRQ=$g" || fail "Bonsai OTPU_GRQ=$g"
      check_ecc || exit 1
    done
    if [ -f "$OUT/bonsai-grq0/board_logits.npy" ] && [ -f "$OUT/bonsai-grq1/board_logits.npy" ]; then
      python - "$OUT/bonsai-grq0/board_logits.npy" "$OUT/bonsai-grq1/board_logits.npy" <<'PY'
import sys, numpy as np
a, b = (np.load(p) for p in sys.argv[1:])
print('GRQ off/on logits bit-identical:', np.array_equal(a, b))
sys.exit(0 if np.array_equal(a, b) else 1)
PY
      [ "$?" = 0 ] && pass "Bonsai GRQ off/on logits" || fail "Bonsai GRQ off/on logits"
    else
      fail "Bonsai GRQ off/on logits missing"
    fi
  else
    fail "Bonsai CPU reference preparation"
  fi
fi

if [ "$FAILS" -ne 0 ]; then exit 1; fi
