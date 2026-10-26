#!/bin/bash
# Claude, run r5b (after r5a: ECC counters cleared after loadcal, the M sweep, SmolVLA with the
# eltwise kernels and base with chunked attention, vLLM's second engine, the full pytest log;
# COCO done in r5a: 0.2825 / 0.2831). The whole validation of bitstream r5-m8 (build 2f2320e2, m8 137.5 / 275 MHz,
# TMEM 128K: r4 + TFN table functions, VOP MAC,
# QST T4; CFG_HASH 3b7f244e) on the card, after r5_1_jtag.cmd + r5_2_pcie.cmd. No bitstream load, no
# rescan here. Every workload family on the card, against the ISA simulator / frameworks / r3-r4 runs.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/claude-r5b-$TS; mkdir -p "$OUT"
MD=$R/models/bitnet-b1.58-2B-4T; DONE=/tmp/claude_run.done; rm -f $DONE
finish() { echo "== END $(date -Is)"; cp /tmp/claude_run.log "$OUT/run.log" 2>/dev/null
           tar czf /tmp/claude_run.tgz -C "$R/runs" "claude-r5b-$TS" 2>/dev/null; touch $DONE; }
trap finish EXIT
log() { echo; echo "== $(date +%H:%M:%S) $*"; }
show() { grep -v -i "warn" | grep -vE "^\s*$" ; }
FAILS=0; check() { if [ "$1" = 0 ]; then echo "CHECK PASS: $2"; else echo "CHECK FAIL: $2"; FAILS=$((FAILS+1)); fi; }
cd "$R" || exit 1
log "sync $(git rev-parse --short HEAD)"
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature \
  && git merge -q --ff-only refs/remotes/claude/feature || { log "SYNC FAILED"; exit 1; }
git log --oneline -1
H=$HOME/hfref/bin/python; V=$HOME/vllmenv312; LV=$HOME/lrenv312; Y=$HOME/yolox_ref; CO=$HOME/coco
. .venv/bin/activate
for d in /sys/bus/pci/devices/*; do
  [ "$(cat $d/vendor)" = 0x10ee ] && echo "PCIe $d $(cat $d/current_link_speed) x$(cat $d/current_link_width)"
done

log "1. the card: r5's build, CAPS2, CFG_HASH; selftest; every r5 instruction vs the ISA simulator"
python - <<'PY' 2>&1 | tee "$OUT/card.txt"
from opentpu.host.board import Board, XdmaTransport, device_config, check_hash
i = Board(XdmaTransport("/dev/xdma0"), check=False).info()
c = device_config(i)
print("build", hex(i.get("build_id") or 0), "caps2", i.get("caps2"), "TPAIR", c.TPAIR, "TQUAD", c.TQUAD, "TFN", c.TFN)
check_hash(i); print("CFG_HASH ok")
assert (i.get("build_id") or 0) == 0x2f2320e2 and c.TPAIR and c.TQUAD and c.TFN, "not r5"
PY
check ${PIPESTATUS[0]} "the card runs r5 with TPAIR, TQUAD and TFN"
python tools/firmware_contract.py --card 2>&1 | tail -2

log "1b. DRAM after the load: the core's CPU calibrates both channels, then loadcal under load (r4: channel 0's read taps)"
python -m opentpu.host.memcal selfcal 2>&1 | tail -8 | tee "$OUT/selfcal.txt"
check ${PIPESTATUS[0]} "the core's CPU calibrated both channels"
python - <<'PY' 2>&1 | tee "$OUT/loadcal.txt"
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import loadcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
b.scrub(force=True)
r = loadcal.center(b, log=print)
for ch, c in r["channels"].items():
    print("loadcal channel", ch, "taps", [x["tap"] for x in c["lanes"]], "narrow",
          [m for m, x in enumerate(c["lanes"]) if x["narrow"]])
PY
check ${PIPESTATUS[0]} "loadcal centred the read taps under load"
# loadcal sweeps the read taps through failing windows: its ECC counts (r5a: ~9e8 a channel) and
# the sticky ECC_DED are its own; scrub and clear them so the runs below are checked from zero
python - <<'PY' 2>&1 | tee -a "$OUT/loadcal.txt"
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
print("ECC after loadcal", memcal.ecc(b.t))
b.scrub(force=True)
memcal.ecc(b.t, clear=True)
i = b.info()
print("after scrub + clear:", memcal.ecc(b.t), "ECC_DED", bool(i["status"] & (1 << 9)))
PY
check ${PIPESTATUS[0]} "ECC counters cleared after loadcal"
python -m opentpu.host.selftest 2>&1 | tail -16 | tee "$OUT/selftest.txt"; grep -q "ALL PASS" "$OUT/selftest.txt"; check $? "selftest"
python tools/r5_card_check.py 2>&1 | tee "$OUT/r5_card_check.txt"; check ${PIPESTATUS[0]} "r5 instructions on the card == ISA (TFN x10 + edges, MAC, QST T4, TMM PAIR/QUAD, SiLU speed)"

log "2. TMM plain / PAIR / QUAD at M = 1..8 (m8: each column one activation row x a weight block; M = 1 fills 1 / 2 / 4 of the 8)"
# m8: PAIR takes 2M <= MCOLS columns, QUAD 4M: plain M <= 8, PAIR M <= 4, QUAD M <= 2 fill all 8
for M in 1 2 4 8; do
  for m in plain pair quad; do
    [ $m = pair ] && [ $M -gt 4 ] && continue
    [ $m = quad ] && [ $M -gt 2 ] && continue
    echo "-- M=$M $m" | tee -a "$OUT/tmm_bench.txt"
    timeout 600 python tools/bench_tmm.py --m $M --k 2560 --n 1024 --mode $m 2>&1 | tail -4 | tee -a "$OUT/tmm_bench.txt"
    check ${PIPESTATUS[0]} "TMM $m M=$M bit-exact"
  done
done

log "3. YOLOX-Nano: the graph on the card (TFN SiLU / sigmoid, rolling patches), every op vs the ISA; COCO 500 (r3: 0.2821 / 0.44 s, ws 0.2852)"
PYTHONPATH="$R" timeout 3600 "$H" tools/yolox_graph.py "$Y" "$Y/yolox_nano.pth" --image "$Y/assets/dog.jpg" --size 416 --backend board --strict-fpga --check-isa --per-op 2>&1 | show | tail -30 | tee "$OUT/yolox_graph_check.txt"
check ${PIPESTATUS[0]} "YOLOX graph card == ISA"
for ws in; do   # COCO 500 done in r5a (plain 0.2825 at 0.22 s/image, --wsplit 0.2831)
  PYTHONPATH="$R" timeout 7200 "$H" tools/yolox_coco.py "$Y" "$Y/yolox_nano.pth" "$CO" --backend board --limit 500 $ws --out "$OUT/coco_board_500${ws:+_ws}.json" 2>&1 | grep -E "RESULT|s/image|Error|Traceback" | tee -a "$OUT/coco.txt"
done

log "4. SmolVLA: tiny on the card vs ISA (TFN GELU / SiLU, MAC LayerNorm, QST T4), then smolvla_base vs float32"
if "$LV/bin/python" -c "import lerobot" 2>/dev/null; then
  PYTHONPATH="$R" timeout 3600 "$LV/bin/python" tools/smolvla_run.py --tiny --backend board --check-isa 2>&1 | show | tail -12 | tee "$OUT/smolvla_tiny.txt"
  check ${PIPESTATUS[0]} "SmolVLA tiny card == ISA"
  PYTHONPATH="$R" timeout 3600 "$LV/bin/python" tools/smolvla_run.py --tiny --backend board --check-isa --device-rows 2>&1 | show | tail -14 | tee "$OUT/smolvla_tiny_rows.txt"
  check ${PIPESTATUS[0]} "SmolVLA tiny with activations / norms / add / mul on the card == ISA"
  PYTHONPATH="$R" timeout 10800 "$LV/bin/python" tools/smolvla_run.py --policy lerobot/smolvla_base --backend board --save "$OUT/smolvla_base.npz" 2>&1 | show | tail -20 | tee "$OUT/smolvla_base.txt"
  check ${PIPESTATUS[0]} "SmolVLA base ran on the card (attention in IMEM-sized chunks)"
fi

log "5. BitNet 2B4T: generate (TMM QUAD) 32 tokens; chat on the card (strict FPGA)"
PYTHONPATH="$R" timeout 3600 "$H" tools/bitnet_engine.py "$MD" --backend board --tokens 32 --mode generate --save "$OUT/E30_r5_generate.npz" 2>&1 | show | tee "$OUT/bitnet_r5_generate.txt"
PYTHONPATH="$R" timeout 3600 "$H" -m opentpu.host.chat --model "$MD" --backend board --greedy --max-new 64 \
  --cap 1024 --strict-fpga --prompt "What is the capital of France? Answer in one sentence." 2>&1 | show | tail -14 | tee "$OUT/chat_bitnet.txt"
for m in "$R"/models/*/; do
  t=$("$H" -c "import json,sys; print(json.load(open(sys.argv[1]+'/config.json')).get('model_type'))" "$m" 2>/dev/null)
  case "$t" in qwen3|llama|smollm3|lfm2|qwen3_5|qwen3_5_text) ;; *) continue;; esac
  log "otpu-chat: $m ($t)"
  PYTHONPATH="$R" timeout 3600 "$H" -m opentpu.host.chat --model "$m" --backend board --greedy --max-new 64 \
    --cap 1024 --prompt "What is the capital of France? Answer in one sentence." 2>&1 | show | tail -6 | tee -a "$OUT/chat_models.txt"
done

log "6. vLLM on the card"
if "$V/bin/python" -c "import vllm" 2>/dev/null; then
  "$HOME/hfref/bin/uv" pip install -q --python "$V/bin/python" --no-deps -e "$R" 2>&1 | tail -1
  OTPU_VLLM_BACKEND=board PYTHONPATH="$R" timeout 3600 "$V/bin/python" tools/vllm_smoke.py "$MD" --slots 1 --tokens 16 --max-model-len 256 --prompts 2 2>&1 | grep -E "vLLM|MATCH|RESULT|tok|Error|error|Traceback" | tee -a "$OUT/vllm_board.txt"
fi

log "7. upstream's card qualification (qual.sh fast, LOAD=0)"
RUNS=""
for r in qwen3:Qwen3-0.6B lfm2:LFM2.5-230M qwen35:Qwen3.5-0.8B; do
  [ -d "$R/models/${r#*:}" ] && RUNS="$RUNS ${r%%:*}:int8:- ${r%%:*}:fp4:int8"
done
if [ -n "$RUNS" ]; then
  LOAD=0 EXPECT=2f2320e2 PYTHON="$R/.venv/bin/python" RUNS="$RUNS" OUT="$OUT/qual" \
    timeout 7200 bash tools/qual/qual.sh "$R/build-r5" fast 2>&1 | tail -40 | tee "$OUT/qual.txt"
  check ${PIPESTATUS[0]} "qual.sh fast"
fi

log "8. the host's full pytest"
VL=$HOME/verilator-5.046; [ -x "$VL/bin/verilator" ] && export PATH="$VL/bin:$PATH"
XD=""; python -c "import xdist" 2>/dev/null && XD="-n auto"
timeout 14400 python -m pytest -q -p no:cacheprovider $XD -rfE > "$OUT/pytest_full.txt" 2>&1; tail -60 "$OUT/pytest_full.txt"

log "9. selftest at the end, the kernel's xdma errors"
python -m opentpu.host.selftest 2>&1 | tail -16 | tee "$OUT/selftest_end.txt"; grep -q "ALL PASS" "$OUT/selftest_end.txt"; check $? "selftest at the end"
journalctl -k --since "-8h" 2>/dev/null | grep -iE "xdma.*(err|fail|timeout)" | tail -10 | tee "$OUT/xdma_errors.txt"
log "SUMMARY: $FAILS check(s) failed"; grep -h "CHECK " /tmp/claude_run.log | tee "$OUT/checks.txt"
log "ALL DONE $OUT"
