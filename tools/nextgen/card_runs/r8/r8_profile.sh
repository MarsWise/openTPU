#!/bin/bash
# Claude: where a decode token's cycles go on the loaded firmware (tools/decode_profile.py
# --device-counters): for each model, the card's counters over the decode of one fixed greedy prompt
# (64 tokens): running cycles a token, DRAM bytes read / written a token, the MXU's busy and MAC
# cycles. r8_profile.sh [models...]; default: bonsai bonsai-grq bitnet qwen35 qwen35-2b
# (bonsai-grq: Bonsai 2 4B with OTPU_GQ4=0, plain GRQ).
# First the DRAM calibration (selfcal + loadcal ternary:4, until every lane's window is 4 taps;
# opentpu.host.loadcal --min-window 4 --tries 5); before each model: zeros over DRAM (scrub).
# decode_profile refuses a run whose ECC counters move. No bitstream load, no rescan.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
export PYTHONPATH=$R
card() { "$R/.venv/bin/python" -c "
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False)
if sys.argv[1] == 'scrub': b.scrub(force=True)
i = b.info()
if sys.argv[1] == 'ids': print(hex(i.get('build_id') or 0), hex(i['cfg']['hash'])); sys.exit()
print('   ' + sys.argv[1] + ': build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'ECC_DED', bool(i['status'] & (1 << 9)), 'temp', i.get('temp_c'))" "$1"; }
read -r BUILD HASH < <(card ids | tail -1)
echo "== card build $BUILD, CFG_HASH $HASH"
echo; echo "######## $(date +%H:%M:%S) calibration (until every lane's window is 4 taps)"
OTPU_LOAD_KIND=ternary:4 "$R/.venv/bin/python" -m opentpu.host.loadcal --min-window 4 --tries 5
crc=$?
[ $crc = 0 ] || echo "== calibration rc $crc (3: a lane still narrower than 4 taps): the profile runs anyway"
[ $# -gt 0 ] || set -- bonsai bonsai-grq bitnet qwen35 qwen35-2b
rc=0
for m in "$@"; do
  env=(); case "$m" in
    bitnet) M=$R/models/bitnet-b1.58-2B-4T ;;
    bonsai) M=bonsai2-4b ;;
    bonsai-grq) M=bonsai2-4b; env=(OTPU_GQ4=0) ;;
    *) M=$m ;;
  esac
  echo; echo "######## $(date +%H:%M:%S) $m ($M ${env[*]})"
  card scrub
  # decode_profile takes the card's lock itself (/tmp/claude_run.lock): this script lends it its own
  flock -u 9
  env "${env[@]}" timeout 3600 "$P" tools/decode_profile.py --model "$M" --backend board --tokens 64 --greedy \
    --device-counters --expected-build "$BUILD" --expected-hash "$HASH" --json "/tmp/prof_$m.json" 9>&-
  prc=$?
  flock -n 9 || { echo "== the card's lock was taken by another run meanwhile: stop"; exit 75; }
  [ $prc = 0 ] || { echo "== $m FAILED (rc $prc)"; rc=1; card after; continue; }
  card after
  "$R/.venv/bin/python" - "/tmp/prof_$m.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c = d.get("device_counters") or {}
t = c.get("tokens") or 0; cyc = c.get("running_cycles_per_token") or 0
rd, wr = c.get("read_bytes_per_token") or 0, c.get("write_bytes_per_token") or 0
khz = c.get("clock_khz") or 133333
mem = (rd + wr) / 128 / cyc if cyc else 0      # the core's DRAM port: 128 bytes a cycle at most
print(f"PROFILE {d.get('model')}: {t} tokens, {cyc / 1e6:.2f} Mcycles a token "
      f"({khz * 1e3 / cyc if cyc else 0:.2f} tok/s on the card); DRAM read {rd / 1e6:.1f} MB, "
      f"write {wr / 1e6:.2f} MB a token = {100 * mem:.0f} % of the port's 128 B/cycle; "
      f"MXU busy {100 * (c.get('mxu_busy_fraction_of_running') or 0):.0f} %, "
      f"MAC {100 * (c.get('mxu_mac_fraction_of_running') or 0):.0f} % of the running cycles; "
      f"port-B requests {c.get('port_b_requests_per_token') or 0:.0f} a token; "
      f"wall decode {c.get('wall_decode_s') or 0:.1f} s, on the card {c.get('device_run_s') or 0:.1f} s")
PY
done
echo; echo "== $(date +%H:%M:%S) DONE rc $rc (JSON: /tmp/prof_*.json, copied next to the log)"
exit $rc
