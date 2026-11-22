#!/bin/bash
# Claude: the long DDR3 soak, the TZ's qualification criterion (ECC 0 / 0 on both channels under
# long real load): one calibration (selfcal + loadcal under KIND, default ternary:4), then real
# models in rotation, with no recalibration, until HOURS have passed. Before each run: zeros over
# DRAM (scrub, the ECC counters cleared); after it: that run's ECC counts, DED, temperature and
# the decode rate. One SOAK line a run, the table and the verdict at the end.
#   r8_soak_long.sh [hours] [models...]     default 4 h; qwen35-2b bitnet qwen35 lfm2 qwen3
# Volatile only (the PHY taps, as every bring-up): no bitstream load, no rescan, no mode register.
set -u
HOURS=${1:-4}; shift || true
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python; KIND=${KIND:-ternary:4}
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%F\ %T) $(git log --oneline -1 | cut -c1-90); $HOURS h, loadcal $KIND"
export PYTHONPATH=$R OTPU_MM_K0_REPORT=1
[ $# -gt 0 ] || set -- qwen35-2b bitnet qwen35 lfm2 qwen3
card() { "$R/.venv/bin/python" - "$@" <<'PY'
import sys
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import loadcal, memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False)
what = sys.argv[1]
if what == 'calib':
    b.scrub(force=True)
    r = loadcal.center(b, log=print)
    for ch, c in r['channels'].items():
        print('   loadcal', sys.argv[2], 'channel', ch, 'windows', [x.get('taps') for x in c['lanes']],
              'taps', [x['tap'] for x in c['lanes']], 'narrow', [m for m, x in enumerate(c['lanes']) if x['narrow']])
if what in ('calib', 'scrub'):
    b.scrub(force=True)
i = b.info()
e = memcal.ecc(b.t)
print('   %s: build %s ECC %s ECC_DED %s temp %s' % (what, hex(i.get('build_id') or 0), e,
      bool(i['status'] & (1 << 9)), i.get('temp_c')))
if what == 'after':      # the record the summary reads
    print('SOAKREC %s %s %d %d %d %d %d %s' % (sys.argv[2], sys.argv[3], e[0][0], e[0][1], e[1][0], e[1][1],
          int(bool(i['status'] & (1 << 9))), i.get('temp_c')))
PY
}
# the core's calibration, zeros, loadcal, again until every lane's window under the load is 4 taps
# or more (at most 5 tries): a narrower lane reads wrong words (2026-11-22: lane 7 at 1 tap, DED)
OTPU_LOAD_KIND="$KIND" "$R/.venv/bin/python" -m opentpu.host.loadcal --min-window 4 --tries 5
case $? in 0) ;; 3) echo "   (a lane stays under 4 taps: its words may be wrong; ECC will say)" ;; *) echo "== calibration failed"; exit 1 ;; esac
chat() { timeout 3600 "$P" -m opentpu.host.chat --model "$M" --backend board --strict-fpga --cap 2048 --plain \
  --greedy --max-new 128 --prompt "Explain in three sentences how a DDR3 memory controller calibrates its read timing."; }
END=$(( $(date +%s) + $(awk "BEGIN { print int($HOURS * 3600) }") ))
RECS=$(mktemp /tmp/r8_soak_recs.XXXXXX)
n=0; rc=0
while [ "$(date +%s)" -lt "$END" ]; do
  for m in "$@"; do
    [ "$(date +%s)" -lt "$END" ] || break
    n=$((n + 1))
    case "$m" in bitnet) M=$R/models/bitnet-b1.58-2B-4T ;; bonsai) M=bonsai2-4b ;; *) M=$m ;; esac
    echo; echo "######## $(date +%T) run $n: $m"
    card scrub >/dev/null
    out=$(chat 2>&1); crc=$?
    echo "$out" | grep -E "decode|prefill|TTFT|Traceback|Error" | grep -v "Tokenizer" | tail -4
    rate=$(echo "$out" | sed -nE 's/.*decode [0-9]+ tokens, [0-9.]+ \(device ([0-9.]+)\) tok\/s.*/\1/p' | head -1)
    [ "$crc" = 0 ] || { echo "== $m FAILED (rc $crc)"; rc=1; }
    card after "$n" "$m:${rate:-na}:rc$crc" | tee -a "$RECS"
  done
done
echo; echo "== $(date +%F\ %T) soak done: $n runs"
"$R/.venv/bin/python" - "$RECS" <<'PY'
import collections, sys
rows = [l.split() for l in open(sys.argv[1]) if l.startswith('SOAKREC')]
per = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, []])   # runs, c0, u0, c1, u1, DED runs, rates
bad = []
for _, n, label, c0, u0, c1, u1, ded, temp in rows:
    m, rate, rc = label.split(':')
    c0, u0, c1, u1, ded = map(int, (c0, u0, c1, u1, ded))
    p = per[m]
    p[0] += 1; p[1] += c0; p[2] += u0; p[3] += c1; p[4] += u1; p[5] += ded
    if rate != 'na':
        p[6].append(float(rate))
    if c0 + u0 + c1 + u1 + ded or rc != 'rc0':
        bad.append('run %s %s: ch0 %d / %d, ch1 %d / %d, DED %d, %s, temp %s' % (n, m, c0, u0, c1, u1, ded, rc, temp))
print('model          runs  ch0 corr / uncorr  ch1 corr / uncorr  DED  device decode tok/s')
for m, p in per.items():
    r = ('%.2f-%.2f' % (min(p[6]), max(p[6]))) if p[6] else 'na'
    print('%-13s %5d  %8d / %-8d %8d / %-8d %4d  %s' % (m, p[0], p[1], p[2], p[3], p[4], p[5], r))
print('runs with ECC errors or a failure: %d of %d' % (len(bad), len(rows)))
for b in bad:
    print('  ' + b)
print('VERDICT:', 'CLEAN (ECC 0 / 0 in every run)' if rows and not bad else 'NOT CLEAN' if rows else 'NO RUNS')
PY
exit $rc
