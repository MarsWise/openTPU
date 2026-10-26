#!/bin/bash
# Claude: which device DRAM traffic reads back wrong on r5-m8's channel 0 (tools/ddr_diag.py rmw):
# streamed loads with no store / a whole 64-byte beat / a partial beat (read-modify-write) / two
# partial stores to one beat / a 64 KiB tile stored back, each store case on channel 0 and on
# channel 1, the ECC counters per case; twice (repeatability). ~5 minutes. No bitstream load, no
# rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature
echo "== $(date +%H:%M:%S) $(git log --oneline -1)"
. .venv/bin/activate
for pass in 1 2; do
  echo "== $(date +%H:%M:%S) pass $pass"
  timeout 1800 python tools/ddr_diag.py rmw --iters 200000 2>&1 | tail -11
done
echo "== $(date +%H:%M:%S) DONE"
