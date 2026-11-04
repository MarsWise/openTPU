#!/bin/bash
# Claude, DDR3 diagnosis part A on r5-m8 (channel 0's ECC errors under BitNet's batched decode):
# the calibration's state; host pattern tests per channel (random data written and read back
# over PCIe at channel offsets 1-1.5 GiB: OVERWRITES the card's DRAM there; approved 2026-10-26)
# idle, under loadcal's fp4 load and under its TMM load; the decode modes one by one with the
# ECC counters after every run (tools/ddr_diag.py). No calibration change, no bitstream load.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 0; }
R=$HOME/openTPU_BitNET; MD=$R/models/bitnet-b1.58-2B-4T
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature
echo "== $(date +%H:%M:%S) A $(git log --oneline -1)"
. .venv/bin/activate
echo "== $(date +%H:%M:%S) A1. calibration state"; python -m opentpu.host.memcal status 2>&1 | tail -12
for l in none fp4 ternary:4; do
  echo "== $(date +%H:%M:%S) A2. pattern test, load $l"
  timeout 1800 python tools/ddr_diag.py pattern --load $l --mib 512 --passes 2 2>&1 | tail -5
done
echo "== $(date +%H:%M:%S) A3. decode modes, ECC after every run"
PYTHONPATH="$R" timeout 3600 "$HOME/hfref/bin/python" tools/ddr_diag.py steps "$MD" --n 32 2>&1 | grep -viE "warn|regex|tokenizer you are" | tail -8
echo "== $(date +%H:%M:%S) A DONE"
