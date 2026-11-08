#!/bin/bash
# Claude: the r7 tests (simulators only, no card access) on the Linux host's CPU: ISA tests, and
# the RTL ones if Verilator is installed (else they report SKIPPED). Each file alone with a time
# limit; a summary line per file. Not during a card run (the same lock).
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another claude run is in progress: not started"; exit 75; }
R=$HOME/openTPU_BitNET
TS=$(date +%Y%m%d-%H%M%S); OUT=$R/runs/r7-sim-$TS; mkdir -p "$OUT"
cd "$R" || exit 1
sync_tree() {   # Git refuses to overwrite tracked or untracked user data.
  git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature || return 1
  git merge -q --ff-only refs/remotes/claude/feature
}
sync_tree || { echo "== SYNC FAILED"; exit 1; }
echo "== tree $(git log --oneline -1)"
. .venv/bin/activate
python -c "import pytest" 2>/dev/null || pip install -q pytest
command -v verilator >/dev/null && echo "== $(verilator --version)" || echo "== no Verilator: RTL tests will be SKIPPED"
K=""; [ "${1:-all}" = isa ] && K="not rtl"      # "isa": the ISA tests only
FAILS=0
for f in tests/test_tmm_grq.py tests/test_resize.py tests/test_letterbox.py tests/test_resident.py \
         tests/test_minicpmv.py tests/test_vllm_devsample.py tests/test_vllm_plugin.py \
         tests/test_batch_steps.py tests/test_diffusion_device_ops.py tests/test_bonsai2.py \
         tests/test_target.py tests/test_host.py tests/test_observability.py \
         tests/test_isa.py tests/test_compiler.py; do
  echo; echo "== $(date +%H:%M:%S) $f"
  n=$(basename "$f" .py)
  if [ -n "$K" ]; then
    timeout 7200 python -m pytest -q -rs "$f" -k "$K" > "$OUT/$n.txt" 2>&1
  else
    timeout 7200 python -m pytest -q -rs "$f" > "$OUT/$n.txt" 2>&1
  fi
  rc=$?
  tail -1 "$OUT/$n.txt"
  [ $rc = 0 ] || { FAILS=$((FAILS+1)); grep -E "^FAILED|^ERROR" "$OUT/$n.txt" | head -10; [ $rc = 124 ] && echo "   TIMEOUT"; }
done
echo; echo "== $FAILS file(s) with failures; logs in $OUT"
tar czf /tmp/r7_sim_tests.tgz -C "$R/runs" "r7-sim-$TS" || exit 1
[ "$FAILS" -eq 0 ]
