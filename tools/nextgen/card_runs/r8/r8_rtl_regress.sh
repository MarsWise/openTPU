#!/bin/bash
# Claude: the full test suite (RTL on Verilator included) of /w/rtl_src.tar.gz (git archive of one commit), inside a
# Linux container (r8_rtl_regress.cmd). Writes regress.log, junit.xml, summary.txt into /w.
# No card, no FPGA, no network but the package installs. r8_rtl_regress.sh [workers]
# REGRESS_W: the folder (default /w); REGRESS_ARGS: extra pytest arguments (a smoke test: test files)
set -u
N=${1:-8}
W=${REGRESS_W:-/w}   # the mounted folder (rtl_src.tar.gz in, the results out)
cd "$W" || exit 1
# everything from here on also goes to regress.log (progress visible from outside)
exec > >(tee "$W/regress.log") 2>&1
MEM=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
echo "== $(date -Is) container: $(uname -m), $(nproc) CPUs, $MEM GiB"
# one compiler job a worker (OTPU_VERILATOR_JOBS) and a worker per 2 GiB: a big tb_top's cc1plus
# takes about 1 GiB, and an out-of-memory kill reads as a failed test
export OTPU_VERILATOR_JOBS=1
MAXN=$(( MEM / 2 )); [ "$MAXN" -ge 1 ] || MAXN=1
[ "$N" -le "$MAXN" ] || { echo "== $N workers need $((2 * N)) GiB: $MAXN workers"; N=$MAXN; }
echo "== $(verilator --version)"
echo "== $(date +%T) checking python / git in the image"
if ! command -v python3 >/dev/null || ! python3 -m venv --help >/dev/null 2>&1 || ! command -v git >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  echo "== $(date +%T) apt-get: python3, venv, pip, git"
  apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip git >/dev/null \
    || { echo "== apt-get failed: no python / git in the container"; exit 2; }
fi
# an existing venv (OTPU_VENV: WSL's ~/otpu_venv of claude_rtl_r8_wsl.cmd) or a new one
if [ -n "${OTPU_VENV:-}" ] && [ -x "$OTPU_VENV/bin/python" ]; then V=$OTPU_VENV; else V=$(mktemp -d)/v; python3 -m venv "$V" || V=; fi
echo "== $(date +%T) pip: numpy, pytest, pytest-xdist in ${V:-(no venv)}"
[ -n "$V" ] && "$V/bin/pip" install -q numpy pytest pytest-xdist \
  || { echo "== pip install failed (no internet, or Ubuntu without python3-venv: sudo apt install python3-venv)"; exit 2; }
echo "== $(date +%T) unpacking the tree"
# the tree: git archive of the commit under test (rtl_src.tar.gz; the commit id is in the tar)
SRC=$(mktemp -d) && mkdir "$SRC/t" && tar -xzf "$W/rtl_src.tar.gz" -C "$SRC/t" || { echo "== cannot unpack rtl_src.tar.gz"; exit 2; }
cd "$SRC/t" || exit 1
# a one-commit repository (tests that ask git about the tree): no history
git init -q && git add -A && git -c user.name=regress -c user.email=regress@localhost commit -qm tree || { echo "== git init failed"; exit 2; }
echo "== tree $(zcat "$W/rtl_src.tar.gz" | git get-tar-commit-id)"
echo "== python $("$V/bin/python" --version 2>&1), workers $N; $(date +%T) pytest (the first Verilator builds take minutes)"
# the build cache lives in the container (fresh: every Verilator model is built once)
export OTPU_VERILATOR_BUILD_DIR=$SRC/vbuild
"$V/bin/python" -m pytest -q -n "$N" -rfEs --durations=30 --junitxml="$W/junit.xml" ${REGRESS_ARGS:-}
rc=$?
"$V/bin/python" - "$W/junit.xml" <<'PY' | tee "$W/summary.txt"
import collections, sys, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot()
s = r if r.tag == 'testsuite' else r.find('testsuite')
print('tests', s.get('tests'), 'failures', s.get('failures'), 'errors', s.get('errors'), 'skipped', s.get('skipped'))
skip = collections.Counter()
for c in s.iter('testcase'):
    for k in ('failure', 'error'):
        e = c.find(k)
        if e is not None:
            oom = 'Killed signal' in (e.text or '') + (e.get('message') or '')
            print(k.upper(), c.get('classname'), c.get('name'),
                  '(OUT OF MEMORY: the compiler was killed; rerun with fewer workers)' if oom else '',
                  (e.get('message') or '')[:160])
    e = c.find('skipped')
    if e is not None:
        skip[(e.get('message') or '')[:100]] += 1
for m, n in skip.most_common():
    print('SKIP', n, m)
PY
echo "== $(date -Is) pytest exit $rc"
exit $rc
