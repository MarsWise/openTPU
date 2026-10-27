#!/usr/bin/env bash
# Native Bonsai 2 4B only. The matching TGSC bitstream must already be programmed.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
python=${OTPU_PYTHON:-"$HOME/hfref/bin/python"}
model="$repo/models/Ternary-Bonsai-2-4B-mlx-2bit"
exec 9>/tmp/claude_run.lock
if ! flock -n 9; then
  echo 'The card is busy. Waiting up to two hours; Ctrl+C cancels the wait.'
  flock -w 7200 9 || exit 75
fi
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OTPU_IMG_CACHE="$repo/build/bonsai2-4b/imgcache"
export OTPU_PROG_CACHE="$repo/build/bonsai2-4b/progcache"
test -f "$model/model.safetensors" || { echo "Missing checkpoint: $model"; exit 1; }
# Read capabilities before loading any image; an old r5 must fail clearly.
"$python" - <<'PY'
from opentpu.host.board import Board, device_config
with Board(calibrate=False) as board:
    cfg = device_config(board.info())
    if not cfg.TGSC:
        raise SystemExit('This card needs the new TGSC firmware. Native Bonsai 2 was not started.')
PY
mode=()
if ! "$python" -c 'import textual' >/dev/null 2>&1; then mode=(--plain); fi
exec "$python" -m opentpu.host.chat --model "$model" --backend board --strict-fpga \
  --wformat ternary_gsc --head-format ternary_gsc --cap 512 --max-new 128 \
  --no-prompt-runs --greedy "${mode[@]}"
