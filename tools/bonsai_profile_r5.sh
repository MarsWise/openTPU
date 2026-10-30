#!/usr/bin/env bash
# Runs inference and writes FPGA DRAM. It does not program FPGA/flash.
# Start only with explicit permission to run the baseline on the existing r5.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
model=${BONSAI_MODEL:-/home/nvx/openTPU_BitNET/models/Ternary-Bonsai-2-4B-mlx-2bit}
python=${OTPU_PYTHON:-/home/nvx/hfref/bin/python}
out=${BONSAI_REPORT:-"$root/build/bonsai-r5-before.json"}
if [[ ! -f "$model/config.json" || ! -x "$python" ]]; then
  echo "Checkpoint or Python not found: $model / $python" >&2
  exit 2
fi
if [[ -e "$out" ]]; then
  echo "Report already exists; set BONSAI_REPORT to a new path: $out" >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$out")"
export OTPU_IMG_CACHE=${OTPU_IMG_CACHE:-"$root/build/bonsai-imgcache"}
export OTPU_PROG_CACHE=${OTPU_PROG_CACHE:-"$root/build/bonsai-progcache"}
exec "$python" "$root/tools/decode_profile.py" --model "$model" --backend board \
  --wformat ternary_gsc --head-format ternary_gsc --card-loop --greedy \
  --no-prompt-runs --cap 512 --tokens 32 --seed 0 \
  --prompt 'Write a short story.' --device-counters \
  --expected-build 0x45b1c1d3 --expected-hash 0xa5bcbaef --json "$out"
