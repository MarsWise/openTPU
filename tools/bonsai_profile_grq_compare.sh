#!/usr/bin/env bash
# Writes model data to an already-loaded FPGA. Never programs FPGA or flash.
# Requires separate permission before loading any new firmware; run this only
# after the selected bitstream is loaded and PCIe/DDR are initialized.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
model=${BONSAI_MODEL:-/home/nvx/openTPU_BitNET/models/Ternary-Bonsai-2-4B-mlx-2bit}
py=${OTPU_PYTHON:-/home/nvx/hfref/bin/python}
out=${BONSAI_REPORT_DIR:?set BONSAI_REPORT_DIR to a new output directory}
build=${BONSAI_BUILD_ID:-0xb9fe0ca6}
hash=${BONSAI_CFG_HASH:-0xd9b863d5}
if [[ ! -x "$py" || ! -f "$model/config.json" ]]; then
  echo "missing Python or checkpoint" >&2; exit 2
fi
if [[ -e "$out" ]]; then
  echo "refusing to overwrite $out" >&2; exit 2
fi
mkdir -p -- "$out"
for grq in 0 1; do
  OTPU_GRQ="$grq" OTPU_IMG_CACHE="$out/image-cache-$grq" \
    OTPU_PROG_CACHE="$out/program-cache-$grq" "$py" "$root/tools/decode_profile.py" \
    --model "$model" --backend board --wformat ternary_gsc --head-format ternary_gsc \
    --card-loop --greedy --no-prompt-runs --cap 512 --tokens 32 --seed 0 \
    --prompt 'Write a short story.' --device-counters \
    --expected-build "$build" --expected-hash "$hash" --require-tgrq --json "$out/grq-$grq.json"
done
"$py" "$root/tools/compare_bonsai_profiles.py" "$out/grq-0.json" "$out/grq-1.json" \
  | tee "$out/comparison.json"
