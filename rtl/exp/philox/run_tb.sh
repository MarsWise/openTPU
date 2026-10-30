#!/usr/bin/env bash
set -euo pipefail
dir=$(cd -- "$(dirname -- "$0")" && pwd)
obj=$(mktemp -d "${TMPDIR:-/tmp}/otpu-philox.XXXXXXXX")
trap 'rm -rf -- "$obj"' EXIT
verilator --binary --timing --top-module tb_philox --Mdir "$obj" \
  "$dir/otpu_philox4x32.sv" "$dir/tb_philox.sv"
"$obj/Vtb_philox"
