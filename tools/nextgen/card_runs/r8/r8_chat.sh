#!/bin/bash
# Claude: an interactive chat on the card (r8), one model: r8_chat.sh <model> [otpu-chat options].
# <model>: bitnet | bonsai | qwen3 | qwen35 | qwen35-2b | lfm2 | reasonlite | any --model name of
# opentpu.llm.MODELS | a checkpoint directory. --strict-fpga always. ECC counters before and after.
# One user of the card at a time (the runs' lock). No bitstream load, no rescan, no calibration.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python
m=${1:-bitnet}; shift || true
case "$m" in
  bitnet) M=$R/models/bitnet-b1.58-2B-4T ;;
  bonsai) M=bonsai2-4b ;;
  *) M=$m ;;
esac
cd "$R" || exit 1
ecc() { PYTHONPATH=$R "$R/.venv/bin/python" - <<'PY'
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport("/dev/xdma0"), check=False)
i = b.info()
print("   card build", hex(i.get("build_id") or 0), "ECC (corrected, uncorrectable) per channel", memcal.ecc(b.t),
      "ECC_DED", bool(i["status"] & (1 << 9)), "temp", i.get("temp_c"))
PY
}
echo "== model $M (tree $(git log --oneline -1 | cut -c1-60))"
ecc
# zeros over DRAM and the ECC counters cleared (as before every step of r8_card_tests): the
# counts printed after the chat are this session's
PYTHONPATH=$R "$R/.venv/bin/python" -c "
from opentpu.host.board import Board, XdmaTransport
Board(XdmaTransport('/dev/xdma0'), check=False).scrub(force=True); print('   scrubbed')"
ecc
T=; "$P" -c 'import textual' 2>/dev/null || { T=--plain; echo "(textual not installed: plain chat)"; }
PYTHONPATH=$R "$P" -m opentpu.host.chat --model "$M" --backend board --strict-fpga --cap 2048 $T "$@"
rc=$?
echo "== chat exit $rc"
ecc
echo "(a nonzero ECC count: this session's answers are not qualified; r8_card_tests.cmd's step 0b scrubs and recalibrates)"
exit $rc
