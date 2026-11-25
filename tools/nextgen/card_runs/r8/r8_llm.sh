#!/bin/bash
# Claude: a real LLM that ran on the ISA simulator only, on the card: tools/llm_card.py
# (r8_llm.sh lfm3 | lfm3-fp4 | minicpm5 | granite | maple | ouro | dory). The prompt's logits against the fp32 reference (host CPU,
# a check), greedy tokens, tok/s, ECC before / after. Zeros over DRAM first (scrub). The weights
# come with the .cmd (~/llm_card/<model>). No bitstream load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; m=${1:-lfm3}; W=$HOME/llm_card/$m
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
P=$HOME/hfref/bin/python; "$P" -c 'import tokenizers, numpy' 2>/dev/null || P=$R/.venv/bin/python
export PYTHONPATH=$R OTPU_MM_K0_REPORT=1
# the host's program / image caches (r8_ttft: TTFT 30 s cold -> 0.35-0.6 s warm; the same programs)
export OTPU_PROG_CACHE=$HOME/otpu-cache/prog OTPU_IMG_CACHE=$HOME/otpu-cache/img
export OTPU_PROG_CACHE_FREE_GB=5 OTPU_IMG_CACHE_FREE_GB=5; mkdir -p "$OTPU_PROG_CACHE" "$OTPU_IMG_CACHE"
"$R/.venv/bin/python" -c "
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False); b.scrub(force=True); i = b.info()
print('   scrubbed; build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'temp', i.get('temp_c'))"
case "$m" in
  lfm3)     set -- lfm3 "$W" ;;
  lfm3-fp4) W=$HOME/llm_card/lfm3; set -- lfm3 "$W" --wformat fp4 --cos 0.9 ;;   # (4-bit layers)
  minicpm5) set -- gguf "$W/MiniCPM5-2B-Q4_K_M.gguf" --cos 0.998 ;;
  maple)    W=$HOME/openTPU_BitNET/models/maple-preview   # (r8_fetch_models: the TQ2_0 GGUF)
            set -- maple "$W/maple-preview-TQ2_0-head-F16.gguf" --wformat tgsc --head-format int8 \
                   --experts ${EXPERTS:-128} --new ${NEW:-16} --cos 0.95 ;;   # ternary MoE, experts streamed
  ouro)     set -- ouro "$W" --cos 0.94 ;;                       # Ouro-1.4B-Thinking, 4 passes
  dory)     set -- dory "$W/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf" --wformat mix \
                   --formats gateup=fp4 --head-format fp4 --cap 128 --cos 0.95 ;;   # looped nGPT. int8: 4446 MiB, more
            # than a slice's 4096; fp4 down projections: the decode program is 33424 words, more than
            # IMEM's 32768. fp4 gate / up, int8 down at cap 256: 4254 MiB; + fp4 head, cap 128: less (the
            # MemoryError, if any, now lists the parts), the program 29968 words
  granite)  set -- gguf "$W/granite-4.2-3b-Q4_K_M.gguf" --wformat mix --formats gateup=nvfp4,down=fp4 --cos 0.85 ;;
  *) echo "model: lfm3 | lfm3-fp4 | minicpm5 | granite | maple | ouro | dory"; exit 2 ;;
esac
echo "== $(date +%H:%M:%S) $m: tools/llm_card.py $*"
timeout 7200 "$P" tools/llm_card.py "$@"; rc=$?
echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
