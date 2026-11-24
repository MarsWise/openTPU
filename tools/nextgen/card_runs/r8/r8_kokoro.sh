#!/bin/bash
# Claude: Kokoro-82M text to speech on the card, every model operation on the device (opentpu/kokoro_graph.py,
# tools/kokoro_card.py): two phrases, the float reference on the host CPU as a check (durations, F0 / N, log-mel),
# ECC before / after, the WAVs copied back. The checkpoint comes with the .cmd (~/llm_card/kokoro: .pth, config,
# voice), converted here once (tools/kokoro/convert.py: torch.load weights_only, sha256 pinned). Zeros over DRAM
# first. No bitstream load, no rescan, no calibration change.
set -u
exec 9>/tmp/claude_run.lock
flock -n 9 || { echo "== another run is using the card: wait for it"; exit 75; }
R=$HOME/openTPU_BitNET; P=$HOME/hfref/bin/python; K=$HOME/llm_card/kokoro
cd "$R" || exit 1
git fetch -q /tmp/sync.bundle claude-head:refs/remotes/claude/feature && git merge -q --ff-only refs/remotes/claude/feature \
  || { echo "SYNC FAILED (host tree: $(git log --oneline -1))"; exit 1; }
echo "== $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-90)"
export PYTHONPATH=$R:$R/tools
if [ ! -f "$K/conv/kokoro-v1_0.safetensors" ]; then
  mkdir -p "$K/src/voices" && cp "$K/af_heart.pt" "$K/src/voices/" && cp "$K/kokoro-v1_0.pth" "$K/src/" \
    && "$P" tools/kokoro/convert.py "$K/src" "$K/conv" af_heart || { echo "== conversion FAILED"; exit 1; }
fi
"$R/.venv/bin/python" -c "
from opentpu.host.board import Board, XdmaTransport
from opentpu.host import memcal
b = Board(XdmaTransport('/dev/xdma0'), check=False); b.scrub(force=True); i = b.info()
print('   scrubbed; build', hex(i.get('build_id') or 0), 'ECC', memcal.ecc(b.t), 'temp', i.get('temp_c'))"
rc=0; i=0
for ph in "həlˈO wˈɜɹld" "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˌOvəɹ ðə lˈAzi dˈɔɡ."; do
  i=$((i+1)); echo; echo "######## $(date +%H:%M:%S) phrase $i: $ph"
  timeout 5400 "$P" tools/kokoro_card.py --weights "$K/conv" --config "$K/config.json" --phonemes "$ph" \
    --out /tmp/kokoro_$i.wav --ref || rc=1
done
echo; echo "== $(date +%H:%M:%S) DONE rc $rc"
exit $rc
