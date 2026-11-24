#!/bin/bash
# Claude: fetch the new models' files onto the card's host, at most 2 MB/s each, resumable, every
# LFS file checked against the sha256 Hugging Face lists for it (its LFS oid). Plain files
# (configs, tokenizers, licenses) are listed in SHA256SUMS as fetched. Nothing is run from them.
#   maple   deepgrove/maple-preview-GGUF  maple-preview-TQ2_0-head-F16.gguf (5.91 GiB, ~50 min)
#           + deepgrove/maple-preview's config, tokenizer, license (the revision is recorded)
#   ivonar  Ivonar/ivonar-nano            packed_inference_checkpoint.pt (95 MB), tokenizer, license
# Usage: r8_fetch_models.sh [maple] [ivonar]   (default: both)
set -u
M=~/openTPU_BitNET/models
RATE=2M
want=("$@"); [ ${#want[@]} = 0 ] && want=(ivonar maple)
log() { echo "== $(date +%T) $*"; }

fetch_repo() {   # repo dir file...
  local repo=$1 dir=$2; shift 2
  mkdir -p "$dir" || return 1
  local tree="$dir/.tree_$(echo "$repo" | tr / _).json"
  curl -fsSL --limit-rate $RATE -m 120 -o "$tree" "https://huggingface.co/api/models/$repo/tree/main" || { echo "no file list for $repo"; return 1; }
  curl -fsSL --limit-rate $RATE -m 120 "https://huggingface.co/api/models/$repo/revision/main" \
    | python3 -c "import json,sys; print('$repo', json.load(sys.stdin).get('sha'))" >> "$dir/REVISIONS" || true
  local f oid size rc=0
  for f in "$@"; do
    read -r oid size < <(python3 - "$tree" "$f" <<'PY'
import json, sys
for e in json.load(open(sys.argv[1])):
    if e.get("path") == sys.argv[2]:
        print((e.get("lfs") or {}).get("oid", "-"), e.get("size", 0)); break
else:
    print("missing 0")
PY
)
    if [ "$oid" = missing ]; then
      case " ${OPTIONAL:-generation_config.json attribution_bundle.md SHA256SUMS.txt} " in
        *" $f "*) echo "  $f: not in $repo (optional, skipped)" ;;
        *) echo "  $f: not in $repo"; rc=1 ;;
      esac
      continue
    fi
    log "$repo/$f ($size bytes)"
    if [ -f "$dir/$f" ] && [ "$(stat -c %s "$dir/$f")" = "$size" ] && { [ "$oid" = - ] || echo "$oid  $dir/$f" | sha256sum -c --status; }; then
      echo "  already here"
    else
      curl -fL --limit-rate $RATE -C - --retry 5 --retry-delay 10 -o "$dir/$f" "https://huggingface.co/$repo/resolve/main/$f" || { echo "  download failed"; rc=1; continue; }
      if [ "$oid" != - ]; then
        echo "$oid  $dir/$f" | sha256sum -c --status && echo "  sha256 ok ($oid)" || { echo "  SHA256 MISMATCH"; rc=1; continue; }
      fi
    fi
    (cd "$dir" && sha256sum "$f") >> "$dir/SHA256SUMS"
  done
  return $rc
}

rc=0
for w in "${want[@]}"; do
  case $w in
    maple)
      fetch_repo deepgrove/maple-preview "$M/maple-preview" config.json generation_config.json tokenizer.json tokenizer_config.json LICENSE README.md || rc=1
      fetch_repo deepgrove/maple-preview-GGUF "$M/maple-preview" maple-preview-TQ2_0-head-F16.gguf || rc=1 ;;
    ivonar)
      fetch_repo Ivonar/ivonar-nano "$M/ivonar-nano" packed_inference_checkpoint.pt tokenizer.json LICENSE attribution_bundle.md SHA256SUMS.txt README.md || rc=1 ;;
    *) echo "unknown model $w"; rc=1 ;;
  esac
done
for d in "$M/maple-preview" "$M/ivonar-nano"; do [ -d "$d" ] && { echo "--- $d"; ls -la "$d"; }; done
log "DONE rc $rc"
exit $rc
