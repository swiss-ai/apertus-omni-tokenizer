#!/usr/bin/env bash
# Regenerate the canonical checksum manifests under validation/.
#
# Each manifest lists "<md5>  <served-filename>" for the files a deployed model
# directory is expected to contain (flat layout), mapped from where those files
# live in this repo. validate_model.sh consumes these manifests.
#
# Run from the repo root: bash validation/gen_checksums.sh
# CI regenerates and `git diff --exit-code`s the result, so the manifests can
# never silently drift from the checked-in tokenizers.
set -euo pipefail

cd "$(dirname "$0")/.."

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    echo "error: need md5sum or md5" >&2
    exit 3
  fi
}

# served-filename : repo-path  (one model per call)
gen_manifest() {
  local model="$1"; shift
  local out="validation/${model}.md5"
  : > "$out"
  while [ "$#" -gt 0 ]; do
    local served="$1" repo_path="$2"; shift 2
    if [ ! -f "$repo_path" ]; then
      echo "error: missing canonical file: $repo_path" >&2
      exit 2
    fi
    printf '%s  %s\n' "$(md5_of "$repo_path")" "$served" >> "$out"
  done
  echo "wrote $out"
}

gen_manifest Apertus_1p5 \
  chat_template.jinja      chat_templates/Apertus_1p5/chat_template.jinja \
  tokenizer.json           tokenizers/Apertus_1p5/tokenizer.json \
  tokenizer_config.json    tokenizers/Apertus_1p5/tokenizer_config.json \
  special_tokens_map.json  tokenizers/Apertus_1p5/special_tokens_map.json

gen_manifest Apertus_1 \
  chat_template.jinja      chat_templates/Apertus_1/chat_template.jinja \
  tokenizer.json           tokenizers/Apertus_1/tokenizer.json \
  tokenizer_config.json    tokenizers/Apertus_1/tokenizer_config.json \
  special_tokens_map.json  tokenizers/Apertus_1/special_tokens_map.json
