#!/usr/bin/env bash
# Validate that a deployed model directory ships the canonical Apertus tokenizer.
#
# Usage (remote, the common case):
#   curl -fsSL https://raw.githubusercontent.com/swiss-ai/apertus-omni-tokenizer/main/validate_model.sh \
#     | bash -s -- [MODEL_PATH] [MODEL_NAME]
#
# Usage (local checkout):
#   bash validate_model.sh [MODEL_PATH] [MODEL_NAME]
#
#   MODEL_PATH   directory to validate (default: current working directory)
#   MODEL_NAME   manifest to check against (default: Apertus_1p5)
#
# It downloads the checksum manifest for MODEL_NAME from this repo and compares
# the md5 of each expected file in MODEL_PATH. Exits 0 if all match, 1 otherwise.
#
# Override the manifest source with env vars (e.g. to validate a branch/fork):
#   BASE_URL=https://raw.githubusercontent.com/<owner>/<repo>  BRANCH=<ref>
set -euo pipefail

MODEL_PATH="${1:-$PWD}"
MODEL_NAME="${2:-Apertus_1p5}"
BASE_URL="${BASE_URL:-https://raw.githubusercontent.com/swiss-ai/apertus-omni-tokenizer}"
BRANCH="${BRANCH:-main}"

GREEN=''; RED=''; RESET=''
if [ -t 1 ]; then GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; RESET=$'\033[0m'; fi
ok()   { printf '%s✔%s %s\n' "$GREEN" "$RESET" "$1"; }
fail() { printf '%s✗%s %s\n' "$RED" "$RESET" "$1"; }

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    echo "error: need md5sum or md5 on PATH" >&2
    exit 3
  fi
}

if [ ! -d "$MODEL_PATH" ]; then
  echo "error: model path not found: $MODEL_PATH" >&2
  exit 2
fi

# Fetch the canonical manifest. Fall back to a local copy when run inside a
# checkout (handy for offline use / testing).
MANIFEST=""
local_manifest="$(dirname "$0")/validation/${MODEL_NAME}.md5"
manifest_url="${BASE_URL}/${BRANCH}/validation/${MODEL_NAME}.md5"
if MANIFEST="$(curl -fsSL "$manifest_url" 2>/dev/null)" && [ -n "$MANIFEST" ]; then
  :
elif [ -f "$local_manifest" ]; then
  MANIFEST="$(cat "$local_manifest")"
else
  echo "error: could not fetch manifest for '$MODEL_NAME' from $manifest_url" >&2
  exit 2
fi

echo "Validating $MODEL_NAME tokenizer in: $MODEL_PATH"
echo

failures=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  expected="${line%% *}"
  fname="${line##* }"
  target="$MODEL_PATH/$fname"
  if [ ! -f "$target" ]; then
    fail "$fname (missing)"
    failures=$((failures + 1))
    continue
  fi
  actual="$(md5_of "$target")"
  if [ "$actual" = "$expected" ]; then
    ok "$fname"
  else
    fail "$fname (md5 $actual != $expected)"
    failures=$((failures + 1))
  fi
done <<< "$MANIFEST"

echo
if [ "$failures" -ne 0 ]; then
  echo "${RED}FAILED${RESET}: $failures file(s) did not match the canonical $MODEL_NAME tokenizer." >&2
  exit 1
fi
echo "${GREEN}OK${RESET}: $MODEL_NAME tokenizer matches the canonical checksums."
