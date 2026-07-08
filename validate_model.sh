#!/usr/bin/env bash
# Validate that a deployed model directory ships the canonical Apertus tokenizer.
#
# Usage (remote, the common case): cd into the model dir, then validate it
#   cd /path/to/model
#   curl -fsSL https://raw.githubusercontent.com/swiss-ai/apertus-omni-tokenizer/main/validate_model.sh \
#     | bash
#   (defaults to validating the current directory against Apertus_1p5)
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

# --- generation_config.json: serving stop-token sanity ------------------------
# Field-level check, not md5: the file legitimately differs between checkpoints
# (transformers_version, sampling defaults). eos_token_id must include
# end-of-sequence, end-of-turn AND end-of-tool-call. Raw training exports
# ("_from_model_config": true) ship only the config.json eos (68); without 72
# the engine runs straight past a tool call and the model hallucinates the
# tool's output (fabricated results, duplicated answer blocks).
case "$MODEL_NAME" in
  Apertus_1p5) required_eos_ids="2 68 72" ;;  # </s> <|assistant_end|> <|tools_suffix|>
  *)           required_eos_ids="" ;;
esac

if [ -n "$required_eos_ids" ]; then
  gc="$MODEL_PATH/generation_config.json"
  if [ ! -f "$gc" ]; then
    fail "generation_config.json (missing: engines fall back to config.json eos and won't stop at tool calls)"
    failures=$((failures + 1))
  else
    compact="$(tr -d '[:space:]' < "$gc")"
    case "$compact" in
      *'"eos_token_id":'*)
        eos="${compact#*\"eos_token_id\":}"
        case "$eos" in
          \[*) eos="${eos#\[}"; eos="${eos%%\]*}" ;;
          *)   eos="${eos%%[,\}]*}" ;;
        esac
        ;;
      *) eos="" ;;
    esac
    eos_ids=" $(printf '%s' "$eos" | tr -c '0-9' ' ') "
    missing_ids=""
    for id in $required_eos_ids; do
      case "$eos_ids" in
        *" $id "*) ;;
        *) missing_ids="$missing_ids $id" ;;
      esac
    done
    if [ -z "$missing_ids" ]; then
      ok "generation_config.json (eos_token_id includes:$(printf ' %s' $required_eos_ids))"
    else
      fail "generation_config.json (eos_token_id [$eos] missing id(s):$missing_ids — 2=</s>, 68=<|assistant_end|>, 72=<|tools_suffix|>; a missing 72 means tool calls don't stop and tool output is hallucinated)"
      failures=$((failures + 1))
    fi
  fi
fi

echo
if [ "$failures" -ne 0 ]; then
  echo "${RED}FAILED${RESET}: $failures file(s) did not match the canonical $MODEL_NAME tokenizer." >&2
  exit 1
fi
echo "${GREEN}OK${RESET}: $MODEL_NAME tokenizer matches the canonical checksums."
