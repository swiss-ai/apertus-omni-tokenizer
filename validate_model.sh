#!/usr/bin/env bash
# Validate that a deployed model directory ships the canonical Apertus tokenizer.
#
# Usage (remote, the common case): cd into the model dir, then validate it
#   cd /path/to/model
#   curl -fsSL https://raw.githubusercontent.com/swiss-ai/apertus-omni-tokenizer/main/validate_model.sh \
#     | bash
#   (defaults to validating the current directory against Apertus_1p5)
#
#   To also repair mismatches, pass --fix through bash:
#   curl -fsSL .../validate_model.sh | bash -s -- --fix
#
# Usage (local checkout):
#   bash validate_model.sh [--fix] [MODEL_PATH] [MODEL_NAME]
#
#   MODEL_PATH   directory to validate (default: current working directory)
#   MODEL_NAME   manifest to check against (default: Apertus_1p5)
#   --fix        download the canonical copy of each missing/mismatched manifest
#                file from this repo and install it (the existing file, if any,
#                is saved as <file>.bak first), then re-run the validation.
#                generation_config.json is checked field-level and has no
#                canonical copy in the repo, so --fix cannot repair it.
#
# It downloads the checksum manifest for MODEL_NAME from this repo and compares
# the md5 of each expected file in MODEL_PATH. Exits 0 if all match, 1 otherwise.
#
# Override the manifest source with env vars (e.g. to validate a branch/fork):
#   BASE_URL=https://raw.githubusercontent.com/<owner>/<repo>  BRANCH=<ref>
set -euo pipefail

MODEL_PATH=""
MODEL_NAME=""
FIX=0
for arg in "$@"; do
  case "$arg" in
    --fix) FIX=1 ;;
    -*)
      echo "error: unknown option: $arg" >&2
      echo "usage: validate_model.sh [--fix] [MODEL_PATH] [MODEL_NAME]" >&2
      exit 2
      ;;
    *)
      if [ -z "$MODEL_PATH" ]; then MODEL_PATH="$arg"
      elif [ -z "$MODEL_NAME" ]; then MODEL_NAME="$arg"
      else
        echo "error: unexpected argument: $arg" >&2
        exit 2
      fi
      ;;
  esac
done
MODEL_PATH="${MODEL_PATH:-$PWD}"
MODEL_NAME="${MODEL_NAME:-Apertus_1p5}"
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

# Where a served file lives in this repo (source for --fix downloads).
repo_path_for() {
  case "$1" in
    chat_template.jinja) printf 'chat_templates/%s/%s' "$MODEL_NAME" "$1" ;;
    *)                   printf 'tokenizers/%s/%s' "$MODEL_NAME" "$1" ;;
  esac
}

# Runs the full validation once. Sets:
#   failures        total failed checks
#   FAILED_MANIFEST newline-separated "<expected-md5> <fname>" for manifest
#                   files that were missing or mismatched (fixable via --fix)
run_checks() {
  failures=0
  FAILED_MANIFEST=""

  echo "Validating $MODEL_NAME tokenizer in: $MODEL_PATH"
  echo

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    expected="${line%% *}"
    fname="${line##* }"
    target="$MODEL_PATH/$fname"
    if [ ! -f "$target" ]; then
      fail "$fname (missing)"
      failures=$((failures + 1))
      FAILED_MANIFEST="${FAILED_MANIFEST}${expected} ${fname}"$'\n'
      continue
    fi
    actual="$(md5_of "$target")"
    if [ "$actual" = "$expected" ]; then
      ok "$fname"
    else
      fail "$fname (md5 $actual != $expected)"
      failures=$((failures + 1))
      FAILED_MANIFEST="${FAILED_MANIFEST}${expected} ${fname}"$'\n'
    fi
  done <<< "$MANIFEST"

  # --- generation_config.json: serving stop-token sanity ----------------------
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
        if [ "$FIX" -eq 1 ]; then
          echo "  note: --fix cannot repair generation_config.json (no canonical copy in the repo); edit eos_token_id manually"
        fi
      fi
    fi
  fi
}

# Download the canonical copy of a manifest file, verify it against the
# manifest md5, back up the existing file as <fname>.bak, and install it.
fix_file() {
  local expected="$1" fname="$2"
  local repo_path url tmp got target
  repo_path="$(repo_path_for "$fname")"
  url="${BASE_URL}/${BRANCH}/${repo_path}"
  target="$MODEL_PATH/$fname"

  tmp="$(mktemp "${TMPDIR:-/tmp}/omnitok_fix.XXXXXX")"
  if ! curl -fsSL "$url" -o "$tmp" 2>/dev/null; then
    local local_src
    local_src="$(dirname "$0")/$repo_path"
    if [ -f "$local_src" ]; then
      cp "$local_src" "$tmp"
    else
      rm -f "$tmp"
      fail "$fname (could not download canonical copy from $url)"
      return 1
    fi
  fi

  got="$(md5_of "$tmp")"
  if [ "$got" != "$expected" ]; then
    rm -f "$tmp"
    fail "$fname (canonical copy md5 $got != manifest $expected; not replacing — is BRANCH/BASE_URL out of sync with the manifest?)"
    return 1
  fi

  if [ -f "$target" ]; then
    cp -p "$target" "$target.bak"
    mv "$tmp" "$target"
    chmod 644 "$target"
    ok "$fname (replaced; original saved as $fname.bak)"
  else
    mv "$tmp" "$target"
    chmod 644 "$target"
    ok "$fname (installed; was missing)"
  fi
}

run_checks

if [ "$failures" -ne 0 ] && [ "$FIX" -eq 1 ] && [ -n "$FAILED_MANIFEST" ]; then
  echo
  echo "Fixing $(printf '%s' "$FAILED_MANIFEST" | grep -c .) file(s) from ${BASE_URL}/${BRANCH}"
  echo
  fix_errors=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    if ! fix_file "${line%% *}" "${line#* }"; then
      fix_errors=$((fix_errors + 1))
    fi
  done <<< "$FAILED_MANIFEST"

  echo
  echo "Re-running validation"
  echo
  run_checks
fi

echo
if [ "$failures" -ne 0 ]; then
  echo "${RED}FAILED${RESET}: $failures file(s) did not match the canonical $MODEL_NAME tokenizer." >&2
  exit 1
fi
echo "${GREEN}OK${RESET}: $MODEL_NAME tokenizer matches the canonical checksums."
