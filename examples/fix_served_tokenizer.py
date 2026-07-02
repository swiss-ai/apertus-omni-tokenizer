"""Apply all serving-side tokenizer fixes to a deployed model directory.

Idempotent; safe to re-run. Applies:

1. mark_tokens_non_special  -- reasoning delimiters survive skip_special_tokens
   so the vLLM reasoning parser can separate `reasoning` from `content`
   (apertus-omni-tokenizer #5).
2. strip_bos_from_post_processor -- the post-processor no longer auto-prepends
   BOS, so a chat-templated prompt posted to /completions is not double-BOSed
   (apertus-program #420). Chat is unaffected (it encodes with
   add_special_tokens=False and takes its BOS from the template).

After running, restart the vLLM server so it reloads the tokenizer.

Usage::

    python examples/fix_served_tokenizer.py /path/to/served/model
"""

import argparse

from omnitok.io import mark_tokens_non_special, strip_bos_from_post_processor

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="Served model directory (edited in place).")
    args = ap.parse_args()

    flipped = mark_tokens_non_special(args.model_dir)
    stripped = strip_bos_from_post_processor(args.model_dir)
    print(
        f"Done: delimiters non-special={flipped or 'none'}; "
        f"post-processor BOS stripped={stripped}. Restart vLLM to apply."
    )
