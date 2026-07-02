"""Mark the reasoning delimiters non-special on a served model directory.

Fixes apertus-omni-tokenizer #5: the deliberation delimiters ship as *special*
tokens, so vLLM's default ``skip_special_tokens=True`` strips them from the
detokenized string before the reasoning parser runs. On the non-streaming path
the parser can then no longer find the end-of-reasoning delimiter, so the whole
deliberation block leaks into ``content`` and the ``reasoning`` channel stays
empty. Flipping the delimiters to non-special keeps them in the decoded output
for every client (no per-request ``skip_special_tokens`` override needed); their
ids are unchanged, so the streaming parser and any encoding of the literal string
behave exactly as before.

Scheme-agnostic: flips whichever of <|inner_prefix|>/<|inner_suffix|> and
<think>/</think> the tokenizer actually registers. Idempotent.

Usage::

    python examples/mark_reasoning_delimiters_nonspecial.py /path/to/served/model
"""

import argparse

from omnitok.io import mark_tokens_non_special

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="Served model directory (edited in place).")
    args = ap.parse_args()

    flipped = mark_tokens_non_special(args.model_dir)
    if flipped:
        print(f"OK: marked non-special in {args.model_dir}: {', '.join(flipped)}")
    else:
        print(f"No reasoning delimiters needed changing in {args.model_dir}")
