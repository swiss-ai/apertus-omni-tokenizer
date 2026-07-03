"""Strip the auto-prepended BOS from a deployed model's tokenizer.

Fixes apertus-program #420: the fast-tokenizer post-processor hardcodes the BOS
(``<s>``) in front of every sequence, so ``add_special_tokens=True`` (the default
on the ``/completions`` path) prepends it. The chat template ALSO emits the BOS,
so a client that posts a chat-templated prompt to ``/completions`` gets
``<s><s>...`` -> degeneration. Removing the BOS from the post-processor makes a
``<s>``-prefixed ``/completions`` prompt yield exactly one BOS; the chat path is
unaffected (vLLM encodes it with ``add_special_tokens=False``).

Idempotent. After running, restart the vLLM server so it reloads the tokenizer.

Usage::

    python examples/strip_completions_double_bos.py /path/to/served/model
"""

import argparse

from omnitok.io import strip_bos_from_post_processor

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="Served model directory (edited in place).")
    args = ap.parse_args()

    changed = strip_bos_from_post_processor(args.model_dir)
    print(
        f"{'Stripped post-processor BOS' if changed else 'No change needed'} "
        f"in {args.model_dir}. Restart vLLM to apply."
    )
