# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM reasoning parser for Apertus models.

Apertus wraps its inner reasoning ("Deliberation") between a start/end pair of
special tokens. Which *string* those tokens carry depends on the tokenizer
build, but the model always emits the same low special-token **ids**:

    id 32 = reasoning start,  id 33 = reasoning end

The apertus-omni-tokenizer repo canonicalises these as
``<|inner_prefix|>``/``<|inner_suffix|>`` (ids 32/33); some deployed builds
additionally register ``<think>``/``</think>`` at ids 32/33 (with the
``<|inner_*|>`` strings pushed to other ids). ``BaseThinkingReasoningParser``
resolves the token id from the *string* via ``vocab.get(token)``, so this parser
picks whichever candidate pair the loaded tokenizer actually exposes at the
lower (emitted) id -- working across both tokenizer schemes without edits.

Both serving modes are covered:
  * non-streaming -> ``extract_reasoning`` (overridden below)
  * streaming     -> ``extract_reasoning_streaming`` (inherited; keys on the
    generated start/end token ids, so it is unaffected by ``skip_special_tokens``)

The non-streaming override is the important bit: when the model emits NO inner
tokens at all -- a direct tool call (``<|tools_prefix|>...``) or a plain answer
with no deliberation -- the whole output is returned as ``content`` (not
reasoning), so the tool-call parser still sees it and direct answers are not
swallowed. Note (apertus-omni-tokenizer #5): because the delimiters are *special*
tokens, the non-streaming string split only works when they survive
detokenization (``skip_special_tokens=false``, which the tool parser forces when
tools are active); streaming is unaffected. Registering the delimiters as
non-special in the tokenizer builder removes that caveat.

Load with::

    --reasoning-parser apertus \
    --reasoning-parser-plugin /path/to/apertus_reasoning_parser.py
"""

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

# Candidate (start, end) delimiter pairs, in preference order. The parser uses
# whichever pair the loaded tokenizer has, choosing the one whose start token
# sits at the lower vocab id (the id the model was trained to emit).
_CANDIDATE_PAIRS = (
    ("<|inner_prefix|>", "<|inner_suffix|>"),
    ("<think>", "</think>"),
)


def _pick_delimiter_pair(vocab) -> tuple:
    """Pick the (start, end) reasoning delimiter pair the tokenizer exposes,
    preferring the pair whose start token sits at the lower vocab id (the id the
    model was trained to emit). Falls back to the repo-canonical pair."""
    present = sorted(
        (vocab[start], start, end)
        for start, end in _CANDIDATE_PAIRS
        if start in vocab and end in vocab
    )
    if present:
        return present[0][1], present[0][2]
    return _CANDIDATE_PAIRS[0]


@ReasoningParserManager.register_module("apertus")
class ApertusReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for the Apertus deliberation block (tokenizer-agnostic
    across the ``<|inner_prefix|>`` and ``<think>`` delimiter schemes)."""

    def _pick_pair(self) -> tuple:
        # self.vocab is set by the base __init__ before start_token is read.
        return _pick_delimiter_pair(self.vocab)

    @property
    def start_token(self) -> str:
        """Token that starts reasoning (resolved to the emitted id in __init__)."""
        return self._pick_pair()[0]

    @property
    def end_token(self) -> str:
        """Token that ends reasoning."""
        return self._pick_pair()[1]

    def extract_reasoning(self, model_output, request):
        # No deliberation block at all (direct tool call or plain answer) -> all
        # content. Without this the base class would label the whole output as
        # reasoning whenever the end token is absent, hiding tool calls/answers.
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output, request)
