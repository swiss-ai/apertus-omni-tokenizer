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

Tool calls inside thinking (swiss-ai/apertus-program#1134). Apertus 1.5 calls
tools inside its thinking: a reply that calls a tool ends with
``<|tools_prefix|>[...]<|tools_suffix|>`` and has no reasoning-end token. The
text before ``<|tools_prefix|>`` is reasoning and the call is content, so the
tool parser sees it (problem 4 of the issue). In streaming, ``<|tools_prefix|>``
ends the reasoning phase like the end token does.

Independent of the chat template. A template may leave the thinking section open
at the end of the prompt (``enable_thinking`` in the patched 1.5 template) or not
(the released template). The parser never guesses this from the request; it uses
the prompt token ids, which vLLM hands to reasoning parsers through
``set_prompt_token_ids`` (both modes, recent vLLM) and
``adjust_initial_state_from_prompt`` (streaming). Where vLLM provides neither,
output without reasoning tokens is content: a thought may then show up as
content, but an answer or a call is never hidden in ``reasoning``. So the parser
and the template can be rolled out in either order.

Both serving modes are covered:
  * non-streaming -> ``extract_reasoning``
  * streaming     -> ``extract_reasoning_streaming`` (token-id based, so it is
    unaffected by ``skip_special_tokens``)

The non-streaming string split needs the delimiters to survive detokenization;
the tokenizer builder registers them as *non-special* so they do under the
default ``skip_special_tokens=true`` (apertus-omni-tokenizer #5, fixed via
``mark_tokens_non_special``).

Based on the parser by Imanol Schlag in swiss-ai/apertus-tool-use-thinking-demo.

Load with::

    --reasoning-parser apertus \
    --reasoning-parser-plugin /path/to/apertus_reasoning_parser.py
"""

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

# DeltaMessage moved twice across the supported vLLM range.
try:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
except ImportError:
    try:
        from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    except ImportError:
        from vllm.entrypoints.openai.protocol import DeltaMessage

# Candidate (start, end) delimiter pairs, in preference order. The parser uses
# whichever pair the loaded tokenizer has, choosing the one whose start token
# sits at the lower vocab id (the id the model was trained to emit).
_CANDIDATE_PAIRS = (
    ("<|inner_prefix|>", "<|inner_suffix|>"),
    ("<think>", "</think>"),
)

TOOLS_PREFIX = "<|tools_prefix|>"
# A turn boundary after the last reasoning start means the prompt is not inside
# a thinking section (e.g. "<|assistant_start|>" with thinking off).
TURN_MARKERS = (
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|user_end|>",
    "<|system_end|>",
    "<|developer_end|>",
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


def _get(message, key):
    return message.get(key) if isinstance(message, dict) else getattr(message, key, None)


def _after_tool(request) -> bool:
    messages = getattr(request, "messages", None) or []
    return bool(messages) and _get(messages[-1], "role") == "tool"


@ReasoningParserManager.register_module("apertus")
class ApertusReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for the Apertus deliberation block (tokenizer-agnostic
    across the ``<|inner_prefix|>`` and ``<think>`` delimiter schemes, and
    chat-template-agnostic, see the module docstring)."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tools_prefix_id = self.vocab.get(TOOLS_PREFIX)
        self.turn_ids = {self.vocab[t] for t in TURN_MARKERS if t in self.vocab}
        # True only when the prompt token ids show an open thinking section.
        self.open_from_prompt = False

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

    # ---- prompt state (from token ids, never from the request) ----

    def _last_marker(self, input_ids) -> str:
        """'open' (reasoning start), 'closed' (reasoning end) or 'turn' (a turn
        boundary, or no marker at all) -- whichever comes last."""
        for token_id in reversed(input_ids):
            if token_id == self.end_token_id:
                return "closed"
            if token_id == self.start_token_id:
                return "open"
            if token_id in self.turn_ids:
                return "turn"
        return "turn"

    def is_reasoning_end(self, input_ids) -> bool:
        """Reasoning has ended only after a reasoning end token. After a turn
        boundary the reply may still open a thinking section of its own, so the
        stream starts undecided: text is content until a reasoning start appears
        (see ``extract_reasoning_streaming``), and ``<|tools_prefix|>`` hands
        the rest to the tool parser."""
        return self._last_marker(input_ids) == "closed"

    def set_prompt_token_ids(self, prompt_token_ids) -> None:
        # Recent vLLM calls this in both modes before parsing.
        self.open_from_prompt = self._last_marker(prompt_token_ids) == "open"

    def adjust_initial_state_from_prompt(self, prompt_token_ids) -> None:
        # Streaming; called when is_reasoning_end(prompt) is False, which also
        # covers a prompt that ends at a turn boundary.
        self.open_from_prompt = self._last_marker(prompt_token_ids) == "open"

    # ---- non-streaming ----

    def extract_reasoning(self, model_output, request):
        start, end = self.start_token, self.end_token
        opened_here = start in model_output
        if not opened_here and self.open_from_prompt:
            model_output = start + model_output
        # No deliberation block at all (direct tool call or plain answer) -> all
        # content, so the tool parser sees any call and answers are not swallowed.
        if start not in model_output and end not in model_output:
            return None, model_output
        # A reasoning end without a start: the template opened the section and
        # vLLM did not pass the prompt (older vLLM); text before it is reasoning.
        text = model_output.partition(start)[2] if start in model_output else model_output
        if end in text:
            reasoning, _, content = text.partition(end)
            return reasoning or None, content or None
        index = text.find(TOOLS_PREFIX)
        if index >= 0:
            return text[:index] or None, text[index:]
        # After a tool result the model often answers without closing a section
        # the prompt left open; that text is the answer.
        if _after_tool(request) and not opened_here:
            return None, text or None
        return text or None, None

    # ---- streaming ----

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return self.end_token_id in delta_ids or (
            self.tools_prefix_id is not None and self.tools_prefix_id in delta_ids
        )

    def extract_content_ids(self, input_ids):
        if self.end_token_id in input_ids:
            return input_ids[input_ids.index(self.end_token_id) + 1 :]
        if self.tools_prefix_id is not None and self.tools_prefix_id in input_ids:
            return input_ids[input_ids.index(self.tools_prefix_id) :]
        return []

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            self.start_token_id,
            self.end_token_id,
        ):
            return None
        in_thinking = (
            self.open_from_prompt
            or self.start_token_id in previous_token_ids
            or self.start_token_id in delta_token_ids
        ) and self.end_token_id not in previous_token_ids
        if in_thinking and self.end_token_id not in delta_token_ids and TOOLS_PREFIX in delta_text:
            head, _, tail = delta_text.partition(TOOLS_PREFIX)
            head = head.replace(self.start_token, "")
            return DeltaMessage(reasoning=head or None, content=TOOLS_PREFIX + tail)
        if (
            self.open_from_prompt
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                head, _, tail = delta_text.partition(self.end_token)
                return DeltaMessage(reasoning=head or None, content=tail or None)
            if self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text)
            return DeltaMessage(reasoning=delta_text)
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def get_streaming_fallback_content(self, previous_text, request):
        """The stream ended inside the reasoning phase: after a tool result, text
        with no reasoning start and no call is the model's answer."""
        if not _after_tool(request) or self.start_token in previous_text or TOOLS_PREFIX in previous_text:
            return None
        return previous_text.replace(self.end_token, "").strip() or None
