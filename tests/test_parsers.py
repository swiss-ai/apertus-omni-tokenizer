"""Checks for the vLLM parser plugins under ``parsers/vllm/``.

Two tiers:

  * Static (always run, no vLLM needed): the files parse, register under
    ``apertus``, and cover BOTH serving modes -- streaming and non-streaming.
  * Live (``pytest.importorskip("vllm")``): load the plugins, instantiate them
    against this repo's tokenizer, and parse a canned tool-call / reasoning
    string. These run only where vLLM is installed (the serving image / a dev
    box); they skip cleanly in a tokenizer-only environment.
"""

import ast
import importlib.util
import json
import os

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PARSER_DIR = os.path.join(_ROOT, "parsers", "vllm")
TOOL_PARSER = os.path.join(_PARSER_DIR, "apertus_tool_parser.py")
REASONING_PARSER = os.path.join(_PARSER_DIR, "apertus_reasoning_parser.py")


def _src(path: str) -> str:
    return open(path, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# Static checks -- no vLLM required.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", [TOOL_PARSER, REASONING_PARSER], ids=os.path.basename)
def test_parser_file_is_valid_python(path):
    ast.parse(_src(path))


@pytest.mark.parametrize("path", [TOOL_PARSER, REASONING_PARSER], ids=os.path.basename)
def test_parser_registers_as_apertus(path):
    assert 'register_module("apertus")' in _src(path), (
        "parser must self-register under the name 'apertus'"
    )


def test_tool_parser_covers_streaming_and_non_streaming():
    src = _src(TOOL_PARSER)
    assert "def extract_tool_calls(" in src, "missing non-streaming method"
    assert "def extract_tool_calls_streaming(" in src, "missing streaming method"


def test_reasoning_parser_covers_streaming_and_non_streaming():
    src = _src(REASONING_PARSER)
    # Non-streaming is overridden in-file; streaming is inherited from
    # BaseThinkingReasoningParser (token-id based), so assert the subclass.
    assert "def extract_reasoning(" in src, "missing non-streaming override"
    assert "BaseThinkingReasoningParser" in src, (
        "must subclass BaseThinkingReasoningParser (provides extract_reasoning_streaming)"
    )


# --------------------------------------------------------------------------- #
# Live checks -- need vLLM installed (serving image / dev box).
# --------------------------------------------------------------------------- #


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tokenizer():
    pytest.importorskip("vllm")  # live tests only
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        os.path.join(_ROOT, "tokenizers", "Apertus_1p5"), trust_remote_code=True
    )


def test_reasoning_parser_splits_deliberation(tokenizer):
    mod = _load(REASONING_PARSER, "apertus_reasoning_parser")
    parser = mod.ApertusReasoningParser(tokenizer)

    # This repo's tokenizer canonicalises <|inner_prefix|>/<|inner_suffix|> (id 32/33).
    assert parser.start_token == "<|inner_prefix|>"
    assert parser.end_token == "<|inner_suffix|>"

    reasoning, content = parser.extract_reasoning(
        "<|inner_prefix|>let me think about it<|inner_suffix|>The answer is 42.", None
    )
    assert reasoning.strip() == "let me think about it"
    assert content.strip() == "The answer is 42."

    # No deliberation block -> everything is content (never swallow the answer).
    r, c = parser.extract_reasoning("The answer is 42.", None)
    assert r is None and c == "The answer is 42."

    # Both modes present.
    assert callable(parser.extract_reasoning)
    assert callable(parser.extract_reasoning_streaming)


def test_pick_delimiter_pair_handles_both_schemes():
    """The 'handle both' requirement: pick <|inner_prefix|> on this repo's
    tokenizer, but <think> when that is the pair at the emitted (lower) id."""
    pytest.importorskip("vllm")
    mod = _load(REASONING_PARSER, "apertus_reasoning_parser_pick")

    # Repo scheme: only <|inner_prefix|>/<|inner_suffix|> present (id 32/33).
    repo_vocab = {"<|inner_prefix|>": 32, "<|inner_suffix|>": 33}
    assert mod._pick_delimiter_pair(repo_vocab) == ("<|inner_prefix|>", "<|inner_suffix|>")

    # Deployed scheme: both present, but <think>/</think> are at the lower id.
    deployed_vocab = {
        "<|inner_prefix|>": 69,
        "<|inner_suffix|>": 70,
        "<think>": 32,
        "</think>": 33,
    }
    assert mod._pick_delimiter_pair(deployed_vocab) == ("<think>", "</think>")


def test_tool_parser_parses_tool_call(tokenizer):
    mod = _load(TOOL_PARSER, "apertus_tool_parser")
    parser = mod.ApertusToolParser(tokenizer)

    out = 'Sure. <|tools_prefix|>[{"get_weather": {"city": "Paris"}}]<|tools_suffix|>'
    info = parser.extract_tool_calls(out, None)
    assert info.tools_called
    assert info.tool_calls[0].function.name == "get_weather"
    assert json.loads(info.tool_calls[0].function.arguments) == {"city": "Paris"}

    # Both modes present.
    assert callable(parser.extract_tool_calls)
    assert callable(parser.extract_tool_calls_streaming)


# --------------------------------------------------------------------------- #
# Reasoning parser logic (swiss-ai/apertus-program#1134). Runs everywhere: with
# real vLLM when installed, otherwise against tests/_vllm_stub.py. The parser
# must work with any chat template -- released (prompt ends at
# <|assistant_start|>), fixes 1-3 (prompt ends at <|tool_output_end|> after a
# tool result) and the variant that opens the thinking section in the prompt --
# because parser and template roll out independently.
# --------------------------------------------------------------------------- #

VOCAB = {
    "<|inner_prefix|>": 32,
    "<|inner_suffix|>": 33,
    "<|user_end|>": 65,
    "<|assistant_start|>": 66,
    "<|assistant_end|>": 68,
    "<|tools_prefix|>": 72,
    "<|tools_suffix|>": 73,
}
CALL = '<|tools_prefix|>[{"get_weather": {"city": "Bern"}}]<|tools_suffix|>'
# Prompt endings, as token ids.
PROMPT_NEW_TURN = [65, 66]  # released template, or thinking off
PROMPT_THINKING_OPEN = [65, 66, 32]  # template opens the thinking section
PROMPT_AFTER_CALL_IN_THINKING = [66, 32, 7, 72, 8, 73, 9]  # tool output, turn continues
PROMPT_AFTER_CLOSED_THINKING = [66, 32, 7, 33, 72, 8, 73, 9]


class _FakeTokenizer:
    def get_vocab(self):
        return dict(VOCAB)


class _Request:
    def __init__(self, after_tool=False):
        self.messages = [{"role": "user", "content": "Weather in Bern?"}]
        if after_tool:
            self.messages.append({"role": "tool", "content": '{"temp": 18}'})
        self.chat_template_kwargs = {"enable_thinking": True}


@pytest.fixture
def make_parser():
    """Load the reasoning parser against real vLLM, or the stub with sys.modules
    restored afterwards (so the live tests above still skip without vLLM)."""
    import importlib.util as iu
    import sys

    saved = None
    if iu.find_spec("vllm") is None:
        sys.path.insert(0, os.path.dirname(__file__))
        import _vllm_stub

        saved = {k: v for k, v in sys.modules.items() if k == "vllm" or k.startswith("vllm.")}
        _vllm_stub.install()
    try:
        mod = _load(REASONING_PARSER, "apertus_reasoning_parser_logic")
    finally:
        if saved is not None:
            for k in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")]:
                del sys.modules[k]
            sys.modules.update(saved)

    def make(prompt_ids=None):
        parser = mod.ApertusReasoningParser(_FakeTokenizer())
        if prompt_ids is not None:
            parser.set_prompt_token_ids(prompt_ids)
        return parser

    return make


def test_answer_without_thinking_is_content_on_released_template(make_parser):
    """The bug of the first tool-aware parser: with thinking on but a template
    that does not open the section, a direct answer must not become reasoning."""
    for prompt in (PROMPT_NEW_TURN, None):  # recent vLLM / vLLM without prompt ids
        r, c = make_parser(prompt).extract_reasoning("Hello! How can I help?", _Request())
        assert r is None and c == "Hello! How can I help?"


def test_call_inside_thinking_reaches_content(make_parser):
    """Problem 4: a call made inside an unclosed thinking section is content."""
    for prompt in (PROMPT_NEW_TURN, None):
        r, c = make_parser(prompt).extract_reasoning("<|inner_prefix|>Need the weather." + CALL, _Request())
        assert r == "Need the weather." and c == CALL


def test_thinking_opened_by_prompt(make_parser):
    parser = make_parser(PROMPT_THINKING_OPEN)
    r, c = parser.extract_reasoning("Need the weather." + CALL, _Request())
    assert r == "Need the weather." and c == CALL
    r, c = parser.extract_reasoning("Short think.<|inner_suffix|>It is 18 degrees.", _Request())
    assert r == "Short think." and c == "It is 18 degrees."


def test_thinking_opened_by_prompt_without_prompt_ids(make_parser):
    """Older vLLM passes no prompt ids: a reasoning end still splits correctly,
    and output with no markers is content (a thought may show, nothing is hidden)."""
    parser = make_parser(None)
    r, c = parser.extract_reasoning("Short think.<|inner_suffix|>It is 18 degrees.", _Request())
    assert r == "Short think." and c == "It is 18 degrees."
    r, c = parser.extract_reasoning("Need the weather." + CALL, _Request())
    assert r is None and c == "Need the weather." + CALL


def test_unclosed_answer_after_tool_result_is_content(make_parser):
    for prompt in (PROMPT_AFTER_CALL_IN_THINKING, PROMPT_AFTER_CLOSED_THINKING, None):
        r, c = make_parser(prompt).extract_reasoning("It is 18 degrees in Bern.", _Request(after_tool=True))
        assert r is None and c == "It is 18 degrees in Bern."


def test_prompt_state(make_parser):
    parser = make_parser()
    assert parser.is_reasoning_end(PROMPT_THINKING_OPEN) is False
    assert parser.is_reasoning_end(PROMPT_AFTER_CALL_IN_THINKING) is False
    # Undecided after a turn boundary: the reply may open thinking itself.
    assert parser.is_reasoning_end(PROMPT_NEW_TURN) is False
    assert parser.is_reasoning_end([66, 32, 7, 33]) is True
    for prompt, is_open in ((PROMPT_NEW_TURN, False), (PROMPT_THINKING_OPEN, True),
                            (PROMPT_AFTER_CALL_IN_THINKING, True), (PROMPT_AFTER_CLOSED_THINKING, False)):
        streaming = make_parser()
        streaming.adjust_initial_state_from_prompt(prompt)
        assert streaming.open_from_prompt is is_open
        assert make_parser(prompt).open_from_prompt is is_open


def test_streaming_released_template(make_parser):
    """Prompt ends at a turn boundary: plain text streams as content, a thinking
    section the model opens itself streams as reasoning, a call ends reasoning."""
    parser = make_parser()
    parser.adjust_initial_state_from_prompt(PROMPT_NEW_TURN)
    assert parser.extract_reasoning_streaming("", "Hi", "Hi", [], [7], [7]).content == "Hi"
    thinking = make_parser()
    thinking.adjust_initial_state_from_prompt(PROMPT_NEW_TURN)
    assert thinking.extract_reasoning_streaming("", "<|inner_prefix|>", "<|inner_prefix|>", [], [32], [32]) is None
    delta = thinking.extract_reasoning_streaming("<|inner_prefix|>", "<|inner_prefix|>Need", "Need", [32], [32, 7], [7])
    assert delta.reasoning == "Need" and delta.content is None
    split = thinking.extract_reasoning_streaming(
        "<|inner_prefix|>Need", "<|inner_prefix|>Need<|tools_prefix|>", "<|tools_prefix|>", [32, 7], [32, 7, 72], [72]
    )
    assert split.reasoning is None and split.content == "<|tools_prefix|>"
    assert thinking.is_reasoning_end_streaming([32, 7, 72], [72]) is True
    assert thinking.extract_content_ids([32, 7, 72, 9]) == [72, 9]


def test_streaming_with_thinking_opened_by_prompt(make_parser):
    parser = make_parser()
    parser.adjust_initial_state_from_prompt(PROMPT_THINKING_OPEN)
    first = parser.extract_reasoning_streaming("", "Need", "Need", [], [7], [7])
    assert first.reasoning == "Need" and first.content is None
    split = parser.extract_reasoning_streaming("Need", "Need<|tools_prefix|>", "<|tools_prefix|>", [7], [7, 72], [72])
    assert split.reasoning is None and split.content == "<|tools_prefix|>"


def test_streaming_fallback_promotes_unclosed_answer_after_tool(make_parser):
    parser = make_parser(PROMPT_AFTER_CALL_IN_THINKING)
    assert parser.get_streaming_fallback_content("It is 18 degrees.", _Request(after_tool=True)) == "It is 18 degrees."
    assert parser.get_streaming_fallback_content("Still thinking", _Request()) is None
    assert parser.get_streaming_fallback_content("x" + CALL, _Request(after_tool=True)) is None


def _stream_like_vllm(parser, chunks, request):
    """Drive the parser the way vLLM's streaming loop does: every delta goes
    through extract_reasoning_streaming while the reasoning phase is open, and
    when the stream ends inside it, get_streaming_fallback_content is appended
    to the content. Returns (reasoning, content) as a client would join them."""
    ids = {chunk: 100 + i for i, chunk in enumerate(dict.fromkeys(chunks))}
    reasoning = content = previous_text = ""
    previous_ids, reasoning_ended = [], False
    for chunk in chunks:
        delta_ids = [VOCAB.get(chunk, ids[chunk])]
        delta = parser.extract_reasoning_streaming(
            previous_text, previous_text + chunk, chunk, previous_ids, previous_ids + delta_ids, delta_ids
        )
        if delta is not None:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
        reasoning_ended = reasoning_ended or parser.is_reasoning_end_streaming(previous_ids + delta_ids, delta_ids)
        previous_text, previous_ids = previous_text + chunk, previous_ids + delta_ids
    if not reasoning_ended:
        content += parser.get_streaming_fallback_content(previous_text, request) or ""
    return reasoning, content


ANSWER = ["It", " is", " 18", " degrees."]


@pytest.mark.parametrize("prompt_ids", [PROMPT_NEW_TURN, [66, 72, 8, 73, 9]], ids=["new_turn", "after_tool_output"])
def test_streamed_answer_after_tool_is_sent_once(make_parser, prompt_ids):
    """On a prompt that does not leave thinking open, the answer is streamed as
    content; the fallback must not append it a second time."""
    parser = make_parser(prompt_ids)
    reasoning, content = _stream_like_vllm(parser, ANSWER, _Request(after_tool=True))
    assert content == "".join(ANSWER)
    assert reasoning == ""


def test_streamed_unclosed_answer_after_tool_reaches_content_when_prompt_opened_thinking(make_parser):
    """When the prompt left thinking open, the deltas are reasoning; the fallback
    hands the same text to the client as the answer, once."""
    parser = make_parser(PROMPT_AFTER_CALL_IN_THINKING)
    reasoning, content = _stream_like_vllm(parser, ANSWER, _Request(after_tool=True))
    assert content == "".join(ANSWER)
    assert reasoning == "".join(ANSWER)


# ---- forced tool calls (tool_choice "required" or a named tool) ----
# vLLM constrains the output to JSON once reasoning has ended and reads the
# content back as JSON, so the content must not carry the Apertus call markers.

REQUIRED_JSON = '[{"name": "get_weather", "parameters": {"city": "Bern"}}]'
NAMED_JSON = '{"city": "Bern"}'
NAMED_CHOICE = {"type": "function", "function": {"name": "get_weather"}}


def _forced(tool_choice, **kwargs):
    request = _Request(**kwargs)
    request.tool_choice = tool_choice
    return request


@pytest.mark.parametrize(
    "tool_choice, call_json", [("required", REQUIRED_JSON), (NAMED_CHOICE, NAMED_JSON)], ids=["required", "named"]
)
def test_forced_call_content_is_json_only(make_parser, tool_choice, call_json):
    request = _forced(tool_choice)
    for prompt_ids, output, reasoning in [
        (PROMPT_NEW_TURN, "<|tools_prefix|>" + call_json, None),
        (PROMPT_NEW_TURN, "<|tools_prefix|>" + call_json + "<|tools_suffix|>", None),
        (PROMPT_THINKING_OPEN, "Bern, so I call the tool.<|tools_prefix|>" + call_json, "Bern, so I call the tool."),
        (PROMPT_THINKING_OPEN, "Easy.<|inner_suffix|>" + call_json, "Easy."),
        (PROMPT_NEW_TURN, call_json, None),
    ]:
        parser = make_parser(prompt_ids)
        assert parser.extract_reasoning(output, request) == (reasoning, call_json), output


def test_forced_call_drops_text_before_the_call_outside_thinking(make_parser):
    """vLLM discards the content of a forced call anyway; the arguments must be JSON."""
    parser = make_parser(PROMPT_NEW_TURN)
    output = "I'll check the weather.<|tools_prefix|>" + NAMED_JSON
    assert parser.extract_reasoning(output, _forced(NAMED_CHOICE)) == (None, NAMED_JSON)


@pytest.mark.parametrize("tool_choice", [None, "auto", "none"])
def test_call_markers_stay_without_a_forced_call(make_parser, tool_choice):
    request = _forced(tool_choice)
    parser = make_parser(PROMPT_THINKING_OPEN)
    assert parser.extract_reasoning("Thought." + CALL, request) == ("Thought.", CALL)
