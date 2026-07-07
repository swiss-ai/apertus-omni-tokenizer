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
