"""Format and rendering checks for chat template Jinja files.

Checks run on every .jinja file under chat_templates/: apply_chat_template()
renders conversations with the correct structure (role delimiters, content in
the right block, turns in order, a system block). Rendering compiles the
template, so broken syntax and unknown filters fail here too.

Run locally:
    pytest tests/test_chat_template.py -v
"""
from __future__ import annotations

import glob
import os

import jinja2
import pytest
from transformers import PreTrainedTokenizerFast

from tokenizer_factory import make_word_level_tokenizer

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TEMPLATES_DIR = os.path.join(_REPO_ROOT, "chat_templates")


def _jinja_files() -> list[str]:
    """Return all .jinja files under chat_templates/, sorted."""
    return sorted(
        glob.glob(os.path.join(_TEMPLATES_DIR, "**", "*.jinja"), recursive=True)
    )


def _rel(path: str) -> str:
    return os.path.relpath(path, _REPO_ROOT)


def _make_tokenizer(template_source: str) -> PreTrainedTokenizerFast:
    """Minimal tokenizer that can render any chat template with tokenize=False."""
    return make_word_level_tokenizer(
        whitespace=True, bos_eos=True, chat_template=template_source
    )


def _render(path: str, messages: list[dict]) -> str:
    source = open(path, encoding="utf-8").read()
    tok = _make_tokenizer(source)
    return tok.apply_chat_template(messages, tokenize=False)


def test_at_least_one_jinja_file_present():
    """Guards the discovery: with no files, every check below would skip."""
    assert _jinja_files(), f"No .jinja files found under {_TEMPLATES_DIR}"


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_contains_role_delimiters(path):
    """Rendered output must contain the Apertus role-delimiter tokens."""
    rendered = _render(path, [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ])
    assert "<|user_start|>" in rendered
    assert "<|user_end|>" in rendered
    assert "<|assistant_start|>" in rendered
    assert "<|assistant_end|>" in rendered


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_user_content_inside_user_delimiters(path):
    """User content must sit between <|user_start|> and <|user_end|>."""
    marker = "INSIDE_USER_BLOCK"
    rendered = _render(path, [
        {"role": "user", "content": marker},
        {"role": "assistant", "content": "ok"},
    ])
    start = rendered.index("<|user_start|>") + len("<|user_start|>")
    end = rendered.index("<|user_end|>")
    assert marker in rendered[start:end], (
        "User content is not between <|user_start|> and <|user_end|>"
    )


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_includes_system_token(path):
    """Rendered output must include a system block (default or explicit)."""
    rendered = _render(path, [{"role": "user", "content": "Hi"}])
    assert "<|system_start|>" in rendered, (
        "No <|system_start|> found - template must emit a system block"
    )


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_explicit_system_message_honored(path):
    """An explicit system message must appear in the rendered output."""
    marker = "CUSTOM_SYSTEM_PROMPT_99"
    rendered = _render(path, [
        {"role": "system", "content": marker},
        {"role": "user", "content": "Hi"},
    ])
    assert marker in rendered, "Explicit system message content missing from output"


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_multi_turn_all_turns_present_in_order(path):
    """Every turn of a multi-turn conversation is rendered, in order."""
    markers = ["TURN_1_USER", "TURN_1_ASSISTANT", "TURN_2_USER", "TURN_2_ASSISTANT"]
    roles = ["user", "assistant", "user", "assistant"]
    rendered = _render(path, [
        {"role": role, "content": marker} for role, marker in zip(roles, markers)
    ])
    for marker in markers:
        assert marker in rendered, f"{marker} missing from multi-turn render"
    positions = [rendered.index(marker) for marker in markers]
    assert positions == sorted(positions), "turns rendered out of order"


# ---------------------------------------------------------------------------
# Tool-call rendering: arguments may arrive as a JSON string (HF
# apply_chat_template passes them through verbatim) or as an already-parsed dict
# (vLLM json-parses tool_calls before rendering the prompt). The template must
# render both. Regression test for issue #4: replaying a prior assistant tool
# call under vLLM raised 'can only concatenate str (not "dict") to str'.
# ---------------------------------------------------------------------------

_APERTUS_1P5 = os.path.join(_TEMPLATES_DIR, "Apertus_1p5", "chat_template.jinja")


def _weather_tools() -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


def _multi_turn_messages(arguments) -> list[dict]:
    """A user→assistant(tool_call)→tool conversation, parametrized on the type
    of the replayed ``arguments`` field (str for HF, dict for vLLM)."""
    return [
        {"role": "user", "content": "Weather in Zurich? Use the tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": arguments},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "14C light rain"},
    ]


def _render_with_tools(messages: list[dict]) -> str:
    source = open(_APERTUS_1P5, encoding="utf-8").read()
    tok = _make_tokenizer(source)
    return tok.apply_chat_template(messages, tokenize=False, tools=_weather_tools())


def test_tool_call_arguments_render_the_same_as_string_or_dict():
    """HF passes arguments as a JSON string, vLLM as a parsed dict; both render
    the call identically. Before the issue #4 fix the dict path raised
    'can only concatenate str (not "dict") to str'."""
    as_string = _render_with_tools(_multi_turn_messages('{"city": "Zurich"}'))
    as_dict = _render_with_tools(_multi_turn_messages({"city": "Zurich"}))
    assert '{"get_weather": {"city": "Zurich"}}' in as_string
    assert as_dict == as_string


# ---------------------------------------------------------------------------
# Meta-tests: guard the guards.
#
# Deliberately broken templates (tests/examples/) are fed through the SAME
# checks the suite runs on real templates; each check must report a failure. If
# someone weakens a check so it passes everything, the matching meta-test below
# starts failing. These live under tests/ (not chat_templates/) so the
# parametrized suite above never picks them up.
# ---------------------------------------------------------------------------

_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


def _example(name: str) -> str:
    return os.path.join(_EXAMPLES_DIR, name)


def test_meta_good_example_passes_the_checks():
    """Sanity check: the unbroken baseline example passes the real checks, so a
    failure below is attributable to the breakage and not to the baseline."""
    path = _example("good_chat_template.jinja")
    test_template_contains_role_delimiters(path)
    test_template_user_content_inside_user_delimiters(path)


def test_meta_broken_syntax_is_reported():
    """broken_syntax.jinja (an unterminated tag) fails as soon as it renders."""
    with pytest.raises(jinja2.TemplateSyntaxError):
        test_template_contains_role_delimiters(_example("broken_syntax.jinja"))


def test_meta_unknown_filter_is_reported():
    """bad_filter.jinja (a typo'd ``|jion``) fails as soon as it renders:
    rendering compiles the template, which resolves filters."""
    with pytest.raises(jinja2.TemplateAssertionError):
        test_template_contains_role_delimiters(_example("bad_filter.jinja"))


def test_meta_corrupted_delimiters_are_reported():
    """corrupted_delimiters.jinja (the djlint --reformat failure mode: a stray
    space inside every special token) must be caught by the delimiter and
    content-placement checks."""
    path = _example("corrupted_delimiters.jinja")
    with pytest.raises(AssertionError):
        test_template_contains_role_delimiters(path)
    # Content-placement check locates the token with str.index, which raises
    # ValueError when the (now corrupted) token is absent.
    with pytest.raises((AssertionError, ValueError)):
        test_template_user_content_inside_user_delimiters(path)


def test_meta_dropped_user_content_is_reported():
    """dropped_user_content.jinja (discards the user message) must fail the
    user-block check."""
    with pytest.raises(AssertionError):
        test_template_user_content_inside_user_delimiters(
            _example("dropped_user_content.jinja")
        )