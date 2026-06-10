"""Format and rendering checks for chat template Jinja files.

Checks run on every .jinja file under chat_templates/:

  1. Syntax    - jinja2.Environment().parse() catches broken template syntax.
  2. Rendering - apply_chat_template() renders conversations with the correct
                 structure: right role delimiters, content preserved, ordering
                 respected, system prompt present.

Run locally:
    pytest tests/test_chat_template.py -v
"""
from __future__ import annotations

import glob
import os

import jinja2
import pytest
from _pytest.outcomes import Failed
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

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
    """
    Minimal tokenizer that can render any chat template with tokenize=False.

    The vocabulary is intentionally tiny — only bos/eos/unk are needed because
    apply_chat_template(..., tokenize=False) returns a plain string without
    looking up any token IDs.
    """
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    inner = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    inner.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=inner,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    tok.chat_template = template_source
    return tok


def _render(path: str, messages: list[dict]) -> str:
    source = open(path, encoding="utf-8").read()
    tok = _make_tokenizer(source)
    return tok.apply_chat_template(messages, tokenize=False)


def test_chat_templates_directory_exists():
    assert os.path.isdir(_TEMPLATES_DIR), (
        f"chat_templates/ directory not found at {_TEMPLATES_DIR}"
    )


def test_at_least_one_jinja_file_present():
    assert _jinja_files(), f"No .jinja files found under {_TEMPLATES_DIR}"


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_jinja_syntax_is_valid(path):
    """jinja2 must compile the file without error.

    compile() is stricter than parse(): besides grammar it resolves filters and
    raises TemplateAssertionError (a TemplateSyntaxError subclass) on an unknown
    one, e.g. a typo'd ``|jion``. Function calls (strftime_now, raise_exception)
    are not compile-checked, so the real templates still pass.
    """
    source = open(path, encoding="utf-8").read()
    try:
        jinja2.Environment().compile(source)
    except jinja2.TemplateSyntaxError as exc:
        pytest.fail(f"{_rel(path)}: syntax error on line {exc.lineno}: {exc.message}")


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_renders_without_error(path):
    """apply_chat_template must not raise on a basic conversation."""
    _render(path, [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ])


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
def test_template_preserves_user_content(path):
    """User message content must appear in the rendered output."""
    marker = "UNIQUE_USER_CONTENT_ABC"
    rendered = _render(path, [
        {"role": "user", "content": marker},
        {"role": "assistant", "content": "response"},
    ])
    assert marker in rendered, "User content missing from rendered template"


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_preserves_assistant_content(path):
    """Assistant message content must appear in the rendered output."""
    marker = "UNIQUE_ASSISTANT_CONTENT_XYZ"
    rendered = _render(path, [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": marker},
    ])
    assert marker in rendered, "Assistant content missing from rendered template"


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
def test_template_message_order_preserved(path):
    """User turn must appear before assistant turn in the rendered output."""
    rendered = _render(path, [
        {"role": "user", "content": "SENTINEL_USER"},
        {"role": "assistant", "content": "SENTINEL_ASSISTANT"},
    ])
    assert rendered.index("SENTINEL_USER") < rendered.index("SENTINEL_ASSISTANT")


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
def test_template_multi_turn_all_turns_present(path):
    """All turns in a multi-turn conversation must appear in the rendered output."""
    rendered = _render(path, [
        {"role": "user", "content": "TURN_1_USER"},
        {"role": "assistant", "content": "TURN_1_ASSISTANT"},
        {"role": "user", "content": "TURN_2_USER"},
        {"role": "assistant", "content": "TURN_2_ASSISTANT"},
    ])
    for marker in ["TURN_1_USER", "TURN_1_ASSISTANT", "TURN_2_USER", "TURN_2_ASSISTANT"]:
        assert marker in rendered, f"{marker} missing from multi-turn render"


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
    test_jinja_syntax_is_valid(path)
    test_template_contains_role_delimiters(path)
    test_template_preserves_user_content(path)
    test_template_user_content_inside_user_delimiters(path)


def test_meta_broken_syntax_is_reported():
    """broken_syntax.jinja (an unterminated tag) must trip the syntax check."""
    # test_jinja_syntax_is_valid calls pytest.fail() on a syntax error.
    with pytest.raises(Failed):
        test_jinja_syntax_is_valid(_example("broken_syntax.jinja"))


def test_meta_unknown_filter_is_reported():
    """bad_filter.jinja (a typo'd ``|jion``) must trip the syntax check via
    compile(); plain parse() would let it through. Locks in the stricter check."""
    with pytest.raises(Failed):
        test_jinja_syntax_is_valid(_example("bad_filter.jinja"))


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
    """dropped_user_content.jinja (discards the user message) must fail content
    preservation."""
    with pytest.raises(AssertionError):
        test_template_preserves_user_content(_example("dropped_user_content.jinja"))