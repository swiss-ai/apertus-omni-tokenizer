"""Validate the checked-in tokenizers under tokenizers/.

Goes beyond "is it valid JSON" (covered by the check-json pre-commit hook):
every tokenizer directory must actually load via transformers, and the
special-token encode/decode behavior is pinned per tokenizer so that
regenerating or replacing a tokenizer surfaces any change in a diff.

Note the documented asymmetry in Apertus_1p5: <think>/</think> and
<|inner_prefix|>/<|inner_suffix|> both encode to 32/33, but 32/33 decode
back to the <|inner_*|> form. Apertus_1 has no such collision. See PR #7.
"""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

TOKENIZERS_DIR = Path(__file__).resolve().parent.parent / "tokenizers"

# Every directory containing a tokenizer.json is discovered automatically,
# so a newly added tokenizer is load-tested without touching this file.
TOKENIZER_DIRS = sorted(
    p.parent for p in TOKENIZERS_DIR.glob("*/tokenizer.json")
)

# Pinned special-token behavior. Keyed by directory name; a directory without
# an entry here is still load-tested but not behavior-checked.
EXPECTED = {
    "Apertus_1": {
        "encode": {
            "<think>": [32],
            "</think>": [33],
            "<|inner_prefix|>": [69],
            "<|inner_suffix|>": [70],
        },
        "decode": {32: "<think>", 33: "</think>"},
    },
    "Apertus_1p5": {
        "encode": {
            "<think>": [32],
            "</think>": [33],
            "<|inner_prefix|>": [32],
            "<|inner_suffix|>": [33],
        },
        "decode": {32: "<|inner_prefix|>", 33: "<|inner_suffix|>"},
    },
}


@pytest.mark.parametrize(
    "tok_dir", TOKENIZER_DIRS, ids=[p.name for p in TOKENIZER_DIRS]
)
def test_tokenizer_loads(tok_dir):
    """Every checked-in tokenizer loads (catches truncated/corrupt files)."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    assert tok.vocab_size > 0


@pytest.mark.parametrize(
    "tok_dir", TOKENIZER_DIRS, ids=[p.name for p in TOKENIZER_DIRS]
)
def test_text_roundtrip(tok_dir):
    """Plain text survives an encode -> decode round trip."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    text = "Hello world, this is a tokenizer test."
    decoded = tok.decode(tok.encode(text, add_special_tokens=False))
    assert "Hello world" in decoded


@pytest.mark.parametrize(
    "tok_dir",
    [p for p in TOKENIZER_DIRS if p.name in EXPECTED],
    ids=[p.name for p in TOKENIZER_DIRS if p.name in EXPECTED],
)
def test_special_token_encode(tok_dir):
    """Special tokens encode to their pinned IDs."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    for token, ids in EXPECTED[tok_dir.name]["encode"].items():
        assert tok.encode(token, add_special_tokens=False) == ids, token


@pytest.mark.parametrize(
    "tok_dir",
    [p for p in TOKENIZER_DIRS if p.name in EXPECTED],
    ids=[p.name for p in TOKENIZER_DIRS if p.name in EXPECTED],
)
def test_special_token_decode(tok_dir):
    """Reserved IDs decode to their pinned strings (pins the 32/33 asymmetry)."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    for token_id, expected in EXPECTED[tok_dir.name]["decode"].items():
        assert tok.decode([token_id]) == expected, token_id
