"""Validate the checked-in tokenizers under tokenizers/.

Goes beyond "is it valid JSON" (covered by the check-json pre-commit hook):
every tokenizer directory must actually load via transformers, and the
special-token encode/decode behavior is pinned per tokenizer so that
regenerating or replacing a tokenizer surfaces any change in a diff.

Note the documented asymmetry in Apertus_1p5: <think>/</think> and
<|inner_prefix|>/<|inner_suffix|> both encode to 32/33, but 32/33 decode
back to the <|inner_*|> form. Apertus_1 has no such collision. See PR #7.
"""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from omnitok.io import mark_tokens_non_special

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


@pytest.mark.parametrize(
    "tok_dir",
    [p for p in TOKENIZER_DIRS if p.name in EXPECTED],
    ids=[p.name for p in TOKENIZER_DIRS if p.name in EXPECTED],
)
def test_reasoning_delimiters_survive_skip_special(tok_dir):
    """The reasoning delimiters (ids 32/33) are non-special, so they survive
    decode under skip_special_tokens=True. If they were special, the default
    detokenization would strip them and a vLLM reasoning parser could not find
    the end-of-reasoning delimiter -- the whole deliberation block would leak
    into `content` (apertus-omni-tokenizer #5)."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    for token_id, expected in EXPECTED[tok_dir.name]["decode"].items():
        assert tok.decode([token_id], skip_special_tokens=True) == expected, token_id


@pytest.mark.parametrize(
    "tok_dir",
    [p for p in TOKENIZER_DIRS if p.name in EXPECTED],
    ids=[p.name for p in TOKENIZER_DIRS if p.name in EXPECTED],
)
def test_no_double_bos_on_completions_path(tok_dir):
    """The post-processor must not auto-prepend BOS. The chat template already
    emits it, so a chat-templated prompt posted to /completions (where
    add_special_tokens defaults to True) would otherwise be double-BOSed and
    degenerate (apertus-program #420).

    Checks (a) add_special_tokens no longer adds a leading BOS, and (b) a
    <s>-prefixed prompt encodes to exactly one leading BOS."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    bos_id = tok.bos_token_id
    assert bos_id is not None
    # (a) no auto-prepended BOS
    with_special = tok.encode("Paris", add_special_tokens=True)
    without_special = tok.encode("Paris", add_special_tokens=False)
    assert with_special == without_special, (
        f"add_special_tokens still prepends {with_special[:2]!r}"
    )
    # (b) a <s>-prefixed prompt -> single BOS, not two
    ids = tok.encode(f"{tok.bos_token}The capital of France is Paris.",
                     add_special_tokens=True)
    assert ids[:2] != [bos_id, bos_id], f"double BOS: {ids[:3]}"
    assert ids[0] == bos_id, f"expected one leading BOS, got {ids[:3]}"


def test_strip_bos_from_post_processor_is_surgical_and_idempotent(tmp_path):
    """strip_bos_from_post_processor removes only the BOS SpecialToken entries
    from single/pair, keeps the Sequence entries, and is a no-op on a second
    run."""
    import json as _json
    from omnitok.io import strip_bos_from_post_processor

    tj = {
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<s>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
        }
    }
    (tmp_path / "tokenizer.json").write_text(_json.dumps(tj, indent=2))
    (tmp_path / "tokenizer_config.json").write_text(_json.dumps({"bos_token": "<s>"}))

    assert strip_bos_from_post_processor(str(tmp_path)) is True
    pp = _json.loads((tmp_path / "tokenizer.json").read_text())["post_processor"]
    assert pp["single"] == [{"Sequence": {"id": "A", "type_id": 0}}]
    assert pp["pair"] == [
        {"Sequence": {"id": "A", "type_id": 0}},
        {"Sequence": {"id": "B", "type_id": 1}},
    ]
    # Idempotent: nothing left to strip.
    assert strip_bos_from_post_processor(str(tmp_path)) is False


def test_mark_tokens_non_special_flips_and_is_idempotent(tmp_path):
    """mark_tokens_non_special flips only the reasoning delimiters' `special`
    flag across tokenizer.json + tokenizer_config.json, leaves other tokens and
    all formatting intact, and is a no-op on a second run."""
    tj = {
        "added_tokens": [
            {"id": 32, "content": "<|inner_prefix|>", "special": True},
            {"id": 33, "content": "<|inner_suffix|>", "special": True},
            {"id": 1, "content": "<eos>", "special": True},  # unrelated: untouched
        ]
    }
    tc = {
        "added_tokens_decoder": {
            "32": {"content": "<|inner_prefix|>", "special": True},
            "33": {"content": "<|inner_suffix|>", "special": True},
        }
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tj, indent=2))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))

    flipped = mark_tokens_non_special(str(tmp_path))
    assert flipped == ["<|inner_prefix|>", "<|inner_suffix|>"]

    out_tj = json.loads((tmp_path / "tokenizer.json").read_text())
    by_id = {e["id"]: e["special"] for e in out_tj["added_tokens"]}
    assert by_id[32] is False and by_id[33] is False
    assert by_id[1] is True  # unrelated special token left alone
    out_tc = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert out_tc["added_tokens_decoder"]["32"]["special"] is False
    assert out_tc["added_tokens_decoder"]["33"]["special"] is False

    # Idempotent: nothing left to flip on a second run.
    assert mark_tokens_non_special(str(tmp_path)) == []
