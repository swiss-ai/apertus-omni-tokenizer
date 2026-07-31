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

from omnitok.apertus import REASONING_DELIMITER_TOKENS
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
            "<SPECIAL_73>": [73],
        },
        "decode": {32: "<think>", 33: "</think>"},
        "eos": "<|assistant_end|>",
    },
    "Apertus_1p5": {
        "encode": {
            "<think>": [32],
            "</think>": [33],
            "<|inner_prefix|>": [32],
            "<|inner_suffix|>": [33],
            "<|tool_output_start|>": [73],
            "<|tool_output_end|>": [74],
            "<|image|>": [131079],
            "<image>": [131079],
            "<|audio|>": [131085],
        },
        "decode": {32: "<|inner_prefix|>", 33: "<|inner_suffix|>"},
        "eos": "</s>",
        "normalizer_rules": [
            ("Regex", "<\\|channel\\|?>thought\\s*\\n", "<|inner_prefix|>"),
            ("String", "<channel|>", "<|inner_suffix|>"),
            ("String", "<thought>", "<|inner_prefix|>"),
            ("String", "</thought>", "<|inner_suffix|>"),
            ("String", "</answer>", ""),
            ("String", "<answer>", ""),
            ("Regex", "<\\|inner_suffix\\|>\\s+", "<|inner_suffix|>"),
            ("String", "<audio>", "<|audio|>"),
            ("String", "<image>", "<|image|>"),
            ("String", "<think>", "<|inner_prefix|>"),
            ("String", "</think>", "<|inner_suffix|>"),
        ],
    },
}

# The reasoning-delimiter fix is applied to Apertus 1.5 only; the 1.0 tokenizer
# is intentionally left unchanged, so the fix-behavior test runs on 1.5 alone.
FIXED_TOKENIZERS = {"Apertus_1p5"}

PINNED_DIRS = [p for p in TOKENIZER_DIRS if p.name in EXPECTED]
RULE_PINNED_DIRS = [
    p for p in TOKENIZER_DIRS if "normalizer_rules" in EXPECTED.get(p.name, {})
]


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


@pytest.mark.parametrize("tok_dir", PINNED_DIRS, ids=lambda p: p.name)
def test_special_token_encode(tok_dir):
    """Special tokens encode to their pinned IDs."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    for token, ids in EXPECTED[tok_dir.name]["encode"].items():
        assert tok.encode(token, add_special_tokens=False) == ids, token


@pytest.mark.parametrize("tok_dir", PINNED_DIRS, ids=lambda p: p.name)
def test_special_token_decode(tok_dir):
    """Reserved IDs decode to their pinned strings (pins the 32/33 asymmetry)."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    for token_id, expected in EXPECTED[tok_dir.name]["decode"].items():
        assert tok.decode([token_id]) == expected, token_id


@pytest.mark.parametrize("tok_dir", PINNED_DIRS, ids=lambda p: p.name)
def test_eos_token(tok_dir):
    """Apertus_1 mirrors upstream's eos; Apertus_1p5 carries the production
    convention (eos = </s>, turn/tool stops live in generation_config)."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    assert tok.eos_token == EXPECTED[tok_dir.name]["eos"]


@pytest.mark.parametrize("tok_dir", RULE_PINNED_DIRS, ids=lambda p: p.name)
def test_normalizer_rules(tok_dir):
    """The canonical's Replace rules, in order: the reasoning-format rewrites
    (<|channel|>thought / <thought> / <think> -> delimiters, <answer> strips,
    whitespace collapse) and the modality aliases. The rule set lives only in
    the artifact; this pins it against silent drift."""
    with open(tok_dir / "tokenizer.json") as f:
        norm = json.load(f)["normalizer"]
    rules = []
    for r in norm["normalizers"]:
        if r["type"] == "Replace":
            kind = "Regex" if "Regex" in r["pattern"] else "String"
            rules.append((kind, r["pattern"][kind], r["content"]))
    assert rules == [tuple(r) for r in EXPECTED[tok_dir.name]["normalizer_rules"]]


@pytest.mark.parametrize(
    "tok_dir",
    [p for p in TOKENIZER_DIRS if p.name in FIXED_TOKENIZERS],
    ids=[p.name for p in TOKENIZER_DIRS if p.name in FIXED_TOKENIZERS],
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

    flipped = mark_tokens_non_special(str(tmp_path), REASONING_DELIMITER_TOKENS)
    assert flipped == ["<|inner_prefix|>", "<|inner_suffix|>"]

    out_tj = json.loads((tmp_path / "tokenizer.json").read_text())
    by_id = {e["id"]: e["special"] for e in out_tj["added_tokens"]}
    assert by_id[32] is False and by_id[33] is False
    assert by_id[1] is True  # unrelated special token left alone
    out_tc = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert out_tc["added_tokens_decoder"]["32"]["special"] is False
    assert out_tc["added_tokens_decoder"]["33"]["special"] is False

    # Idempotent: nothing left to flip on a second run.
    assert mark_tokens_non_special(str(tmp_path), REASONING_DELIMITER_TOKENS) == []


def test_mark_tokens_non_special_updates_special_tokens_map(tmp_path):
    """The special_tokens_map.json branch drops reasoning delimiters from
    additional_special_tokens, leaves unrelated tokens, and reports ONLY the
    tokens actually removed -- not every candidate that happens to be absent."""
    # Only <|inner_prefix|> is present; the other three candidates are absent.
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": []}))
    (tmp_path / "special_tokens_map.json").write_text(json.dumps({
        "additional_special_tokens": ["<|inner_prefix|>", "<|keep_me|>"]
    }))

    flipped = mark_tokens_non_special(str(tmp_path), REASONING_DELIMITER_TOKENS)

    # Reports the one present delimiter, NOT <|inner_suffix|>/<think>/</think>.
    assert flipped == ["<|inner_prefix|>"]
    stm = json.loads((tmp_path / "special_tokens_map.json").read_text())
    assert stm["additional_special_tokens"] == ["<|keep_me|>"]
    # Idempotent: nothing left to remove.
    assert mark_tokens_non_special(str(tmp_path), REASONING_DELIMITER_TOKENS) == []


def test_mark_tokens_non_special_log_distinguishes_absent_from_already_fixed(
    tmp_path, capsys
):
    """The no-op log must distinguish 'delimiters present but already
    non-special' from 'delimiters not found' -- otherwise re-running on an
    already-fixed dir misleadingly reports them as absent."""
    # Present but already non-special -> "already non-special", not "not found".
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "added_tokens": [{"id": 32, "content": "<|inner_prefix|>", "special": False}]
    }))
    assert mark_tokens_non_special(str(tmp_path), REASONING_DELIMITER_TOKENS) == []
    out = capsys.readouterr().out
    assert "already non-special" in out
    assert "No reasoning delimiters found" not in out

    # Genuinely absent -> "not found".
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": []}))
    assert mark_tokens_non_special(str(tmp_path), REASONING_DELIMITER_TOKENS) == []
    assert "No reasoning delimiters found" in capsys.readouterr().out


def test_mark_tokens_non_special_ignores_non_added_token_refs_and_key_order(
    tmp_path, capsys
):
    """Presence/flip key on flat objects carrying BOTH content and special. A
    normalizer Replace rule that only mentions the token (nested pattern, no
    special) -- as the real Apertus 1.5 tokenizer.json has for <|inner_prefix|>
    -- must not be treated as a delimiter, and a reversed content/special key
    order must still flip."""
    tj = {
        # Replace rule: references the token in "content" but is NOT an
        # added-token entry (nested "pattern" object, no "special").
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "Replace",
                 "pattern": {"String": "<think>"},
                 "content": "<|inner_prefix|>"},
            ],
        },
        # Reversed key order: "special" before "content".
        "added_tokens": [
            {"id": 33, "special": True, "content": "<|inner_suffix|>"},
        ],
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tj, indent=2))

    flipped = mark_tokens_non_special(str(tmp_path), REASONING_DELIMITER_TOKENS)

    # inner_suffix flips despite reversed key order; inner_prefix is untouched
    # and not even reported "present" (it only appears in the Replace rule).
    assert flipped == ["<|inner_suffix|>"]
    assert "<|inner_prefix|>" not in capsys.readouterr().out
    out = json.loads((tmp_path / "tokenizer.json").read_text())
    assert out["added_tokens"][0]["special"] is False
    assert out["normalizer"]["normalizers"][0]["content"] == "<|inner_prefix|>"
