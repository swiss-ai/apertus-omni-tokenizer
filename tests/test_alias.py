"""Test token aliasing: <image> -> <|image|>, <audio> -> <|audio|>.

Verifies that the normalizer-based alias produces the same token IDs
as a manual string replacement, making dataloader-side transforms
(e.g. llava_to_apertus) unnecessary for the image/audio token.
"""

import json

import pytest
from tokenizers import normalizers
from transformers import AddedToken, AutoTokenizer

from omnitok.recipes.apertus_1p5 import add_reasoning_aliases
from omnitok.io import add_token_alias
from tokenizer_factory import make_word_level_tokenizer


# Every built fixture and the aliases it must honour.
ALIAS_CASES = [
    ("vision_tokenizer", {"<image>": "<|image|>"}),
    ("audio_tokenizer", {"<audio>": "<|audio|>"}),
    ("stacked_tokenizer", {"<image>": "<|image|>", "<audio>": "<|audio|>"}),
]


@pytest.mark.parametrize("built,aliases", ALIAS_CASES, ids=["vision", "audio", "stacked"])
def test_alias_encodes_to_its_target(built, aliases, request):
    tok = AutoTokenizer.from_pretrained(request.getfixturevalue(built))
    for alias, target in aliases.items():
        tid = tok.convert_tokens_to_ids(target)
        assert tid != tok.unk_token_id, target
        assert tok.encode(alias, add_special_tokens=False) == [tid], alias


@pytest.mark.parametrize("built,aliases", ALIAS_CASES, ids=["vision", "audio", "stacked"])
def test_alias_in_text_matches_manual_replacement(built, aliases, request):
    """The alias replaces the dataloader-side .replace() calls."""
    tok = AutoTokenizer.from_pretrained(request.getfixturevalue(built))
    text = " ".join(f"look at {alias} and {alias} again" for alias in aliases)
    replaced = text
    for alias, target in aliases.items():
        replaced = replaced.replace(alias, target)
    assert tok.encode(text) == tok.encode(replaced)


class TestAliasNormalizerChain:
    """Aliases must not drop earlier normalizer rules.
    Synthetic bases -- no network, no build."""

    @staticmethod
    def _make_base(tmp_path, base_normalizer):
        tok = make_word_level_tokenizer(
            ("<unk>", "hi"),
            normalizer=base_normalizer,
            added_tokens=[
                AddedToken("<|image|>", special=True, normalized=True),
                AddedToken("<|audio|>", special=True, normalized=True),
            ],
        )
        out = str(tmp_path / "base")
        tok.save_pretrained(out)
        return out

    @staticmethod
    def _chain_types(tok):
        chain = json.loads(tok.backend_tokenizer.to_str())["normalizer"]
        assert chain["type"] == "Sequence"
        return [n["type"] for n in chain["normalizers"]]

    @staticmethod
    def _assert_aliases_resolve(base):
        tok = AutoTokenizer.from_pretrained(base)
        for alias, target in (("<image>", "<|image|>"), ("<audio>", "<|audio|>")):
            tid = tok.convert_tokens_to_ids(target)
            assert tok.encode(alias, add_special_tokens=False) == [tid]
            assert tok.encode(target, add_special_tokens=False) == [tid]
        return tok

    def test_second_alias_keeps_first_and_base_normalizer(self, tmp_path):
        base = self._make_base(tmp_path, normalizers.NFC())
        add_token_alias(base, "<|image|>", "<image>")
        add_token_alias(base, "<|audio|>", "<audio>")

        types = self._chain_types(self._assert_aliases_resolve(base))
        assert types.count("Replace") == 2
        assert "NFC" in types

    def test_sequence_base_normalizer_survives(self, tmp_path):
        base = self._make_base(
            tmp_path, normalizers.Sequence([normalizers.NFC(), normalizers.NFKC()])
        )
        add_token_alias(base, "<|image|>", "<image>")

        tok = AutoTokenizer.from_pretrained(base)
        img = tok.convert_tokens_to_ids("<|image|>")
        assert tok.encode("<image>", add_special_tokens=False) == [img]

        types = self._chain_types(tok)
        assert "NFC" in types and "NFKC" in types

    def test_save_false_without_tokenizer_raises(self, tmp_path):
        base = self._make_base(tmp_path, normalizers.NFC())
        with pytest.raises(ValueError, match="discard"):
            add_token_alias(base, "<|image|>", "<image>", save=False)


def _reasoning_base(tmp_path):
    """A synthetic base carrying unaliased, unnormalized reasoning delimiters."""
    tok = make_word_level_tokenizer(
        ("<unk>", "hi"),
        added_tokens=[
            AddedToken("<|inner_prefix|>", special=False, normalized=False),
            AddedToken("<|inner_suffix|>", special=False, normalized=False),
        ],
    )
    out = str(tmp_path / "base")
    tok.save_pretrained(out)
    return out


@pytest.fixture(scope="module")
def reasoning_aliased(tmp_path_factory):
    base = _reasoning_base(tmp_path_factory.mktemp("reasoning"))
    add_reasoning_aliases(base)
    return AutoTokenizer.from_pretrained(base)


class TestReasoningAliases:
    """add_reasoning_aliases installs the canonical reasoning rewrites.
    Synthetic bases -- no network, no build."""

    @pytest.mark.parametrize(
        "text,canonical",
        [
            ("<think>", "<|inner_prefix|>"),
            ("<thought>", "<|inner_prefix|>"),
            ("<|channel|>thought\n", "<|inner_prefix|>"),
            ("</think>", "<|inner_suffix|>"),
            ("</thought>", "<|inner_suffix|>"),
            ("<channel|>", "<|inner_suffix|>"),
            ("<answer>hi</answer>", "hi"),
            ("<|inner_suffix|>   hi", "<|inner_suffix|>hi"),
        ],
    )
    def test_legacy_spelling_encodes_as_canonical(self, reasoning_aliased, text, canonical):
        """The targets start normalized=False, so resolving after a reload
        also proves the alias flipped them."""
        tok = reasoning_aliased
        assert tok.encode(text, add_special_tokens=False) == tok.encode(
            canonical, add_special_tokens=False
        )

    def test_targets_flipped_and_idempotent(self, tmp_path):
        base = _reasoning_base(tmp_path)
        add_reasoning_aliases(base)
        with open(f"{base}/tokenizer.json") as f:
            tj = json.load(f)
        flags = {t["content"]: t["normalized"] for t in tj["added_tokens"]}
        assert flags["<|inner_prefix|>"] is True and flags["<|inner_suffix|>"] is True
        n_rules = len(tj["normalizer"]["normalizers"])
        add_reasoning_aliases(base)
        with open(f"{base}/tokenizer.json") as f:
            assert len(json.load(f)["normalizer"]["normalizers"]) == n_rules

    def test_missing_delimiter_raises(self, tmp_path):
        tok = make_word_level_tokenizer()
        out = str(tmp_path / "bare")
        tok.save_pretrained(out)
        with pytest.raises(ValueError, match="not an added token"):
            add_reasoning_aliases(out)
