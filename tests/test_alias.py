"""Test token aliasing: <image> -> <|image|>, <audio> -> <|audio|>.

Verifies that the normalizer-based alias produces the same token IDs
as a manual string replacement, making dataloader-side transforms
(e.g. llava_to_apertus) unnecessary for the image/audio token.
"""

import json

import pytest
from tokenizers import normalizers
from transformers import AddedToken, AutoTokenizer

from omnitok.apertus import add_reasoning_aliases
from omnitok.io import add_token_alias
from tokenizer_factory import make_word_level_tokenizer


class TestVisionAlias:
    """Test <image> -> <|image|> alias on a vision tokenizer."""

    def test_bare_tokens_same_id(self, vision_tokenizer):
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        assert tok.encode("<image>") == tok.encode("<|image|>")

    def test_in_sentence(self, vision_tokenizer):
        """Encoding '<image>' inside text produces the same IDs as '<|image|>'."""
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        ids_alias = tok.encode("Describe this: <image> please")
        ids_canonical = tok.encode("Describe this: <|image|> please")
        assert ids_alias == ids_canonical

    def test_matches_manual_replacement(self, vision_tokenizer):
        """Alias produces the same result as the old manual .replace() approach."""
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        text = "User asked about <image> in the conversation"

        # Old way: manual string replacement then tokenize
        ids_old = tok.encode(text.replace("<image>", "<|image|>"))
        # New way: tokenize directly, alias handles it
        ids_new = tok.encode(text)

        assert ids_old == ids_new

    def test_multiple_occurrences(self, vision_tokenizer):
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        text = "<image> first and <image> second"
        ids_alias = tok.encode(text)
        ids_canonical = tok.encode(text.replace("<image>", "<|image|>"))
        assert ids_alias == ids_canonical

    def test_image_token_is_single_id(self, vision_tokenizer):
        """<image> should encode to a single token, not be split into subwords."""
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        ids = tok.encode("<image>", add_special_tokens=False)
        assert len(ids) == 1
        assert ids[0] != tok.unk_token_id


class TestAudioAlias:
    """Test <audio> -> <|audio|> alias on an audio tokenizer."""

    def test_bare_tokens_same_id(self, audio_tokenizer):
        tok = AutoTokenizer.from_pretrained(audio_tokenizer)
        assert tok.encode("<audio>") == tok.encode("<|audio|>")

    def test_matches_manual_replacement(self, audio_tokenizer):
        tok = AutoTokenizer.from_pretrained(audio_tokenizer)
        text = "Transcribe this: <audio> clip"
        ids_old = tok.encode(text.replace("<audio>", "<|audio|>"))
        ids_new = tok.encode(text)
        assert ids_old == ids_new

    def test_audio_token_is_single_id(self, audio_tokenizer):
        tok = AutoTokenizer.from_pretrained(audio_tokenizer)
        ids = tok.encode("<audio>", add_special_tokens=False)
        assert len(ids) == 1
        assert ids[0] != tok.unk_token_id


class TestStackedAlias:
    """Both aliases survive after stacking vision + audio."""

    def test_image_alias_survives_stacking(self, stacked_tokenizer):
        tok = AutoTokenizer.from_pretrained(stacked_tokenizer)
        assert tok.encode("<image>") == tok.encode("<|image|>")

    def test_audio_alias_survives_stacking(self, stacked_tokenizer):
        tok = AutoTokenizer.from_pretrained(stacked_tokenizer)
        assert tok.encode("<audio>") == tok.encode("<|audio|>")

    def test_both_aliases_in_same_text(self, stacked_tokenizer):
        tok = AutoTokenizer.from_pretrained(stacked_tokenizer)
        text = "Show <image> and play <audio> now"
        ids_alias = tok.encode(text)
        ids_canonical = tok.encode(
            text.replace("<image>", "<|image|>").replace("<audio>", "<|audio|>")
        )
        assert ids_alias == ids_canonical


class TestAliasNormalizerChain:
    """Aliases must not drop earlier normalizer rules or depend on the target's normalized flag.
    Synthetic bases -- no network, no build."""

    @staticmethod
    def _make_base(tmp_path, base_normalizer, normalized=True):
        tok = make_word_level_tokenizer(
            ("<unk>", "hi"),
            normalizer=base_normalizer,
            added_tokens=[
                AddedToken("<|image|>", special=True, normalized=normalized),
                AddedToken("<|audio|>", special=True, normalized=normalized),
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

    def test_batched_in_memory_aliases(self, tmp_path):
        base = self._make_base(tmp_path, normalizers.NFC())
        tok = AutoTokenizer.from_pretrained(base)
        add_token_alias(base, "<|image|>", "<image>", tokenizer=tok, save=False)
        add_token_alias(base, "<|audio|>", "<audio>", tokenizer=tok, save=False)
        tok.save_pretrained(base)

        self._assert_aliases_resolve(base)

    def test_alias_target_flag_set_at_creation(self, tmp_path):
        """A normalized=False target is switched to True, in both saved files."""
        base = self._make_base(tmp_path, normalizers.NFC(), normalized=False)
        add_token_alias(base, "<|image|>", "<image>")
        add_token_alias(base, "<|audio|>", "<audio>")
        self._assert_aliases_resolve(base)

        with open(f"{base}/tokenizer.json") as f:
            tj = json.load(f)
        flags = {e["content"]: e["normalized"] for e in tj["added_tokens"]}
        assert flags["<|image|>"] is True and flags["<|audio|>"] is True
        # The flip must survive a reload through whichever save format the
        # running transformers wrote (4.x trusts the config mirror over
        # tokenizer.json; 5.x writes no mirror).
        reloaded = AutoTokenizer.from_pretrained(base)
        for name in ("<|image|>", "<|audio|>"):
            entry = reloaded.added_tokens_decoder[reloaded.convert_tokens_to_ids(name)]
            assert entry.normalized is True, name

    def test_realias_is_idempotent(self, tmp_path):
        base = self._make_base(tmp_path, normalizers.NFC())
        add_token_alias(base, "<|image|>", "<image>")
        add_token_alias(base, "<|image|>", "<image>")

        tok = AutoTokenizer.from_pretrained(base)
        assert self._chain_types(tok).count("Replace") == 1

    def test_unknown_alias_target_raises(self, tmp_path):
        base = self._make_base(tmp_path, normalizers.NFC())
        with pytest.raises(ValueError, match="not an added token"):
            add_token_alias(base, "<|missing|>", "<missing>")

    def test_save_false_without_tokenizer_raises(self, tmp_path):
        base = self._make_base(tmp_path, normalizers.NFC())
        with pytest.raises(ValueError, match="discard"):
            add_token_alias(base, "<|image|>", "<image>", save=False)


class TestReasoningAliases:
    """add_reasoning_aliases installs the canonical reasoning rewrites.
    Synthetic bases -- no network, no build."""

    @staticmethod
    def _base(tmp_path):
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

    def test_all_spellings_resolve_to_delimiters(self, tmp_path):
        base = self._base(tmp_path)
        add_reasoning_aliases(base)
        tok = AutoTokenizer.from_pretrained(base)
        prefix = tok.convert_tokens_to_ids("<|inner_prefix|>")
        suffix = tok.convert_tokens_to_ids("<|inner_suffix|>")
        for spelling, tid in (
            ("<think>", prefix), ("<thought>", prefix),
            ("</think>", suffix), ("</thought>", suffix), ("<channel|>", suffix),
        ):
            assert tok.encode(spelling, add_special_tokens=False) == [tid], spelling
        assert tok.encode("<|channel|>thought\n", add_special_tokens=False) == [prefix]

    def test_answer_wrappers_are_stripped(self, tmp_path):
        base = self._base(tmp_path)
        add_reasoning_aliases(base)
        tok = AutoTokenizer.from_pretrained(base)
        assert tok.encode("<answer>hi</answer>", add_special_tokens=False) == tok.encode(
            "hi", add_special_tokens=False
        )

    def test_whitespace_after_suffix_collapses(self, tmp_path):
        base = self._base(tmp_path)
        add_reasoning_aliases(base)
        tok = AutoTokenizer.from_pretrained(base)
        assert tok.encode("<|inner_suffix|>   hi", add_special_tokens=False) == tok.encode(
            "<|inner_suffix|>hi", add_special_tokens=False
        )

    def test_targets_flipped_and_idempotent(self, tmp_path):
        base = self._base(tmp_path)
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
