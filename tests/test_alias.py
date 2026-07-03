"""Test token aliasing: <image> -> <|image|>, <audio> -> <|audio|>.

Verifies that the normalizer-based alias produces the same token IDs
as a manual string replacement, making dataloader-side transforms
(e.g. llava_to_apertus) unnecessary for the image/audio token.
"""

import json

import pytest
from tokenizers import Tokenizer, models, normalizers
from transformers import AddedToken, AutoTokenizer, PreTrainedTokenizerFast

from omnitok.io import add_token_alias


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
    """Regression: adding an alias must not drop earlier normalizer rules.

    Wrapping a getter-derived Sequence in a new Sequence silently loses its
    children on some tokenizers versions, so a second alias used to erase
    the first, and a Sequence base normalizer lost e.g. its NFC step.
    These fixtures are synthetic (no network, no full build).
    """

    @staticmethod
    def _make_base(tmp_path, base_normalizer):
        backend = Tokenizer(models.WordLevel({"<unk>": 0, "hi": 1}, unk_token="<unk>"))
        backend.normalizer = base_normalizer
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
        tok.add_tokens(
            [
                AddedToken("<|image|>", special=True, normalized=True),
                AddedToken("<|audio|>", special=True, normalized=True),
            ]
        )
        out = str(tmp_path / "base")
        tok.save_pretrained(out)
        return out

    @staticmethod
    def _chain_types(tok):
        chain = json.loads(tok.backend_tokenizer.to_str())["normalizer"]
        assert chain["type"] == "Sequence"
        return [n["type"] for n in chain["normalizers"]]

    def test_second_alias_keeps_first_and_base_normalizer(self, tmp_path):
        base = self._make_base(tmp_path, normalizers.NFC())
        add_token_alias(base, "<|image|>", "<image>")
        add_token_alias(base, "<|audio|>", "<audio>")

        tok = AutoTokenizer.from_pretrained(base)
        img = tok.convert_tokens_to_ids("<|image|>")
        aud = tok.convert_tokens_to_ids("<|audio|>")
        assert tok.encode("<image>", add_special_tokens=False) == [img]
        assert tok.encode("<audio>", add_special_tokens=False) == [aud]

        types = self._chain_types(tok)
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

        tok = AutoTokenizer.from_pretrained(base)
        img = tok.convert_tokens_to_ids("<|image|>")
        aud = tok.convert_tokens_to_ids("<|audio|>")
        assert tok.encode("<image>", add_special_tokens=False) == [img]
        assert tok.encode("<audio>", add_special_tokens=False) == [aud]
