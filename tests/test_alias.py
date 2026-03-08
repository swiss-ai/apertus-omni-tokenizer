"""Test token aliasing: <image> -> <|image|>, <audio> -> <|audio|>.

Verifies that the normalizer-based alias produces the same token IDs
as a manual string replacement, making dataloader-side transforms
(e.g. llava_to_apertus) unnecessary for the image/audio token.
"""

import pytest
from transformers import AutoTokenizer


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
