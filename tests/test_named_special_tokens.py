"""HF named media special tokens (see ModalityConfig.hf_named_tokens in omnitok.modalities).

Pins the emitted mapping, keeps the shipped Apertus_1p5 artifact in sync with the code, exercises
the write_tokenizer_config merge/replace branches, and verifies the transformers load surface.
"""

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from omnitok.io import write_tokenizer_config
from omnitok.modalities import AUDIO, VISION, ModalityConfig, hf_extra_special_tokens

APERTUS_1P5_DIR = Path(__file__).resolve().parent.parent / "tokenizers" / "Apertus_1p5"

EXPECTED_NAMED_TOKENS = {
    "image_token": "<|image|>",
    "boi_token": "<|img_start|>",
    "eoi_token": "<|img_end|>",
    "image_wrapper_token": "<|img_token_start|>",
    "eol_token": "<|img_end_of_row|>",
    "audio_token": "<|audio|>",
    "boa_token": "<|audio_start|>",
    "eoa_token": "<|audio_end|>",
}


class TestMapping:
    def test_vision_plus_audio_mapping(self):
        assert hf_extra_special_tokens([VISION, AUDIO]) == EXPECTED_NAMED_TOKENS

    def test_single_modality_subset(self):
        vision_only = hf_extra_special_tokens([VISION])
        assert set(vision_only) == {
            "image_token", "boi_token", "eoi_token", "image_wrapper_token", "eol_token"
        }

    def test_unknown_token_rejected(self):
        bad = ModalityConfig(
            name="bad",
            content_token_format="<|bad token {i}|>",
            start_token="<|bad_start|>",
            end_token="<|bad_end|>",
            structure_tokens=(),
            hf_named_tokens=(("bad_token", "<|not_a_modality_token|>"),),
        )
        with pytest.raises(ValueError, match="not .* the modality's tokens"):
            hf_extra_special_tokens([bad])

    def test_duplicate_name_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            hf_extra_special_tokens([VISION, VISION])

    def test_reserved_sft_key_name_rejected(self):
        bad = ModalityConfig(
            name="audio2",
            content_token_format="<|audio2 token {i}|>",
            start_token="<|a2_start|>",
            end_token="<|a2_end|>",
            structure_tokens=(),
            hf_named_tokens=(("audio2_end_token", "<|a2_end|>"),),
        )
        with pytest.raises(ValueError, match="reserved"):
            hf_extra_special_tokens([bad])


class _StubTokenizer:
    def get_vocab(self):
        return {f"t{i}": i for i in range(10)}


class TestWritePath:
    """The extra_special_tokens overlay in write_tokenizer_config."""

    def _write(self, tmp_path, initial, named):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(initial))
        write_tokenizer_config(str(tmp_path), _StubTokenizer(), 5, extra_special_tokens=named)
        return json.loads((tmp_path / "tokenizer_config.json").read_text())

    def test_merges_into_existing_dict(self, tmp_path):
        out = self._write(
            tmp_path,
            {"extra_special_tokens": {"keep_token": "<|keep|>"}},
            {"image_token": "<|image|>"},
        )
        assert out["extra_special_tokens"] == {"image_token": "<|image|>", "keep_token": "<|keep|>"}

    def test_replaces_list_form(self, tmp_path):
        # see the overlay comment in write_tokenizer_config for why lists are replaced
        out = self._write(
            tmp_path,
            {"extra_special_tokens": ["<|a|>", "<|b|>"]},
            {"image_token": "<|image|>"},
        )
        assert out["extra_special_tokens"] == {"image_token": "<|image|>"}

    def test_entries_sorted(self, tmp_path):
        out = self._write(tmp_path, {}, {"eol_token": "<|e|>", "boi_token": "<|b|>"})
        assert list(out["extra_special_tokens"]) == ["boi_token", "eol_token"]


class TestShippedArtifact:
    """The checked-in Apertus_1p5 config must match the code-declared mapping."""

    def test_artifact_matches_declared_mapping(self):
        config = json.loads((APERTUS_1P5_DIR / "tokenizer_config.json").read_text())
        assert config["extra_special_tokens"] == hf_extra_special_tokens([VISION, AUDIO])

    def test_transformers_exposes_named_attributes(self):
        # Loading the named mapping must only attach semantic attributes. If a
        # configured value were missing from tokenizer.json, transformers would
        # append a new token and increase len(tokenizer), potentially past the
        # model's embedding range.
        backend = Tokenizer.from_file(str(APERTUS_1P5_DIR / "tokenizer.json"))
        backend_vocab_size = backend.get_vocab_size(with_added_tokens=True)
        tokenizer = AutoTokenizer.from_pretrained(APERTUS_1P5_DIR)

        assert len(tokenizer) == backend_vocab_size
        assert len(tokenizer.get_vocab()) == backend_vocab_size

        for name, token in EXPECTED_NAMED_TOKENS.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert getattr(tokenizer, name, None) == token, name
            assert getattr(tokenizer, f"{name}_id", None) == token_id, f"{name}_id"
            assert token_id != tokenizer.unk_token_id, name
