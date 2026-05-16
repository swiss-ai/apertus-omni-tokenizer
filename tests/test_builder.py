"""Tests for omnitok.builder.add_modality()."""

import json
import os

import pytest
from transformers import AutoTokenizer

from omnitok import add_modality
from omnitok.modalities import MODALITY_REGISTRY, VISION, AUDIO

BASE_TOKENIZER = "swiss-ai/Apertus-8B-2509"
SMALL_VOCAB = 32


# ── Vocab size & mapping file ───────────────────────────────────────────────


class TestVisionOnly:
    def test_final_vocab_size(self, vision_tokenizer):
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        base = tok.vocab_size  # text-only base
        assert len(tok) == base + 200 + SMALL_VOCAB

    def test_mapping_file_exists(self, vision_tokenizer):
        assert os.path.exists(
            os.path.join(vision_tokenizer, "vision_token_mapping.json")
        )

    def test_mapping_has_all_entries(self, vision_tokenizer):
        with open(os.path.join(vision_tokenizer, "vision_token_mapping.json")) as f:
            data = json.load(f)
        assert data["visual_vocab_size"] == SMALL_VOCAB
        assert len(data["vision_token_ids"]) == SMALL_VOCAB

    def test_structure_tokens_resolve(self, vision_tokenizer):
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        for rename in VISION.structure_tokens:
            tid = tok.convert_tokens_to_ids(rename.target_name)
            assert tid != tok.unk_token_id, f"{rename.target_name} is unk"

    def test_base_vocab_size_in_config(self, vision_tokenizer):
        with open(os.path.join(vision_tokenizer, "tokenizer_config.json")) as f:
            config = json.load(f)
        assert "base_vocab_size" in config
        assert config["base_vocab_size"] > 0

    def test_content_tokens_contiguous(self, vision_tokenizer):
        with open(os.path.join(vision_tokenizer, "vision_token_mapping.json")) as f:
            data = json.load(f)
        ids = [data["vision_token_ids"][str(i)] for i in range(SMALL_VOCAB)]
        assert ids == list(range(ids[0], ids[0] + SMALL_VOCAB))


class TestAudioOnly:
    def test_final_vocab_size(self, audio_tokenizer):
        tok = AutoTokenizer.from_pretrained(audio_tokenizer)
        base = tok.vocab_size
        assert len(tok) == base + 200 + SMALL_VOCAB

    def test_mapping_file_exists(self, audio_tokenizer):
        assert os.path.exists(
            os.path.join(audio_tokenizer, "audio_token_mapping.json")
        )

    def test_structure_tokens_resolve(self, audio_tokenizer):
        tok = AutoTokenizer.from_pretrained(audio_tokenizer)
        for rename in AUDIO.structure_tokens:
            tid = tok.convert_tokens_to_ids(rename.target_name)
            assert tid != tok.unk_token_id, f"{rename.target_name} is unk"


# ── Stacking ────────────────────────────────────────────────────────────────


class TestStacking:
    def test_both_mapping_files_exist(self, stacked_tokenizer):
        assert os.path.exists(
            os.path.join(stacked_tokenizer, "vision_token_mapping.json")
        )
        assert os.path.exists(
            os.path.join(stacked_tokenizer, "audio_token_mapping.json")
        )

    def test_vision_ids_preserved(self, vision_tokenizer, stacked_tokenizer):
        with open(os.path.join(vision_tokenizer, "vision_token_mapping.json")) as f:
            vision_only = json.load(f)
        with open(os.path.join(stacked_tokenizer, "vision_token_mapping.json")) as f:
            stacked = json.load(f)
        for key in vision_only["vision_token_ids"]:
            assert vision_only["vision_token_ids"][key] == stacked["vision_token_ids"][key]

    def test_audio_after_vision(self, stacked_tokenizer):
        with open(os.path.join(stacked_tokenizer, "vision_token_mapping.json")) as f:
            vision = json.load(f)
        with open(os.path.join(stacked_tokenizer, "audio_token_mapping.json")) as f:
            audio = json.load(f)
        vision_last = vision["vision_token_offset"] + vision["visual_vocab_size"] - 1
        assert audio["audio_token_offset"] > vision_last

    def test_omnimodal_config(self, stacked_tokenizer):
        with open(os.path.join(stacked_tokenizer, "tokenizer_config.json")) as f:
            config = json.load(f)
        omc = config["omnimodal_config"]
        assert "omni_special_token_offset" in omc
        names = [m["name"] for m in omc["modalities"]]
        assert "vision" in names
        assert "audio" in names

    def test_omnimodal_config_sorted_by_offset(self, stacked_tokenizer):
        with open(os.path.join(stacked_tokenizer, "tokenizer_config.json")) as f:
            config = json.load(f)
        offsets = [m["offset"] for m in config["omnimodal_config"]["modalities"]]
        assert offsets == sorted(offsets)

    def test_all_structure_tokens_resolve(self, stacked_tokenizer):
        tok = AutoTokenizer.from_pretrained(stacked_tokenizer)
        for mc in [VISION, AUDIO]:
            for rename in mc.structure_tokens:
                tid = tok.convert_tokens_to_ids(rename.target_name)
                assert tid != tok.unk_token_id, f"{rename.target_name} is unk"


# ── Idempotency ─────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_same_vocab_size_skips(self, vision_tokenizer, tmp_path):
        out = str(tmp_path / "idem")
        # First call creates
        add_modality(BASE_TOKENIZER, out, "vision", SMALL_VOCAB)
        tok1 = AutoTokenizer.from_pretrained(out)
        size1 = len(tok1)
        # Second call with same vocab_size should skip, no error
        add_modality(out, out, "vision", SMALL_VOCAB)
        tok2 = AutoTokenizer.from_pretrained(out)
        assert len(tok2) == size1

    def test_different_vocab_size_raises(self, tmp_path):
        out = str(tmp_path / "mismatch")
        add_modality(BASE_TOKENIZER, out, "vision", SMALL_VOCAB)
        with pytest.raises(ValueError, match="already exists"):
            add_modality(out, str(tmp_path / "out2"), "vision", SMALL_VOCAB + 1)


# ── Error handling ──────────────────────────────────────────────────────────


class TestErrors:
    def test_unknown_modality(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown modality"):
            add_modality(BASE_TOKENIZER, str(tmp_path / "out"), "video", 32)

    def test_slot_validation(self, tmp_path):
        # Audio uses slot 13, so num_reserved_tokens=5 should fail
        with pytest.raises(ValueError, match="too small"):
            add_modality(
                BASE_TOKENIZER,
                str(tmp_path / "out"),
                "audio",
                32,
                num_reserved_tokens=5,
            )


# ── Returned tokenizer quality ──────────────────────────────────────────────


class TestReturnedTokenizer:
    def test_no_reserved_omni_placeholders(self, tmp_path):
        out = str(tmp_path / "fresh")
        tok, _ = add_modality(BASE_TOKENIZER, out, "vision", SMALL_VOCAB)
        vocab = tok.get_vocab()
        placeholders = [t for t in vocab if "RESERVED_OMNI" in t and t in [
            f"<|RESERVED_OMNI_{r.reserved_index:03d}|>"
            for r in VISION.structure_tokens
        ]]
        assert placeholders == [], f"Stale placeholders in returned vocab: {placeholders}"

    def test_returned_stats(self, tmp_path):
        out = str(tmp_path / "stats")
        _, stats = add_modality(BASE_TOKENIZER, out, "vision", SMALL_VOCAB)
        assert stats["modality"] == "vision"
        assert stats["content_tokens_added"] == SMALL_VOCAB
        assert stats["final_vocab_size"] > stats["original_vocab_size"]

    def test_returned_tokenizer_resave_preserves_omnimodal_config(self, tmp_path):
        out = str(tmp_path / "fresh")
        resave = str(tmp_path / "fresh_resave")
        tok, _ = add_modality(BASE_TOKENIZER, out, "vision", SMALL_VOCAB)

        assert "omnimodal_config" in tok.init_kwargs

        tok.save_pretrained(resave)
        with open(os.path.join(resave, "tokenizer_config.json")) as f:
            config = json.load(f)
        assert "omnimodal_config" in config

    def test_skip_path_returned_tokenizer_resave_preserves_omnimodal_config(
        self, tmp_path
    ):
        out = str(tmp_path / "idem")
        resave = str(tmp_path / "idem_resave")
        add_modality(BASE_TOKENIZER, out, "vision", SMALL_VOCAB)

        config_path = os.path.join(out, "tokenizer_config.json")
        with open(config_path) as f:
            config = json.load(f)
        config.pop("omnimodal_config", None)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        tok, _ = add_modality(out, out, "vision", SMALL_VOCAB)

        assert "omnimodal_config" in tok.init_kwargs

        tok.save_pretrained(resave)
        with open(os.path.join(resave, "tokenizer_config.json")) as f:
            config = json.load(f)
        assert "omnimodal_config" in config


# ── Extra config ─────────────────────────────────────────────────────────────


class TestExtraConfig:
    def test_extra_config_written(self, tmp_path):
        out = str(tmp_path / "extra")
        add_modality(
            BASE_TOKENIZER, out, "vision", SMALL_VOCAB,
            extra_config={"type": "Emu3.5", "path": "/models/emu"},
        )
        with open(os.path.join(out, "tokenizer_config.json")) as f:
            config = json.load(f)
        assert "vision_tokenizer" in config
        assert config["vision_tokenizer"]["type"] == "Emu3.5"
