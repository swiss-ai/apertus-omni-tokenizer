"""Tests for omnitok.builder.add_modality()."""

import json
import os

import pytest
from transformers import AutoTokenizer

from tokenizer_factory import make_word_level_tokenizer

from omnitok import (
    add_modality,
    detect_existing_modalities,
    get_content_token_id,
    load_modality_mapping,
)
from omnitok.io import (
    build_omnimodal_config,
    rename_reserved_tokens,
)
from omnitok.modalities import MODALITY_REGISTRY, VISION, AUDIO

BASE_TOKENIZER = "swiss-ai/Apertus-8B-2509"
SMALL_VOCAB = 32


def _omnimodal_entry(tokenizer_dir, name):
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json")) as f:
        config = json.load(f)
    return next(
        m for m in config["omnimodal_config"]["modalities"] if m["name"] == name
    )


# ── Vocab size & omnimodal metadata ──────────────────────────────────────────


class TestVisionOnly:
    def test_final_vocab_size(self, vision_tokenizer):
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        base = tok.vocab_size  # text-only base
        assert len(tok) == base + 200 + SMALL_VOCAB

    def test_no_mapping_file_emitted(self, vision_tokenizer):
        assert not os.path.exists(
            os.path.join(vision_tokenizer, "vision_token_mapping.json")
        )

    def test_omnimodal_entry(self, vision_tokenizer):
        entry = _omnimodal_entry(vision_tokenizer, "vision")
        assert entry["vocab_size"] == SMALL_VOCAB

    def test_content_token_id_lookup(self, vision_tokenizer):
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        mapping = load_modality_mapping(vision_tokenizer, "vision")
        for i in (0, SMALL_VOCAB - 1):
            expected = tok.convert_tokens_to_ids(f"<|visual token {i}|>")
            assert get_content_token_id(i, mapping=mapping) == expected
        with pytest.raises(ValueError, match="not found"):
            get_content_token_id(SMALL_VOCAB, mapping=mapping)

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
        tok = AutoTokenizer.from_pretrained(vision_tokenizer)
        offset = _omnimodal_entry(vision_tokenizer, "vision")["offset"]
        ids = [
            tok.convert_tokens_to_ids(f"<|visual token {i}|>")
            for i in range(SMALL_VOCAB)
        ]
        assert ids == list(range(offset, offset + SMALL_VOCAB))


class TestAudioOnly:
    def test_final_vocab_size(self, audio_tokenizer):
        tok = AutoTokenizer.from_pretrained(audio_tokenizer)
        base = tok.vocab_size
        assert len(tok) == base + 200 + SMALL_VOCAB

    def test_no_mapping_file_emitted(self, audio_tokenizer):
        assert not os.path.exists(
            os.path.join(audio_tokenizer, "audio_token_mapping.json")
        )

    def test_structure_tokens_resolve(self, audio_tokenizer):
        tok = AutoTokenizer.from_pretrained(audio_tokenizer)
        for rename in AUDIO.structure_tokens:
            tid = tok.convert_tokens_to_ids(rename.target_name)
            assert tid != tok.unk_token_id, f"{rename.target_name} is unk"


# ── Stacking ────────────────────────────────────────────────────────────────


class TestStacking:
    def test_no_mapping_files_emitted(self, stacked_tokenizer):
        assert not os.path.exists(
            os.path.join(stacked_tokenizer, "vision_token_mapping.json")
        )
        assert not os.path.exists(
            os.path.join(stacked_tokenizer, "audio_token_mapping.json")
        )

    def test_vision_ids_preserved(self, vision_tokenizer, stacked_tokenizer):
        vision_only = _omnimodal_entry(vision_tokenizer, "vision")
        stacked = _omnimodal_entry(stacked_tokenizer, "vision")
        assert vision_only["offset"] == stacked["offset"]
        assert vision_only["vocab_size"] == stacked["vocab_size"]

    def test_audio_after_vision(self, stacked_tokenizer):
        vision = _omnimodal_entry(stacked_tokenizer, "vision")
        audio = _omnimodal_entry(stacked_tokenizer, "audio")
        vision_last = vision["offset"] + vision["vocab_size"] - 1
        assert audio["offset"] > vision_last

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


# ── Omnimodal derivation (synthetic, no network) ─────────────────────────────


def _synthetic_tokenizer(tokens, *, special=False):
    return make_word_level_tokenizer(added_tokens=tokens, added_special=special)


class TestOmnimodalDerivation:
    def test_derives_offset_and_vocab_size(self):
        tok = _synthetic_tokenizer(
            ["<|img_start|>", "<|img_end|>"]
            + [f"<|visual token {i}|>" for i in range(3)]
        )
        omc = build_omnimodal_config(1, tok, registry={"vision": VISION})
        (entry,) = omc["modalities"]
        assert entry["vocab_size"] == 3
        assert entry["offset"] == tok.convert_tokens_to_ids("<|visual token 0|>")

    def test_rejects_gapped_content_ids(self):
        tok = _synthetic_tokenizer(
            ["<|img_start|>", "<|img_end|>", "<|visual token 0|>", "<gap>"]
            + [f"<|visual token {i}|>" for i in range(1, 3)]
        )
        with pytest.raises(ValueError, match="not contiguous"):
            build_omnimodal_config(1, tok, registry={"vision": VISION})

    def test_rejects_deleted_mid_range_tokens(self):
        tok = _synthetic_tokenizer(
            ["<|img_start|>", "<|img_end|>", "<|visual token 0|>", "<|visual token 1|>"]
            + ["<|visual token 4|>"]
        )
        with pytest.raises(ValueError, match="contiguous"):
            build_omnimodal_config(1, tok, registry={"vision": VISION})

    def test_absent_modality_yields_empty_config(self):
        tok = _synthetic_tokenizer(["<|img_start|>"])
        assert build_omnimodal_config(1, tok, registry={"vision": VISION}) == {}

    def test_zero_vocab_size_rejected(self, tmp_path):
        base = str(tmp_path / "base")
        _synthetic_tokenizer([]).save_pretrained(base)
        with pytest.raises(ValueError, match="must be positive"):
            add_modality(base, str(tmp_path / "out"), "vision", 0)


# ── Shipped artifact compatibility ───────────────────────────────────────────


SHIPPED_1P5 = os.path.join(os.path.dirname(__file__), "..", "tokenizers", "Apertus_1p5")


class TestShipped1p5:
    """The config-based readers work against the artifact built by the old code."""

    def test_detect(self):
        det = detect_existing_modalities(SHIPPED_1P5)
        assert det["base_vocab_size"] == 131072
        assert det["modalities"]["vision"]["vocab_size"] == 131072
        assert det["modalities"]["audio"]["vocab_size"] == 4096

    def test_content_token_id(self):
        assert get_content_token_id(0, SHIPPED_1P5, "vision") == 131272
        assert get_content_token_id(131071, SHIPPED_1P5, "vision") == 131272 + 131071
        assert get_content_token_id(4095, SHIPPED_1P5, "audio") == 262344 + 4095


class TestRenameReservedTokens:
    @staticmethod
    def _saved(tmp_path, tokens):
        tok = _synthetic_tokenizer(tokens, special=True)
        path = str(tmp_path / "tok")
        tok.save_pretrained(path)
        return tok, path

    def test_config_string_values_are_renamed(self, tmp_path):
        tok, path = self._saved(tmp_path, ["<A>"])
        config_path = os.path.join(path, "tokenizer_config.json")
        with open(config_path) as f:
            config = json.load(f)
        config["probe"] = "<A>"
        with open(config_path, "w") as f:
            json.dump(config, f)

        rename_reserved_tokens(path, tok, {"<A>": "<|img_start|>"})

        with open(config_path) as f:
            config = json.load(f)
        assert config["probe"] == "<|img_start|>"

    def test_special_tokens_map_is_renamed(self, tmp_path):
        tok, path = self._saved(tmp_path, ["<A>"])
        stm_path = os.path.join(path, "special_tokens_map.json")
        with open(stm_path, "w") as f:
            json.dump({"additional_special_tokens": ["<A>"]}, f)

        rename_reserved_tokens(path, tok, {"<A>": "<|img_start|>"})

        with open(stm_path) as f:
            stm = json.load(f)
        assert stm["additional_special_tokens"] == ["<|img_start|>"]

    def test_all_absent_leaves_files_byte_identical(self, tmp_path):
        tok, path = self._saved(tmp_path, ["<A>"])
        before = {
            name: open(os.path.join(path, name), "rb").read()
            for name in ("tokenizer.json", "tokenizer_config.json")
        }
        rename_reserved_tokens(path, tok, {"<MISSING>": "<X>"})
        for name, content in before.items():
            assert open(os.path.join(path, name), "rb").read() == content

    def test_rejects_chained_renames(self, tmp_path):
        tok, path = self._saved(tmp_path, ["<A>", "<B>"])
        with pytest.raises(ValueError, match="overlap"):
            rename_reserved_tokens(path, tok, {"<A>": "<B>", "<B>": "<C>"})

    def test_rejects_duplicate_targets(self, tmp_path):
        tok, path = self._saved(tmp_path, ["<A>", "<B>"])
        with pytest.raises(ValueError, match="duplicate"):
            rename_reserved_tokens(path, tok, {"<A>": "<C>", "<B>": "<C>"})

    def test_rejects_existing_target(self, tmp_path):
        tok, path = self._saved(tmp_path, ["<A>", "<B>"])
        with pytest.raises(ValueError, match="already in the vocabulary"):
            rename_reserved_tokens(path, tok, {"<A>": "<B>"})

    def test_absent_source_with_existing_target_skips(self, tmp_path):
        tok, path = self._saved(tmp_path, ["<|image|>"])
        before = open(os.path.join(path, "tokenizer.json"), "rb").read()
        rename_reserved_tokens(path, tok, {"<|RESERVED_OMNI_007|>": "<|image|>"})
        assert open(os.path.join(path, "tokenizer.json"), "rb").read() == before

    def test_ids_never_move(self, tmp_path):
        tok, path = self._saved(tmp_path, ["<A>", "<B>"])
        renamed_id = tok.convert_tokens_to_ids("<A>")
        bystander_id = tok.convert_tokens_to_ids("<B>")
        rename_reserved_tokens(path, tok, {"<A>": "<|img_start|>"})
        after = AutoTokenizer.from_pretrained(path)
        assert after.convert_tokens_to_ids("<|img_start|>") == renamed_id
        assert after.convert_tokens_to_ids("<B>") == bystander_id
