"""Tests for omnitok.builder.add_modality()."""

import json
import os

import pytest
from transformers import AutoTokenizer

from omnitok import (
    add_modality,
    detect_existing_modalities,
    get_content_token_id,
    load_modality_mapping,
)
from omnitok.io import build_omnimodal_config
from omnitok.modalities import VISION, AUDIO
from tokenizer_factory import make_word_level_tokenizer

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


class TestOmnimodalDerivation:
    @staticmethod
    def _synthetic(tokens):
        return make_word_level_tokenizer(added_tokens=tokens)

    def test_derives_offset_and_vocab_size(self):
        tok = self._synthetic(
            ["<|img_start|>", "<|img_end|>"]
            + [f"<|visual token {i}|>" for i in range(3)]
        )
        omc = build_omnimodal_config(1, tok, registry={"vision": VISION})
        (entry,) = omc["modalities"]
        assert entry["vocab_size"] == 3
        assert entry["offset"] == tok.convert_tokens_to_ids("<|visual token 0|>")

    def test_rejects_gapped_content_ids(self):
        tok = self._synthetic(
            ["<|img_start|>", "<|img_end|>", "<|visual token 0|>", "<gap>"]
            + [f"<|visual token {i}|>" for i in range(1, 3)]
        )
        with pytest.raises(ValueError, match="not contiguous"):
            build_omnimodal_config(1, tok, registry={"vision": VISION})

    def test_rejects_deleted_mid_range_tokens(self):
        tok = self._synthetic(
            ["<|img_start|>", "<|img_end|>", "<|visual token 0|>", "<|visual token 1|>"]
            + ["<|visual token 4|>"]
        )
        with pytest.raises(ValueError, match="contiguous"):
            build_omnimodal_config(1, tok, registry={"vision": VISION})

    def test_absent_modality_yields_empty_config(self):
        tok = self._synthetic(["<|img_start|>"])
        assert build_omnimodal_config(1, tok, registry={"vision": VISION}) == {}

    def test_zero_vocab_size_rejected(self, tmp_path):
        base = str(tmp_path / "base")
        self._synthetic([]).save_pretrained(base)
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
        vision = load_modality_mapping(SHIPPED_1P5, "vision")
        audio = load_modality_mapping(SHIPPED_1P5, "audio")
        assert get_content_token_id(0, mapping=vision) == 131272
        assert get_content_token_id(131071, mapping=vision) == 131272 + 131071
        assert get_content_token_id(4095, mapping=audio) == 262344 + 4095


# ── Offline end-to-end build ─────────────────────────────────────────────────


class TestAddModalityOffline:
    def test_full_build_on_synthetic_base(self, tmp_path):
        base = str(tmp_path / "base")
        make_word_level_tokenizer(("<unk>", "hi")).save_pretrained(base)
        out = str(tmp_path / "vision")
        tok, stats = add_modality(base, out, "vision", 4)

        entry = _omnimodal_entry(out, "vision")
        assert entry["vocab_size"] == 4
        assert entry["offset"] == tok.convert_tokens_to_ids("<|visual token 0|>")
        for name, tid in entry["structure_token_ids"].items():
            assert tok.convert_tokens_to_ids(name) == tid
        assert "<|img_start|>" in entry["structure_token_ids"]

        vocab = tok.get_vocab()
        for r in VISION.structure_tokens:
            assert f"<|RESERVED_OMNI_{r.reserved_index:03d}|>" not in vocab
        img = tok.convert_tokens_to_ids("<|image|>")
        assert tok.encode("<image>", add_special_tokens=False) == [img]
        assert not os.path.exists(os.path.join(out, "vision_token_mapping.json"))

    def test_stacking_on_synthetic_base(self, tmp_path):
        base = str(tmp_path / "base")
        make_word_level_tokenizer(("<unk>", "hi")).save_pretrained(base)
        vis = str(tmp_path / "vis")
        va = str(tmp_path / "va")
        add_modality(base, vis, "vision", 4)
        tok, _ = add_modality(vis, va, "audio", 2)

        vision = _omnimodal_entry(va, "vision")
        audio = _omnimodal_entry(va, "audio")
        assert audio["offset"] == vision["offset"] + 4
        assert audio["structure_token_ids"]["<|audio_start|>"] == tok.convert_tokens_to_ids("<|audio_start|>")


# ── In-place build (synthetic v2-shaped base) ────────────────────────────────


class TestInPlaceOffline:
    @staticmethod
    def _v2_shaped_base(tmp_path):
        pool = [f"<SPECIAL_{i}>" for i in range(1, 15)]
        tok = make_word_level_tokenizer(
            ("<unk>", "hi"), added_tokens=["<|image|>", "<|audio|>", *pool]
        )
        base = str(tmp_path / "base")
        tok.save_pretrained(base)
        return base, tok

    def test_vision_then_audio(self, tmp_path):
        from omnitok.builder import add_modality_in_place

        base, btok = self._v2_shaped_base(tmp_path)
        img = btok.convert_tokens_to_ids("<|image|>")
        aud = btok.convert_tokens_to_ids("<|audio|>")
        out = str(tmp_path / "omni")

        vision_targets = [
            "<|img_start|>", "<|img_end|>", "<|img_token_start|>",
            "<|img_end_of_row|>", "<|img_end_of_frame|>", "<|img_generation_start|>",
        ]
        audio_targets = [
            "<|audio_start|>", "<|audio_end|>", "<|stt_transcribe|>",
            "<|stt_continue|>", "<|tts_continue|>", "<|stt_translate|>",
            "<|audio_annotate|>",
        ]
        add_modality_in_place(
            base, out, "vision", 4,
            renames={f"<SPECIAL_{i}>": t for i, t in enumerate(vision_targets, start=1)},
            reused_ids={"<|image|>": img},
        )
        tok, stats = add_modality_in_place(
            out, out, "audio", 2,
            renames={f"<SPECIAL_{i}>": t for i, t in enumerate(audio_targets, start=7)},
            reused_ids={"<|audio|>": aud},
        )

        vision = _omnimodal_entry(out, "vision")
        audio = _omnimodal_entry(out, "audio")
        assert vision["structure_token_ids"]["<|img_start|>"] == btok.convert_tokens_to_ids("<SPECIAL_1>")
        assert vision["structure_token_ids"]["<|image|>"] == img
        assert audio["offset"] == vision["offset"] + 4
        assert stats["reserved_tokens_added"] == 0

        assert tok.encode("<image>", add_special_tokens=False) == [img]
        assert tok.encode("<audio>", add_special_tokens=False) == [aud]
        assert "<SPECIAL_1>" not in tok.get_vocab()

        with open(os.path.join(out, "special_tokens_map.json")) as f:
            assert "additional_special_tokens" not in json.load(f)

    def test_base_missing_pool_slot_raises(self, tmp_path):
        from omnitok.builder import add_modality_in_place

        base, _ = self._v2_shaped_base(tmp_path)
        with pytest.raises(ValueError, match="missing reserve slots"):
            add_modality_in_place(
                base, str(tmp_path / "o"), "vision", 4,
                renames={"<SPECIAL_99>": "<|img_start|>"}, reused_ids={},
            )

    def test_wrong_reused_id_raises(self, tmp_path):
        from omnitok.builder import add_modality_in_place

        base, btok = self._v2_shaped_base(tmp_path)
        wrong = btok.convert_tokens_to_ids("<|image|>") + 1
        with pytest.raises(ValueError, match="expected"):
            add_modality_in_place(
                base, str(tmp_path / "o"), "vision", 4,
                renames={"<SPECIAL_1>": "<|img_start|>", "<SPECIAL_2>": "<|img_end|>"},
                reused_ids={"<|image|>": wrong},
            )

    def test_bos_eos_post_processor_is_stripped(self, tmp_path):
        from tokenizers.processors import TemplateProcessing

        from omnitok.builder import add_modality_in_place

        pool = [f"<SPECIAL_{i}>" for i in range(1, 15)]
        tok = make_word_level_tokenizer(
            ("<unk>", "hi"), bos_eos=True,
            added_tokens=["<|image|>", "<|audio|>", *pool],
        )
        tok.backend_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[
                ("<s>", tok.convert_tokens_to_ids("<s>")),
                ("</s>", tok.convert_tokens_to_ids("</s>")),
            ],
        )
        base = str(tmp_path / "base")
        tok.save_pretrained(base)

        out = str(tmp_path / "omni")
        built, _ = add_modality_in_place(
            base, out, "vision", 4,
            renames={"<SPECIAL_1>": "<|img_start|>", "<SPECIAL_2>": "<|img_end|>"},
            reused_ids={"<|image|>": tok.convert_tokens_to_ids("<|image|>")},
        )
        with open(os.path.join(out, "tokenizer.json")) as f:
            assert json.load(f)["post_processor"] is None
        hi = built.convert_tokens_to_ids("hi")
        assert built.encode("hi", add_special_tokens=True) == [hi]

    def test_unrecognized_post_processor_raises(self, tmp_path):
        from tokenizers.processors import TemplateProcessing

        from omnitok.builder import add_modality_in_place

        tok = make_word_level_tokenizer(
            ("<unk>", "hi", "<x>"), added_tokens=["<|image|>", "<SPECIAL_1>", "<SPECIAL_2>"]
        )
        tok.backend_tokenizer.post_processor = TemplateProcessing(
            single="<x> $A",
            special_tokens=[("<x>", tok.convert_tokens_to_ids("<x>"))],
        )
        base = str(tmp_path / "base")
        tok.save_pretrained(base)
        with pytest.raises(ValueError, match="unrecognized base post-processor"):
            add_modality_in_place(
                base, str(tmp_path / "o"), "vision", 4,
                renames={"<SPECIAL_1>": "<|img_start|>", "<SPECIAL_2>": "<|img_end|>"},
                reused_ids={},
            )


def test_is_apertus_1p5_gate():
    """The 1.5 reasoning fix is gated on the emitted delimiter ids: it applies
    only when <|inner_prefix|>/<|inner_suffix|> sit at 32/33 (Apertus 1.5), and
    is skipped for Apertus 1.0 (which has them at 69/70) so a 1.0 rebuild is left
    unchanged."""
    from omnitok.builder import _is_apertus_1p5

    class _Tok:
        def __init__(self, ids):
            self._ids = ids

        def convert_tokens_to_ids(self, token):
            return self._ids.get(token, 0)

    assert _is_apertus_1p5(_Tok({"<|inner_prefix|>": 32, "<|inner_suffix|>": 33}))
    assert not _is_apertus_1p5(_Tok({"<|inner_prefix|>": 69, "<|inner_suffix|>": 70}))
    assert not _is_apertus_1p5(_Tok({"<|inner_prefix|>": 32}))  # partial -> no
