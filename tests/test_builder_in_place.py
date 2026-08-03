"""Tests for omnitok.builder.add_modality_in_place().

The in-place strategy targets bases that pre-bake their special tokens and
ship a reserve pool: slots are renamed where they sit, placeholders are
reused at their pinned ids, and only content tokens are appended.
Synthetic bases throughout -- no network, no real artifact.
"""

import json
import os

import pytest
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from omnitok.builder import (
    _assert_in_place_base,
    _strip_post_processor,
    add_modality_in_place,
)
from omnitok.modalities import VISION
from tokenizer_factory import make_word_level_tokenizer

POOL = {
    "<SPECIAL_1>": "<|img_start|>",
    "<SPECIAL_2>": "<|img_end|>",
    "<SPECIAL_3>": "<|img_token_start|>",
    "<SPECIAL_4>": "<|img_end_of_row|>",
    "<SPECIAL_5>": "<|img_end_of_frame|>",
    "<SPECIAL_6>": "<|img_generation_start|>",
}
CONTENT = 8


def _base(tmp_path, *, post_processor=True, drop=(), image_id=None):
    """A tiny Apertus-2-shaped base: reserve pool, pre-baked <|image|>."""
    added = [s for s in POOL if s not in drop] + ["<|image|>"]
    tok = make_word_level_tokenizer(
        ("<unk>", "hi", "there"), bos_eos=True, added_tokens=added, added_special=True
    )
    if post_processor:
        tok.backend_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A", pair="<s> $A $B", special_tokens=[("<s>", 1)]
        )
    out = str(tmp_path / "base")
    tok.save_pretrained(out)
    return out, tok.convert_tokens_to_ids("<|image|>")


def _build(tmp_path, base, image_id, **kw):
    out = str(tmp_path / "omni")
    return add_modality_in_place(
        base, out, VISION, CONTENT,
        renames=dict(POOL), reused_ids={"<|image|>": image_id}, **kw
    ), out


class TestInPlaceBuild:
    def test_ids_never_move_and_placeholder_is_reused(self, tmp_path):
        base, image_id = _base(tmp_path)
        before = AutoTokenizer.from_pretrained(base)
        pinned = {s: before.convert_tokens_to_ids(s) for s in POOL}

        _, out = _build(tmp_path, base, image_id)

        after = AutoTokenizer.from_pretrained(out)
        for src, target in POOL.items():
            assert after.convert_tokens_to_ids(target) == pinned[src], target
        assert after.convert_tokens_to_ids("<|image|>") == image_id

    def test_content_tokens_are_appended_contiguously(self, tmp_path):
        base, image_id = _base(tmp_path)
        base_size = len(AutoTokenizer.from_pretrained(base))

        _, out = _build(tmp_path, base, image_id)

        after = AutoTokenizer.from_pretrained(out)
        assert len(after) == base_size + CONTENT
        ids = [after.convert_tokens_to_ids(f"<|visual token {i}|>") for i in range(CONTENT)]
        assert ids == list(range(base_size, base_size + CONTENT))

    def test_post_processor_is_stripped_from_the_written_file(self, tmp_path):
        """save_pretrained re-adds an empty TemplateProcessing on 5.x, so the
        strip has to survive every write the pipeline performs."""
        base, image_id = _base(tmp_path)

        _, out = _build(tmp_path, base, image_id)

        state = json.load(open(os.path.join(out, "tokenizer.json")))
        assert state["post_processor"] is None

    def test_structure_token_ids_are_published_on_request(self, tmp_path):
        base, image_id = _base(tmp_path)

        _, out = _build(tmp_path, base, image_id, publish_structure_ids=True)

        cfg = json.load(open(os.path.join(out, "tokenizer_config.json")))
        (vision,) = cfg["omnimodal_config"]["modalities"]
        assert vision["structure_token_ids"]["<|img_end_of_row|>"] == \
            AutoTokenizer.from_pretrained(out).convert_tokens_to_ids("<|img_end_of_row|>")

    def test_structure_token_ids_are_absent_by_default(self, tmp_path):
        base, image_id = _base(tmp_path)

        _, out = _build(tmp_path, base, image_id)

        cfg = json.load(open(os.path.join(out, "tokenizer_config.json")))
        (vision,) = cfg["omnimodal_config"]["modalities"]
        assert "structure_token_ids" not in vision


class TestInPlaceBaseAssertions:
    def test_missing_pool_slot_is_rejected(self, tmp_path):
        base, image_id = _base(tmp_path, drop=("<SPECIAL_4>",))
        with pytest.raises(ValueError, match="missing reserve slots"):
            _build(tmp_path, base, image_id)

    def test_existing_rename_target_is_rejected(self, tmp_path):
        tok = make_word_level_tokenizer(
            ("<unk>", "hi"), bos_eos=True,
            added_tokens=list(POOL) + ["<|image|>", "<|img_start|>"],
            added_special=True,
        )
        base = str(tmp_path / "base")
        tok.save_pretrained(base)
        with pytest.raises(ValueError, match="already exist in the base"):
            _build(tmp_path, base, tok.convert_tokens_to_ids("<|image|>"))

    def test_reused_id_at_the_wrong_position_is_rejected(self, tmp_path):
        base, image_id = _base(tmp_path)
        with pytest.raises(ValueError, match="expected"):
            _build(tmp_path, base, image_id + 1)

    def test_unexpected_base_vocab_size_is_rejected(self, tmp_path):
        base, image_id = _base(tmp_path)
        with pytest.raises(ValueError, match="base vocab is"):
            _build(tmp_path, base, image_id, expected_base_vocab_size=999999)


class TestStripPostProcessor:
    def test_drops_a_template_over_the_declared_specials(self, tmp_path):
        tok = make_word_level_tokenizer(("<unk>", "hi"), bos_eos=True)
        tok.backend_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A", pair="<s> $A $B", special_tokens=[("<s>", 1)]
        )
        _strip_post_processor(tok)
        assert tok.backend_tokenizer.post_processor is None

    def test_refuses_a_template_over_foreign_specials(self, tmp_path):
        tok = make_word_level_tokenizer(
            ("<unk>", "hi", "<|weird|>"), bos_eos=True,
        )
        tok.backend_tokenizer.post_processor = TemplateProcessing(
            single="<|weird|> $A", pair="<|weird|> $A $B",
            special_tokens=[("<|weird|>", 2)],
        )
        with pytest.raises(ValueError, match="unrecognized"):
            _strip_post_processor(tok)

    def test_no_post_processor_is_a_no_op(self, tmp_path):
        tok = make_word_level_tokenizer(("<unk>", "hi"), bos_eos=True)
        _strip_post_processor(tok)
        assert tok.backend_tokenizer.post_processor is None


def test_assert_in_place_base_accepts_a_matching_base(tmp_path):
    base, image_id = _base(tmp_path)
    tok = AutoTokenizer.from_pretrained(base)
    _assert_in_place_base(tok, dict(POOL), {"<|image|>": image_id})
