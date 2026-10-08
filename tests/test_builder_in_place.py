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

from omnitok.builder import _strip_post_processor, add_modality_in_place
from omnitok.modalities import AUDIO, VISION
from tokenizer_factory import make_word_level_tokenizer

POOL = {
    "<SPECIAL_1>": "<|img_start|>",
    "<SPECIAL_2>": "<|img_end|>",
    "<SPECIAL_3>": "<|img_token_start|>",
    "<SPECIAL_4>": "<|img_end_of_row|>",
    "<SPECIAL_5>": "<|img_end_of_frame|>",
    "<SPECIAL_6>": "<|img_generation_start|>",
}
AUDIO_POOL = {
    "<SPECIAL_7>": "<|audio_start|>",
    "<SPECIAL_8>": "<|audio_end|>",
    "<SPECIAL_9>": "<|stt_transcribe|>",
}
CONTENT = 8


def _base(tmp_path, *, drop=()):
    """A tiny Apertus-2-shaped base: reserve pool, pre-baked <|image|>,
    and a BOS-inserting post-processor."""
    added = [s for s in POOL if s not in drop] + ["<|image|>"]
    tok = make_word_level_tokenizer(
        ("<unk>", "hi", "there"), bos_eos=True, added_tokens=added, added_special=True
    )
    bos = tok.convert_tokens_to_ids("<s>")
    tok.backend_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A", pair="<s> $A $B", special_tokens=[("<s>", bos)]
    )
    out = str(tmp_path / "base")
    tok.save_pretrained(out)
    return out, tok.convert_tokens_to_ids("<|image|>")


def _build(tmp_path, base, image_id, **kw):
    out = str(tmp_path / "omni")
    kw.setdefault("strip_post_processor", True)
    return add_modality_in_place(
        base, out, VISION, CONTENT,
        renames=dict(POOL), reused_ids={"<|image|>": image_id}, **kw
    ), out


class TestInPlaceBuild:
    def test_id_layout(self, tmp_path):
        """Pool slots are renamed where they sit, the placeholder is reused,
        and only content tokens are appended, contiguously."""
        base, image_id = _base(tmp_path)
        before = AutoTokenizer.from_pretrained(base)
        pinned = {s: before.convert_tokens_to_ids(s) for s in POOL}

        _, out = _build(tmp_path, base, image_id)

        after = AutoTokenizer.from_pretrained(out)
        for src, target in POOL.items():
            assert after.convert_tokens_to_ids(target) == pinned[src], target
        assert after.convert_tokens_to_ids("<|image|>") == image_id
        assert len(after) == len(before) + CONTENT
        ids = [after.convert_tokens_to_ids(f"<|visual token {i}|>") for i in range(CONTENT)]
        assert ids == list(range(len(before), len(before) + CONTENT))

    def test_post_processor_is_stripped_from_the_written_file(self, tmp_path):
        """save_pretrained re-adds an empty TemplateProcessing on 5.x, so the
        strip has to survive every write the pipeline performs."""
        base, image_id = _base(tmp_path)

        _, out = _build(tmp_path, base, image_id)

        state = json.load(open(os.path.join(out, "tokenizer.json")))
        assert state["post_processor"] is None

    def test_post_processor_is_kept_unless_the_recipe_strips_it(self, tmp_path):
        base, image_id = _base(tmp_path)

        _, out = _build(tmp_path, base, image_id, strip_post_processor=False)

        tok = AutoTokenizer.from_pretrained(out)
        assert tok.encode("hi")[0] == tok.bos_token_id

    @pytest.mark.parametrize("publish", [True, False])
    def test_structure_token_ids_are_published_only_on_request(self, tmp_path, publish):
        base, image_id = _base(tmp_path)

        _, out = _build(tmp_path, base, image_id, publish_structure_ids=publish)

        cfg = json.load(open(os.path.join(out, "tokenizer_config.json")))
        (vision,) = cfg["omnimodal_config"]["modalities"]
        if not publish:
            assert "structure_token_ids" not in vision
            return
        tok = AutoTokenizer.from_pretrained(out)
        assert vision["structure_token_ids"]["<|img_end_of_row|>"] == \
            tok.convert_tokens_to_ids("<|img_end_of_row|>")


class TestStackedInPlaceBuild:
    """Vision, then audio, each renaming its own slots of one shared pool."""

    @pytest.fixture
    def stacked(self, tmp_path):
        added = list(POOL) + list(AUDIO_POOL) + ["<|image|>", "<|audio|>"]
        tok = make_word_level_tokenizer(
            ("<unk>", "hi"), bos_eos=True, added_tokens=added, added_special=True
        )
        base = str(tmp_path / "base")
        tok.save_pretrained(base)
        pinned = {t: tok.convert_tokens_to_ids(t) for t in added}
        common = dict(strip_post_processor=True, publish_structure_ids=True)
        vision = str(tmp_path / "vision")
        add_modality_in_place(
            base, vision, VISION, CONTENT, renames=dict(POOL),
            reused_ids={"<|image|>": pinned["<|image|>"]}, **common,
        )
        out = str(tmp_path / "vision_audio")
        add_modality_in_place(
            vision, out, AUDIO, CONTENT, renames=dict(AUDIO_POOL),
            reused_ids={"<|audio|>": pinned["<|audio|>"]}, **common,
        )
        return out, pinned, len(tok)

    def test_both_modalities_keep_their_slots_and_get_disjoint_content(self, stacked):
        out, pinned, base_size = stacked
        cfg = json.load(open(os.path.join(out, "tokenizer_config.json")))
        vision, audio = cfg["omnimodal_config"]["modalities"]
        assert (vision["name"], vision["offset"]) == ("vision", base_size)
        assert (audio["name"], audio["offset"]) == ("audio", base_size + CONTENT)
        assert vision["vocab_size"] == audio["vocab_size"] == CONTENT

        for pool, published in ((POOL, vision), (AUDIO_POOL, audio)):
            for slot, name in pool.items():
                assert published["structure_token_ids"][name] == pinned[slot], name
        assert not (
            set(vision["structure_token_ids"].values())
            & set(audio["structure_token_ids"].values())
        )

    def test_both_aliases_survive_stacking(self, stacked):
        out, pinned, _ = stacked
        tok = AutoTokenizer.from_pretrained(out)
        assert tok.encode("<image>", add_special_tokens=False) == [pinned["<|image|>"]]
        assert tok.encode("<audio>", add_special_tokens=False) == [pinned["<|audio|>"]]


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
        # transformers 5.x attaches a TemplateProcessing on construction.
        tok.backend_tokenizer.post_processor = None
        _strip_post_processor(tok)
        assert tok.backend_tokenizer.post_processor is None
