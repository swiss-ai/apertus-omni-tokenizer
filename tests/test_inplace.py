"""Tests for in-place reserved-token allocation (allocation="in_place").

These run against a base tokenizer that pre-bakes specials inside its vocab
(<|image|>, <|audio|>, and a <SPECIAL_*> reserve pool), e.g. the 200k tokenizer.
The fixture path is external; tests skip when it is unavailable (e.g. CI).
"""

import json
import os
import re
import shutil

import pytest
from transformers import AutoTokenizer

from omnitok import add_modality
from omnitok.modalities import AUDIO_INPLACE, VISION_INPLACE

INPLACE_BASE = os.environ.get(
    "OMNI_INPLACE_FIXTURE",
    "/users/rkreft/apertus/apertus-tokenizer-development/preliminary_mul_200k",
)
SMALL_VOCAB = 32
POOL_RE = re.compile(r"^<SPECIAL_(\d+)>$")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(INPLACE_BASE),
    reason=f"in-place fixture tokenizer not available at {INPLACE_BASE}",
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _base_config():
    with open(os.path.join(INPLACE_BASE, "tokenizer_config.json")) as f:
        return json.load(f)


def _base_token_id(name):
    for tid, meta in _base_config()["added_tokens_decoder"].items():
        if meta["content"] == name:
            return int(tid)
    return None


def _base_pool_ids():
    """Sorted ids of the free <SPECIAL_*> pool in the base tokenizer."""
    ids = []
    for tid, meta in _base_config()["added_tokens_decoder"].items():
        if POOL_RE.match(meta["content"]):
            ids.append(int(tid))
    return sorted(ids)


def _base_pool_ordinals():
    """Sorted ordinals (the N in <SPECIAL_N>) of the free pool -- ordinals, not ids
    (they coincide in this fixture; slot_assignments take ordinals)."""
    ords = []
    for meta in _base_config()["added_tokens_decoder"].values():
        m = POOL_RE.match(meta["content"])
        if m:
            ords.append(int(m.group(1)))
    return sorted(ords)


def _decoder_normalized(tok_dir, name):
    with open(os.path.join(tok_dir, "tokenizer_config.json")) as f:
        cfg = json.load(f)
    for meta in cfg["added_tokens_decoder"].values():
        if meta["content"] == name:
            return meta.get("normalized")
    return None


def _count_replace(tok_dir, alias):
    with open(os.path.join(tok_dir, "tokenizer.json")) as f:
        tj = json.load(f)

    def walk(n):
        if n is None:
            return
        if n.get("type") == "Sequence":
            for x in n["normalizers"]:
                yield from walk(x)
        else:
            yield n

    return sum(
        1
        for n in walk(tj.get("normalizer"))
        if n.get("type") == "Replace" and n.get("pattern", {}).get("String") == alias
    )


def _mapping(tok_dir, fname):
    with open(os.path.join(tok_dir, fname)) as f:
        return json.load(f)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def inplace_vision(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("ip_vision"))
    add_modality(INPLACE_BASE, out, "vision", SMALL_VOCAB, allocation="in_place")
    return out


@pytest.fixture(scope="module")
def inplace_stacked(tmp_path_factory):
    vis = str(tmp_path_factory.mktemp("ip_v"))
    add_modality(INPLACE_BASE, vis, "vision", SMALL_VOCAB, allocation="in_place")
    both = str(tmp_path_factory.mktemp("ip_va"))
    add_modality(vis, both, "audio", SMALL_VOCAB, allocation="in_place")
    return both


# ── reuse (requirement 1) ────────────────────────────────────────────────────


def test_image_audio_reused(inplace_stacked):
    tok = AutoTokenizer.from_pretrained(inplace_stacked)
    assert tok.convert_tokens_to_ids("<|image|>") == _base_token_id("<|image|>")
    assert tok.convert_tokens_to_ids("<|audio|>") == _base_token_id("<|audio|>")


def test_reused_ids_outside_pool(inplace_stacked):
    cfg = _mapping(inplace_stacked, "tokenizer_config.json")["omnimodal_config"]
    pool_lo, pool_hi = (
        cfg["omni_special_token_offset"],
        cfg["omni_special_token_offset"] + cfg["omni_special_token_count"] - 1,
    )
    for name in ("<|image|>", "<|audio|>"):
        rid = _base_token_id(name)
        assert not (pool_lo <= rid <= pool_hi), f"{name} should be outside the pool"


# ── auto pool (requirement 2a) ───────────────────────────────────────────────


def test_auto_pool_lowest_free_in_order(inplace_vision):
    tok = AutoTokenizer.from_pretrained(inplace_vision)
    pool_tokens = [r.target_name for r in VISION_INPLACE.structure_tokens
                   if r.source == "pool"]
    expected = _base_pool_ids()[: len(pool_tokens)]
    actual = [tok.convert_tokens_to_ids(t) for t in pool_tokens]
    assert actual == expected


def test_stacking_audio_after_vision_pool(inplace_stacked):
    tok = AutoTokenizer.from_pretrained(inplace_stacked)
    n_vis = sum(r.source == "pool" for r in VISION_INPLACE.structure_tokens)
    n_aud = sum(r.source == "pool" for r in AUDIO_INPLACE.structure_tokens)
    pool = _base_pool_ids()
    aud_tokens = [r.target_name for r in AUDIO_INPLACE.structure_tokens
                  if r.source == "pool"]
    expected = pool[n_vis: n_vis + n_aud]
    actual = [tok.convert_tokens_to_ids(t) for t in aud_tokens]
    assert actual == expected


# ── content tokens still appended at the end ─────────────────────────────────


def test_content_appended_after_base(inplace_vision):
    data = _mapping(inplace_vision, "vision_token_mapping.json")
    base = _base_config().get("base_vocab_size") or len(
        AutoTokenizer.from_pretrained(INPLACE_BASE)
    )
    assert data["vision_token_offset"] >= base
    ids = [data["vision_token_ids"][str(i)] for i in range(SMALL_VOCAB)]
    assert ids == list(range(ids[0], ids[0] + SMALL_VOCAB))


# ── aliases (requirement 3) ──────────────────────────────────────────────────


def test_alias_single_id_fresh_reload(inplace_vision):
    tok = AutoTokenizer.from_pretrained(inplace_vision)
    img = tok.convert_tokens_to_ids("<|image|>")
    assert tok.encode("<|image|>", add_special_tokens=False) == [img]
    assert tok.encode("<image>", add_special_tokens=False) == [img]


def test_reused_token_normalized_flipped_on_disk(inplace_vision):
    assert _decoder_normalized(inplace_vision, "<|image|>") is True
    with open(os.path.join(inplace_vision, "tokenizer.json")) as f:
        tj = json.load(f)
    entry = next(e for e in tj["added_tokens"] if e["content"] == "<|image|>")
    assert entry["normalized"] is True


def test_audio_alias_single_id(inplace_stacked):
    tok = AutoTokenizer.from_pretrained(inplace_stacked)
    aid = tok.convert_tokens_to_ids("<|audio|>")
    assert tok.encode("<audio>", add_special_tokens=False) == [aid]


# ── metadata (requirement 4) ─────────────────────────────────────────────────


def test_omnimodal_config_shape(inplace_stacked):
    cfg = _mapping(inplace_stacked, "tokenizer_config.json")["omnimodal_config"]
    assert cfg["allocation"] == "in_place"
    n_pool = sum(r.source == "pool" for r in VISION_INPLACE.structure_tokens) + sum(
        r.source == "pool" for r in AUDIO_INPLACE.structure_tokens
    )
    assert cfg["omni_special_token_count"] == n_pool
    assert cfg["omni_special_token_offset"] == _base_pool_ids()[0]
    # explicit union includes the reused image/audio ids
    assert _base_token_id("<|image|>") in cfg["omni_special_token_ids"]
    assert _base_token_id("<|audio|>") in cfg["omni_special_token_ids"]


def test_per_modality_region(inplace_stacked):
    cfg = _mapping(inplace_stacked, "tokenizer_config.json")["omnimodal_config"]
    by_name = {m["name"]: m for m in cfg["modalities"]}
    vis = by_name["vision"]
    assert vis["reused_special_ids"] == [_base_token_id("<|image|>")]
    assert vis["structure_token_ids"]["image"] == _base_token_id("<|image|>")
    assert vis["special_region_count"] == sum(
        r.source == "pool" for r in VISION_INPLACE.structure_tokens
    )


def test_content_not_in_named_special_role(inplace_stacked):
    # Mirror Apertus 1.5: content tokens are atomic special-flag tokens but are NOT
    # enrolled in the additional_special_tokens named role, so special_tokens_map stays
    # clean (bos/eos/pad/unk only) and content ids stay out of all_special_ids.
    stm = _mapping(inplace_stacked, "special_tokens_map.json")
    assert not stm.get("additional_special_tokens"), \
        "content must not populate the additional_special_tokens named role"
    tok = AutoTokenizer.from_pretrained(inplace_stacked)
    for name in ("<|visual token 0|>", "<|audio token 0|>"):
        tid = tok.convert_tokens_to_ids(name)
        assert tok.encode(name, add_special_tokens=False) == [tid]  # still atomic
        assert tid not in tok.all_special_ids                       # but not "named"


# ── idempotency (re-run) ─────────────────────────────────────────────────────


def test_rerun_single_replace_and_single_id(inplace_vision, tmp_path):
    work = str(tmp_path / "rerun")
    shutil.copytree(inplace_vision, work)
    add_modality(work, work, "vision", SMALL_VOCAB, allocation="in_place")
    assert _count_replace(work, "<image>") == 1
    tok = AutoTokenizer.from_pretrained(work)
    img = tok.convert_tokens_to_ids("<|image|>")
    assert tok.encode("<image>", add_special_tokens=False) == [img]


# ── explicit allocation + validation (requirement 2b) ────────────────────────


def test_explicit_slot_lands_at_requested(tmp_path):
    out = str(tmp_path / "explicit")
    # slot_assignments values are pool ORDINALS (the N in <SPECIAL_N>), not ids.
    ordinal = _base_pool_ordinals()[10]  # a free pool slot past the auto range
    add_modality(
        INPLACE_BASE, out, "vision", SMALL_VOCAB, allocation="in_place",
        slot_assignments={"<|img_start|>": ordinal},
    )
    tok = AutoTokenizer.from_pretrained(out)
    # the renamed token must take the id that <SPECIAL_{ordinal}> held in the base,
    # asserted via the base id (not the ordinal) so this holds even if id != ordinal
    assert tok.convert_tokens_to_ids("<|img_start|>") == _base_token_id(
        f"<SPECIAL_{ordinal}>"
    )


def test_explicit_slot_pins_ordinal_auto_would_take(tmp_path):
    # Pin a later-declared token to the LOWEST pool ordinal (which the first auto
    # token would otherwise grab). The two-pass resolver must reserve it so auto
    # skips it, instead of raising a spurious collision.
    out = str(tmp_path / "mix")
    lowest = _base_pool_ordinals()[0]
    add_modality(
        INPLACE_BASE, out, "vision", SMALL_VOCAB, allocation="in_place",
        slot_assignments={"<|img_end|>": lowest},
    )
    tok = AutoTokenizer.from_pretrained(out)
    assert tok.convert_tokens_to_ids("<|img_end|>") == _base_token_id(f"<SPECIAL_{lowest}>")
    # img_start (auto, declared first) must NOT have stolen `lowest`
    assert tok.convert_tokens_to_ids("<|img_start|>") != tok.convert_tokens_to_ids("<|img_end|>")


def test_explicit_slot_out_of_pool_raises(tmp_path):
    with pytest.raises(ValueError, match="reserve-pool"):
        add_modality(
            INPLACE_BASE, str(tmp_path / "bad"), "vision", SMALL_VOCAB,
            allocation="in_place", slot_assignments={"<|img_start|>": 5},  # chat token
        )


def test_explicit_slot_collision_raises(tmp_path):
    ord0 = _base_pool_ordinals()[0]
    with pytest.raises(ValueError, match="more than one token"):
        add_modality(
            INPLACE_BASE, str(tmp_path / "bad"), "vision", SMALL_VOCAB,
            allocation="in_place",
            slot_assignments={"<|img_start|>": ord0, "<|img_end|>": ord0},
        )


def test_slot_assignment_on_reused_token_raises(tmp_path):
    # <|image|> is a declared reuse token (no pool slot) -> overriding it must error.
    with pytest.raises(ValueError, match="reused token"):
        add_modality(
            INPLACE_BASE, str(tmp_path / "bad"), "vision", SMALL_VOCAB,
            allocation="in_place",
            slot_assignments={"<|image|>": _base_pool_ordinals()[0]},
        )


def test_slot_assignment_by_full_name(tmp_path):
    # slot_assignments values may be a full reserve-token name, not only the ordinal.
    ordinal = _base_pool_ordinals()[8]
    out = str(tmp_path / "byname")
    add_modality(
        INPLACE_BASE, out, "vision", SMALL_VOCAB, allocation="in_place",
        slot_assignments={"<|img_start|>": f"<SPECIAL_{ordinal}>"},
    )
    tok = AutoTokenizer.from_pretrained(out)
    assert tok.convert_tokens_to_ids("<|img_start|>") == _base_token_id(f"<SPECIAL_{ordinal}>")


def test_slot_assignment_full_name_not_in_pool_raises(tmp_path):
    with pytest.raises(ValueError, match="reserve-pool token"):
        add_modality(
            INPLACE_BASE, str(tmp_path / "bad"), "vision", SMALL_VOCAB,
            allocation="in_place",
            slot_assignments={"<|img_start|>": "<SPECIAL_9999>"},
        )


def test_slot_assignment_unknown_key_raises(tmp_path):
    # A key that names no structure token (typo) must error, not silently no-op.
    with pytest.raises(ValueError, match="not a structure token"):
        add_modality(
            INPLACE_BASE, str(tmp_path / "bad"), "vision", SMALL_VOCAB,
            allocation="in_place",
            slot_assignments={"<|img_stat|>": _base_pool_ordinals()[0]},
        )


def test_reserve_pool_pattern_override_restricts_pool(tmp_path):
    # A restrictive pattern (only <SPECIAL_3x>) proves pool_pattern is actually used:
    # img_start auto-fills 30, not the default lowest free (27).
    out = str(tmp_path / "restrict")
    add_modality(
        INPLACE_BASE, out, "vision", SMALL_VOCAB, allocation="in_place",
        reserve_pool_pattern=r"^<SPECIAL_(3\d)>$",
    )
    tok = AutoTokenizer.from_pretrained(out)
    assert tok.convert_tokens_to_ids("<|img_start|>") == _base_token_id("<SPECIAL_30>")


def test_reserve_pool_pattern_no_match_raises(tmp_path):
    # A pattern matching no token -> empty pool -> exhausted when auto-allocating.
    with pytest.raises(ValueError, match="exhausted"):
        add_modality(
            INPLACE_BASE, str(tmp_path / "bad"), "vision", SMALL_VOCAB,
            allocation="in_place", reserve_pool_pattern=r"^<NOPE_(\d+)>$",
        )


def test_reserve_pool_pattern_wrong_group_count_raises(tmp_path):
    with pytest.raises(ValueError, match="one capture group"):
        add_modality(
            INPLACE_BASE, str(tmp_path / "bad"), "vision", SMALL_VOCAB,
            allocation="in_place", reserve_pool_pattern=r"^<SPECIAL_(\d)(\d+)>$",
        )


def test_slot_assignment_bad_type_raises(tmp_path):
    # A bool/float slot value is neither a pool ordinal (int) nor a name (str).
    with pytest.raises(ValueError, match="pool ordinal .int. or a"):
        add_modality(
            INPLACE_BASE, str(tmp_path / "bad"), "vision", SMALL_VOCAB,
            allocation="in_place", slot_assignments={"<|img_start|>": 3.5},
        )


# ── allow_existing ───────────────────────────────────────────────────────────


def test_allow_existing_false_raises_on_preexisting(inplace_vision, tmp_path):
    work = str(tmp_path / "reresolve")
    shutil.copytree(inplace_vision, work)
    # Drop the mapping so detect_existing does not short-circuit, forcing re-resolve.
    # The already-present content tokens (and re-detected structure targets) are
    # unexpected pre-existing tokens, so strict mode must raise. Match the specific
    # allow_existing message so the test can't pass via an unrelated "exists" error.
    os.remove(os.path.join(work, "vision_token_mapping.json"))
    with pytest.raises(ValueError, match="allow_existing=False"):
        add_modality(work, str(tmp_path / "o"), "vision", SMALL_VOCAB,
                     allocation="in_place", allow_existing=False)


def test_realias_path_dedup_guard(inplace_vision, tmp_path):
    # Force the REAL apply/alias path (not the idempotency short-circuit, which never
    # calls add_token_alias) by dropping the mapping file. add_token_alias then runs
    # on an already-aliased tokenizer and its dedup guard must prevent a 2nd Replace.
    work = str(tmp_path / "realias")
    shutil.copytree(inplace_vision, work)
    os.remove(os.path.join(work, "vision_token_mapping.json"))
    add_modality(work, work, "vision", SMALL_VOCAB, allocation="in_place")  # allow_existing default True
    assert _count_replace(work, "<image>") == 1
    tok = AutoTokenizer.from_pretrained(work)
    img = tok.convert_tokens_to_ids("<|image|>")
    assert tok.encode("<image>", add_special_tokens=False) == [img]


# ── dry run ──────────────────────────────────────────────────────────────────


def test_dry_run_writes_nothing(tmp_path):
    out = str(tmp_path / "dry")
    tok, stats = add_modality(
        INPLACE_BASE, out, "vision", SMALL_VOCAB, allocation="in_place", dry_run=True
    )
    assert tok is None
    assert stats["dry_run"] is True
    assert not os.path.exists(out)
    assert "report" in stats and "dry_run=True" in stats["report"]
    assert stats["content_tokens_added"] == SMALL_VOCAB


def test_dry_run_plan_matches_real(tmp_path):
    _, dry_stats = add_modality(
        INPLACE_BASE, str(tmp_path / "d"), "vision", SMALL_VOCAB,
        allocation="in_place", dry_run=True,
    )
    real = str(tmp_path / "r")
    tok, _ = add_modality(INPLACE_BASE, real, "vision", SMALL_VOCAB, allocation="in_place")
    assert dry_stats["final_vocab_size"] == len(tok)


def test_dry_run_plan_lines_up_with_applied_changes(tmp_path):
    """The dry-run plan must match, id-for-id, what a real run actually writes."""
    _, dry = add_modality(
        INPLACE_BASE, str(tmp_path / "d"), "vision", SMALL_VOCAB,
        allocation="in_place", dry_run=True,
    )
    real_dir = str(tmp_path / "r")
    tok, real = add_modality(
        INPLACE_BASE, real_dir, "vision", SMALL_VOCAB, allocation="in_place"
    )

    assert dry["planned"] == real["planned"]  # deterministic resolution
    planned = dry["planned"]

    assert planned["structure_ids"], "expected resolved structure ids"
    for name, pid in planned["structure_ids"].items():
        assert tok.convert_tokens_to_ids(name) == pid, name

    data = _mapping(real_dir, "vision_token_mapping.json")
    assert data["vision_token_offset"] == planned["content_offset"]
    ids = [data["vision_token_ids"][str(i)] for i in range(SMALL_VOCAB)]
    assert ids == list(
        range(planned["content_offset"], planned["content_offset"] + SMALL_VOCAB)
    )

    omc = _mapping(real_dir, "tokenizer_config.json")["omnimodal_config"]
    assert omc["omni_special_token_offset"] == planned["special_region_offset"]
    assert omc["omni_special_token_count"] == planned["special_region_count"]
    assert planned["projected_vocab_size"] == len(tok)
