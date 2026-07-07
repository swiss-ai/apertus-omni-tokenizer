"""BOS ownership: the instruct builder must leave exactly one BOS owner.

When the chat template emits ``{{ bos_token }}``, the tokenizer must not also
auto-prepend the BOS, or every ``add_special_tokens=True`` path produces
``<s><s>...`` -> degeneration (apertus-program #420). These tests pin the
low-level post-processor stripper; they need no network (they operate on a
synthetic ``tokenizer.json``), unlike the full builder tests.
"""

import json

from omnitok.io import strip_bos_from_post_processor


def _write_tokenizer_json(dir_path, *, with_bos):
    """A minimal tokenizer.json whose post-processor optionally prepends <s>."""
    bos_single = [{"SpecialToken": {"id": "<s>", "type_id": 0}}] if with_bos else []
    bos_pair_a = [{"SpecialToken": {"id": "<s>", "type_id": 0}}] if with_bos else []
    bos_pair_b = [{"SpecialToken": {"id": "<s>", "type_id": 1}}] if with_bos else []
    tok = {
        "version": "1.0",
        "post_processor": {
            "type": "TemplateProcessing",
            "single": bos_single + [{"Sequence": {"id": "A", "type_id": 0}}],
            "pair": (
                bos_pair_a
                + [{"Sequence": {"id": "A", "type_id": 0}}]
                + bos_pair_b
                + [{"Sequence": {"id": "B", "type_id": 1}}]
            ),
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
            },
        },
    }
    p = dir_path / "tokenizer.json"
    p.write_text(json.dumps(tok, indent=2), encoding="utf-8")
    return p


def _specials(seq):
    return [x["SpecialToken"]["id"] for x in seq if "SpecialToken" in x]


def test_strips_bos_from_single_and_pair(tmp_path):
    _write_tokenizer_json(tmp_path, with_bos=True)
    changed = strip_bos_from_post_processor(str(tmp_path), "<s>")
    assert changed is True
    pp = json.loads((tmp_path / "tokenizer.json").read_text())["post_processor"]
    # No BOS SpecialToken remains; the Sequence pieces are untouched.
    assert _specials(pp["single"]) == []
    assert _specials(pp["pair"]) == []
    assert [x["Sequence"]["id"] for x in pp["single"]] == ["A"]
    assert [x["Sequence"]["id"] for x in pp["pair"]] == ["A", "B"]


def test_idempotent_and_noop_when_no_bos(tmp_path):
    _write_tokenizer_json(tmp_path, with_bos=False)
    # Nothing to strip on a tokenizer whose post-processor never adds BOS.
    assert strip_bos_from_post_processor(str(tmp_path), "<s>") is False
    # And a second pass after a successful strip is a no-op.
    _write_tokenizer_json(tmp_path, with_bos=True)
    assert strip_bos_from_post_processor(str(tmp_path), "<s>") is True
    assert strip_bos_from_post_processor(str(tmp_path), "<s>") is False


def test_only_touches_the_bos_id(tmp_path):
    """A non-BOS SpecialToken (e.g. EOS) must survive."""
    tok = {
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
            ],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {},
        }
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok, indent=2), encoding="utf-8")
    assert strip_bos_from_post_processor(str(tmp_path), "<s>") is True
    pp = json.loads((tmp_path / "tokenizer.json").read_text())["post_processor"]
    assert _specials(pp["single"]) == ["</s>"]
