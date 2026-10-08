"""Offline text-only artifacts, reproducible builds and conversation controls."""

import json
import sys
from pathlib import Path

import pytest
from omnitok import cli
from omnitok.recipes.apertus_2 import (
    CONTROLS,
    EXTRA_SPECIAL_TOKENS,
    SOURCE_SHA256,
    VOCAB_SIZE,
    build_base,
    build_instruct,
)
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tokenizers" / "Apertus_2"
INSTRUCT = ROOT / "tokenizers" / "Apertus_2_instruct"


def test_base_is_text_only():
    backend = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    assert backend.get_vocab_size() == VOCAB_SIZE
    assert backend.token_to_id("<SPECIAL_27>") == 27
    assert backend.token_to_id("<|img_start|>") is None
    assert backend.token_to_id("<|visual token 0|>") is None
    assert backend.token_to_id("<|audio token 0|>") is None
    assert "omnimodal_config" not in json.loads(
        (BASE / "tokenizer_config.json").read_text()
    )


def test_build_base_copies_the_pinned_files(tmp_path):
    output = build_base(BASE, tmp_path / "built")
    assert {p.name for p in output.iterdir()} == set(SOURCE_SHA256)
    for name in SOURCE_SHA256:
        assert (output / name).read_bytes() == (BASE / name).read_bytes()


def test_instruct_compacts_specials_without_changing_ordinary_vocabulary():
    before = json.loads((BASE / "tokenizer.json").read_bytes())
    after = json.loads((INSTRUCT / "tokenizer.json").read_bytes())
    assigned = [
        "<unk>", "<s>", "</s>", "<|pad|>",
        "<iban-pii>", "<email-pii>", "<ip-pii>",
        "<|in|>", "<|/in|>", "<|hdr|>", "<|out|>", "<|/out|>", "<|wait|>",
    ]
    specials = assigned + [f"<SPECIAL_{i}>" for i in range(13, 124)]
    vocab = after["model"]["vocab"]
    assert vocab["<|out|>"] == before["model"]["vocab"]["<|assistant_start|>"] == 10
    assert vocab["<|/out|>"] == before["model"]["vocab"]["<|assistant_end|>"] == 11
    assert len(vocab) == VOCAB_SIZE
    assert set(vocab.values()) == set(range(VOCAB_SIZE))
    assert {s: i for s, i in vocab.items() if i < 124} == {
        s: i for i, s in enumerate(specials)
    }
    assert after["added_tokens"] == [
        {"id": i, "content": s, "special": True, "normalized": False,
         "single_word": False, "lstrip": False, "rstrip": False}
        for i, s in enumerate(specials)
    ]
    assert {s: i for s, i in vocab.items() if i >= 124} == {
        s: i for s, i in before["model"]["vocab"].items() if i >= 124
    }
    for key in ("normalizer", "post_processor", "padding", "truncation"):
        assert after[key] is None
    # Everything outside the special block and disabled text transforms is inherited.
    for state in (before, after):
        state["model"].pop("vocab")
        for key in ("added_tokens", "normalizer", "post_processor", "padding", "truncation"):
            state.pop(key)
    assert before == after


def test_removed_legacy_spellings_are_ordinary_text():
    tokenizer = Tokenizer.from_file(str(INSTRUCT / "tokenizer.json"))
    legacy = ["<pad>", "<|image|>", "<|audio|>", "<think>", "</think>",
              "<reflection>", "</reflection>"]
    for role in ("system", "developer", "user", "assistant", "tool_output"):
        legacy.extend([f"<|{role}_start|>", f"<|{role}_end|>"])
    for role in ("inner", "tools"):
        legacy.extend([f"<|{role}_prefix|>", f"<|{role}_suffix|>"])
    for text in legacy:
        assert tokenizer.token_to_id(text) is None
        ids = tokenizer.encode(text).ids
        assert not set(range(124)).intersection(ids)
        assert tokenizer.decode(ids) == text


def test_all_specials_encode_and_decode_consistently():
    tokenizer = Tokenizer.from_file(str(INSTRUCT / "tokenizer.json"))
    for token_id, token in tokenizer.get_added_tokens_decoder().items():
        assert tokenizer.encode(token.content).ids == [token_id]
        assert tokenizer.decode([token_id], skip_special_tokens=False) == token.content
        assert tokenizer.decode([token_id], skip_special_tokens=True) == ""
    tokenizer.encode_special_tokens = True
    for token in tokenizer.get_added_tokens_decoder().values():
        text = f"A {token.content} B"
        ids = tokenizer.encode(text).ids
        assert not set(range(124)).intersection(ids)
        assert tokenizer.decode(ids) == text


def test_instruct_controls_and_exact_ordinary_text():
    tokenizer = Tokenizer.from_file(str(INSTRUCT / "tokenizer.json"))
    for glyph, token_id in CONTROLS.items():
        assert tokenizer.encode(glyph).ids == [token_id]
    tokenizer.encode_special_tokens = True
    special_ids = {
        i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special
    }
    for text in [
        "".join(CONTROLS),
        "e\u0301\r\n\x00🙂",
        "<image> <audio>",
        "<|image|>",
    ]:
        ids = tokenizer.encode(text).ids
        assert not special_ids.intersection(ids)
        assert tokenizer.decode(ids, skip_special_tokens=False) == text
    ids = [
        10,
        *tokenizer.encode("reply").ids,
        9,
        *tokenizer.encode("Hello").ids,
        11,
        12,
    ]
    assert ids == [10, 120213, 9, 36971, 11, 12]


def test_framework_loading_roles():
    base = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    instruct = AutoTokenizer.from_pretrained(INSTRUCT, local_files_only=True)
    assert len(base) == len(instruct) == VOCAB_SIZE
    assert (base.bos_token_id, base.eos_token_id, base.pad_token_id) == (1, 2, 3)
    assert (instruct.eos_token_id, instruct.pad_token_id) == (12, 3)
    assert instruct.encode("Hello", add_special_tokens=True) == [36971]
    assert instruct.chat_template is None
    for role, glyph in EXTRA_SPECIAL_TOKENS.items():
        assert getattr(instruct, role) == glyph
        assert getattr(instruct, role + "_id") == CONTROLS[glyph]
    assert {i for i, t in instruct.added_tokens_decoder.items() if t.special} == set(range(124))
    generation = json.loads((INSTRUCT / "generation_config.json").read_text())
    assert generation == {"eos_token_id": [12, 2], "pad_token_id": 3}


@pytest.mark.parametrize("filename", list(SOURCE_SHA256))
def test_source_drift_rejected_without_writing(filename, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in SOURCE_SHA256:
        raw = (BASE / name).read_bytes()
        (source / name).write_bytes(raw + (b" " if name == filename else b""))
    for build in (build_base, build_instruct):
        with pytest.raises(ValueError, match="checksum"):
            build(source, tmp_path / "output")
        assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("build", [build_base, build_instruct])
def test_rejects_overwriting_and_nested_outputs(build, tmp_path):
    for output in (BASE, BASE / "derived", BASE.parent):
        with pytest.raises(ValueError, match="separate"):
            build(BASE, output)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "keep"
    marker.write_text("existing")
    with pytest.raises(ValueError, match="empty"):
        build(BASE, occupied)
    assert marker.read_text() == "existing"


def test_cli_builds_instruct(tmp_path, monkeypatch):
    output = tmp_path / "instruct"
    monkeypatch.setattr(sys, "argv", [
        "omnitok", "build-apertus-2", "instruct",
        "--input-tokenizer", str(BASE), "--output-path", str(output),
    ])
    cli.main()
    assert (output / "tokenizer.json").read_bytes() == (
        INSTRUCT / "tokenizer.json"
    ).read_bytes()
