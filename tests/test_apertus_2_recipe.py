"""Offline text-only artifacts, reproducible builds and conversation controls."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from omnitok.versions.apertus_2 import (
    CONTROLS,
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


def test_base_is_upstream_byte_exact_and_text_only():
    for name, digest in SOURCE_SHA256.items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    backend = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    assert backend.get_vocab_size() == VOCAB_SIZE
    assert backend.token_to_id("<SPECIAL_27>") == 27
    assert backend.token_to_id("<|img_start|>") is None
    assert backend.token_to_id("<|visual token 0|>") is None
    assert backend.token_to_id("<|audio token 0|>") is None
    assert "omnimodal_config" not in json.loads(
        (BASE / "tokenizer_config.json").read_text()
    )


@pytest.mark.parametrize(
    "build,expected", [(build_base, BASE), (build_instruct, INSTRUCT)]
)
def test_rebuild_matches_every_committed_file(build, expected, tmp_path):
    output = build(BASE, tmp_path / "built")
    assert {p.name for p in output.iterdir()} == {p.name for p in expected.iterdir()}
    for artifact in expected.iterdir():
        assert (output / artifact.name).read_bytes() == artifact.read_bytes()


def test_instruct_only_renames_seven_slots_and_disables_text_rewriting():
    before = json.loads((BASE / "tokenizer.json").read_bytes())
    after = json.loads((INSTRUCT / "tokenizer.json").read_bytes())
    expected = json.loads(json.dumps(before))
    by_id = {entry["id"]: entry for entry in expected["added_tokens"]}
    for glyph, token_id in CONTROLS.items():
        assert expected["model"]["vocab"].pop(f"<SPECIAL_{token_id}>") == token_id
        expected["model"]["vocab"][glyph] = token_id
        by_id[token_id].update(content=glyph, special=True, normalized=False)
    expected.update(normalizer=None, post_processor=None, padding=None, truncation=None)
    assert after == expected
    assert len(after["model"]["vocab"]) == VOCAB_SIZE
    assert after["model"]["merges"] == before["model"]["merges"]


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
        42,
        *tokenizer.encode("reply").ids,
        44,
        *tokenizer.encode("Hello").ids,
        43,
        45,
    ]
    assert ids == [42, 120213, 44, 36971, 43, 45]


def test_framework_loading_roles_and_manifest():
    base = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    instruct = AutoTokenizer.from_pretrained(INSTRUCT, local_files_only=True)
    assert len(base) == len(instruct) == VOCAB_SIZE
    assert (base.bos_token_id, base.eos_token_id, base.pad_token_id) == (1, 2, 3)
    assert (instruct.eos_token_id, instruct.pad_token_id) == (45, 46)
    assert instruct.encode("Hello", add_special_tokens=True) == [36971]
    assert instruct.chat_template is None
    manifest = json.loads((INSTRUCT / "apertus_encoding.json").read_text())
    assert (
        manifest["tokenizer_sha256"]
        == hashlib.sha256((INSTRUCT / "tokenizer.json").read_bytes()).hexdigest()
    )
    assert manifest["source_sha256"] == SOURCE_SHA256["tokenizer.json"]
    assert manifest["controls"] == CONTROLS
    assert manifest["special_token_ids"] == list(range(124))


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


def test_cli_builds_instruct(tmp_path):
    output = tmp_path / "instruct"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "omnitok.cli",
            "build-apertus-2",
            "instruct",
            "--input-tokenizer",
            str(BASE),
            "--output-path",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    assert (output / "tokenizer.json").read_bytes() == (
        INSTRUCT / "tokenizer.json"
    ).read_bytes()
