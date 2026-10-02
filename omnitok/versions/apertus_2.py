"""Pinned text-only Apertus 2 base and conversation-token builds.

The base is copied byte-for-byte from preliminary_mul_200k. The instruct
variant renames seven reserved tokens without adding vocabulary or Jinja.
Run ``python -m omnitok.cli build-apertus-2 --help`` for the CLI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/swiss-ai/apertus-tokenizer-development"
SOURCE_REVISION = "28ad57a2757f6f72edb0def57ee725b6812f2df3"
SOURCE_PATH = "preliminary_mul_200k"
SOURCE_SHA256 = {
    "tokenizer.json": "cd403d3f219e2433e3f78b32644b8e6a6134668e15138e6546360330635a96b9",
    "tokenizer_config.json": "7e6b68d5a41fd06d399143b7591df15abf7ae2846fa8f444f66f3d5edee5996f",
    "special_tokens_map.json": "816ec96e37c6d15e3cbc535dc146c898a7218f209fc154384f31fc1e6ad31ba5",
}
PROFILE_REVISION = "85874b84605f2a0452d53fe5874cc5eddac1b7f4"
VOCAB_SIZE = 200_064
CONTROLS = {
    "<|in|>": 40,
    "<|/in|>": 41,
    "<|out|>": 42,
    "<|/out|>": 43,
    "<|hdr|>": 44,
    "<|wait|>": 45,
    "<|pad|>": 46,
}


def _read_source(
    input_path: str | Path, output_path: str | Path
) -> tuple[Path, dict[str, bytes]]:
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            "source and output directories must be separate and non-nested"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be empty or absent")
    files = {name: (source / name).read_bytes() for name in SOURCE_SHA256}
    for name, raw in files.items():
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256[name]:
            raise ValueError(f"{name}: checksum differs from the pinned text base")
    return output, files


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_base(input_path: str | Path, output_path: str | Path) -> Path:
    """Copy the verified upstream base unchanged and record its provenance."""
    output, files = _read_source(input_path, output_path)
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    _write_json(
        output / "source.json",
        {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path": SOURCE_PATH,
            "sha256": SOURCE_SHA256,
        },
    )
    return output


def build_instruct(input_path: str | Path, output_path: str | Path) -> Path:
    """Derive exact-text conversation encoding from the pinned text base."""
    output, files = _read_source(input_path, output_path)
    data = json.loads(files["tokenizer.json"])
    vocab = data["model"]["vocab"]
    added = {entry["id"]: entry for entry in data["added_tokens"]}
    if len(vocab) != VOCAB_SIZE:
        raise ValueError("unexpected text vocabulary size")
    for glyph, token_id in CONTROLS.items():
        reserved = f"<SPECIAL_{token_id}>"
        if (
            vocab.get(reserved) != token_id
            or added[token_id]["content"] != reserved
            or glyph in vocab
        ):
            raise ValueError(f"reserved slot {token_id} differs from the pinned base")
        del vocab[reserved]
        vocab[glyph] = token_id
        added[token_id].update(content=glyph, special=True, normalized=False)
    # Literal glyphs and decomposed Unicode are data; controls are inserted by
    # the consuming library with ordinary-text special recognition disabled.
    data.update(normalizer=None, post_processor=None, padding=None, truncation=None)
    config = json.loads(files["tokenizer_config.json"])
    config.pop("added_tokens_decoder", None)
    config.update(
        eos_token="<|wait|>",
        pad_token="<|pad|>",
        add_bos_token=False,
        add_eos_token=False,
    )
    roles = json.loads(files["special_tokens_map.json"])
    for name, glyph in (("eos_token", "<|wait|>"), ("pad_token", "<|pad|>")):
        roles[name]["content"] = glyph
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "tokenizer.json", data)
    _write_json(output / "tokenizer_config.json", config)
    _write_json(output / "special_tokens_map.json", roles)
    _write_json(
        output / "generation_config.json", {"eos_token_id": 45, "pad_token_id": 46}
    )
    _write_json(
        output / "apertus_encoding.json",
        {
            "schema_version": 1,
            "profile": "apertus_2",
            "profile_revision": PROFILE_REVISION,
            "tokenizer_sha256": hashlib.sha256(
                (output / "tokenizer.json").read_bytes()
            ).hexdigest(),
            "controls": CONTROLS,
            "special_token_ids": sorted(
                token["id"] for token in data["added_tokens"] if token["special"]
            ),
            "tokenizers_version": ">=0.23.2,<0.24",
            "text_mode": "verbatim",
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256["tokenizer.json"],
        },
    )
    return output

