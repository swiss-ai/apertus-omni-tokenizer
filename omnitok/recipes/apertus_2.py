"""Pinned text-only Apertus 2 base and conversation-token builds.

The base is copied byte-for-byte from preliminary_mul_200k in
https://github.com/swiss-ai/apertus-tokenizer-development; its revision and
file hashes are registered as Apertus_2 in omnitok/registry.py. The instruct
variant replaces the legacy special-token layout with 13 consecutive assigned
tokens and 111 reserved slots. Padding is renamed in place at id 3; assistant
delimiters become out delimiters at their original ids 10/11. Ordinary
vocabulary ids and BPE merges stay unchanged. BOS/EOS placement belongs to the conversation encoder; the tokenizer
never inserts either automatically. There is no Jinja template; generation
stops on wait or </s>.
Run ``python -m omnitok.cli build-apertus-2 --help`` for the CLI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .. import registry
from ..io import assert_droppable_post_processor, save_tokenizer_files

_SOURCE = registry.get("Apertus_2")
SOURCE_REVISION = _SOURCE.origin.revision
SOURCE_SHA256 = _SOURCE.sha256
VOCAB_SIZE = 200_064
SPECIAL_TOKEN_COUNT = 124
# Assigned ids 0-12, followed by <SPECIAL_13> through <SPECIAL_123>.
# Preserve the base's assistant_start/end ids when renaming them to out / /out.
SPECIAL_TOKENS = (
    "<unk>", "<s>", "</s>", "<|pad|>",
    "<iban-pii>", "<email-pii>", "<ip-pii>",
    "<|in|>", "<|/in|>", "<|hdr|>", "<|out|>", "<|/out|>", "<|wait|>",
)
CONTROLS = {
    glyph: SPECIAL_TOKENS.index(glyph)
    for glyph in ("<|in|>", "<|/in|>", "<|out|>", "<|/out|>",
                  "<|hdr|>", "<|wait|>", "<|pad|>")
}

EXTRA_SPECIAL_TOKENS = {
    "input_start_token": "<|in|>",
    "input_end_token": "<|/in|>",
    "output_start_token": "<|out|>",
    "output_end_token": "<|/out|>",
    "header_end_token": "<|hdr|>",
    "wait_token": "<|wait|>",
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
    """Copy the verified upstream base unchanged."""
    output, files = _read_source(input_path, output_path)
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    return output


def build_instruct(
    input_path: str | Path, output_path: str | Path, *, save_mode: str = "compatible"
) -> Path:
    """Derive exact-text conversation encoding from the pinned text base.

    Only the canonical compatible save mode is supported by this recipe;
    it reproduces identical artifact bytes across the build matrix.

    Dropping the base's post-processor goes through
    assert_droppable_post_processor first, as everywhere else in this
    package: any code that drops a post-processor must check it is
    droppable, so behaviour beyond declared BOS/EOS insertion is never
    lost silently.
    """
    if save_mode != "compatible":
        raise ValueError("Apertus_2_instruct is saved only in compatible mode")
    output, files = _read_source(input_path, output_path)
    data = json.loads(files["tokenizer.json"])
    vocab = data["model"]["vocab"]
    added = {entry["id"]: entry for entry in data["added_tokens"]}
    if len(vocab) != VOCAB_SIZE:
        raise ValueError("unexpected text vocabulary size")
    # Remove the whole old block before assigning names: compaction moves
    # retained spellings between ids, which would otherwise collide.
    if set(added) != set(range(SPECIAL_TOKEN_COUNT)):
        raise ValueError("unexpected special-token layout")
    for token_id, entry in added.items():
        if vocab.get(entry["content"]) != token_id:
            raise ValueError(f"special slot {token_id} differs from the pinned base")
        del vocab[entry["content"]]
    for token_id in range(SPECIAL_TOKEN_COUNT):
        glyph = (SPECIAL_TOKENS[token_id] if token_id < len(SPECIAL_TOKENS)
                 else f"<SPECIAL_{token_id}>")
        vocab[glyph] = token_id
        added[token_id].update(content=glyph, special=True, normalized=False)
    # The conversation encoder owns BOS/EOS placement. Role metadata below
    # declares the tokens, but neither postprocessing nor add_* may insert them.
    # Ordinary text is encoded by the consumer with special recognition disabled.
    config = json.loads(files["tokenizer_config.json"])
    assert_droppable_post_processor(
        data["post_processor"], {config["bos_token"], config["eos_token"]}
    )
    data.update(normalizer=None, post_processor=None, padding=None, truncation=None)
    config.update(
        eos_token="<|wait|>",
        pad_token="<|pad|>",
        add_bos_token=False,
        add_eos_token=False,
        extra_special_tokens=EXTRA_SPECIAL_TOKENS,
        **EXTRA_SPECIAL_TOKENS,
    )
    # The writer drops the added-token mirror and derives the special-tokens
    # map from the roles; through the backend, the special ids return to their place
    # in the vocab, so a tokenizers load-and-save reproduces the bytes.
    save_tokenizer_files(output, data, config)
    _write_json(
        output / "generation_config.json",
        {"eos_token_id": [CONTROLS["<|wait|>"], vocab["</s>"]],
         "pad_token_id": CONTROLS["<|pad|>"]},
    )
    return output
