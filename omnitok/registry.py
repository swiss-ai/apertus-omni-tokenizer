"""Every tokenizer this repository ships: where it comes from, how it is made.

Each entry is one of two kinds:

- ``Imported``: a byte-identical copy of an upstream file set, pinned to an
  origin revision and to the SHA-256 of every committed file.
- ``Derived``: built offline by a recipe from another registered tokenizer
  (its ``parent``). A recipe is a ``"module:function"`` reference to a
  callable ``(parent_dir, output_dir)``; the committed files are exactly what
  it produces.

Lineage::

    swiss-ai/Apertus-8B-2509 @ 3162c996            -> Apertus_1_base -> Apertus_1p5
    swiss-ai/Apertus-8B-Instruct-2509 @ b946d404   -> Apertus_1
    apertus-tokenizer-development @ 28ad57a2
        /preliminary_mul_200k                      -> Apertus_2 -> Apertus_2_instruct

Committed files live in ``tokenizers/<name>/``, except a chat template,
which lives in ``chat_templates/<name>/chat_template.jinja`` (the layout
validate_model.sh downloads from). Entries are listed parents first.

Adding a tokenizer means adding one entry here, plus a recipe for a derived
one. tests/test_registry.py then verifies the pins, rebuilds derived
tokenizers byte-for-byte and rejects unregistered artifact directories.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAT_TEMPLATE = "chat_template.jinja"


@dataclass(frozen=True)
class HubOrigin:
    """A Hugging Face Hub repository at a pinned commit."""

    repo: str
    revision: str


@dataclass(frozen=True)
class GitOrigin:
    """A directory inside a git repository at a pinned commit."""

    repository: str
    revision: str
    path: str


@dataclass(frozen=True)
class Imported:
    """Upstream files committed unchanged."""

    name: str
    origin: Union[HubOrigin, GitOrigin]
    sha256: dict[str, str] = field(hash=False)
    note: str = ""


@dataclass(frozen=True)
class Derived:
    """Files built by ``recipe`` from the registered tokenizer ``parent``."""

    name: str
    parent: str
    recipe: str
    note: str = ""


Entry = Union[Imported, Derived]

TOKENIZERS: tuple[Entry, ...] = (
    Imported(
        "Apertus_1_base",
        origin=HubOrigin(
            "swiss-ai/Apertus-8B-2509", "3162c99675aa588097cecd4a24b9aa1f712af477"
        ),
        sha256={
            "tokenizer.json": "bb201fb226cde11f66c3cf51c5344fb37b1611f00c21e75c324546d854eff2e1",
            "tokenizer_config.json": "ea64a17b41e1deaa7469212f413676129f33977ca3a48767f0ca68dc346df502",
            "special_tokens_map.json": "816ec96e37c6d15e3cbc535dc146c898a7218f209fc154384f31fc1e6ad31ba5",
        },
        note="Apertus 1 base model; the parent of the 1.5 build.",
    ),
    Imported(
        "Apertus_1",
        origin=HubOrigin(
            "swiss-ai/Apertus-8B-Instruct-2509",
            "b946d40447b2b597999b9c86d44bee0b452c919f",
        ),
        sha256={
            "tokenizer.json": "bb201fb226cde11f66c3cf51c5344fb37b1611f00c21e75c324546d854eff2e1",
            "tokenizer_config.json": "b89e07508603d6e1f2c8642f366c2469063c3eddae0585f1e67934ee98498ab1",
            "special_tokens_map.json": "3a3f3af9c9cbc9cf89c719d3de130913dcd626ac5061cdf3cd9ff21728f9d58f",
            CHAT_TEMPLATE: "56f72c27fa2e565312a830b7315a455c79fb452fdbe7a5b7595d8a094f282b07",
        },
        note="Apertus 1.0 instruct; the validate_model.sh baseline for 1.0.",
    ),
    Derived(
        "Apertus_1p5",
        parent="Apertus_1_base",
        recipe="omnitok.recipes.apertus_1p5:build_from_parent",
        note="Text renames, vision + audio modalities and the hand-maintained "
        "chat template under chat_templates/Apertus_1p5/.",
    ),
    Imported(
        "Apertus_2",
        origin=GitOrigin(
            "https://github.com/swiss-ai/apertus-tokenizer-development",
            "28ad57a2757f6f72edb0def57ee725b6812f2df3",
            "preliminary_mul_200k",
        ),
        sha256={
            "tokenizer.json": "cd403d3f219e2433e3f78b32644b8e6a6134668e15138e6546360330635a96b9",
            "tokenizer_config.json": "7e6b68d5a41fd06d399143b7591df15abf7ae2846fa8f444f66f3d5edee5996f",
            "special_tokens_map.json": "816ec96e37c6d15e3cbc535dc146c898a7218f209fc154384f31fc1e6ad31ba5",
        },
        note="200,064-token text base; no modality vocabulary.",
    ),
    Derived(
        "Apertus_2_instruct",
        parent="Apertus_2",
        recipe="omnitok.recipes.apertus_2:build_instruct",
        note="Compact special-token layout: assigned ids 0-12, reserves 13-123.",
    ),
)

_BY_NAME = {entry.name: entry for entry in TOKENIZERS}


def get(name: str) -> Entry:
    """Return the registry entry for ``name``."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not registered; known: {sorted(_BY_NAME)}"
        ) from None


def artifact_dir(name: str) -> Path:
    """The directory holding ``name``'s committed tokenizer files."""
    return REPO_ROOT / "tokenizers" / get(name).name


def committed_files(name: str) -> dict[str, Path]:
    """Map each committed file name of ``name`` to its path in the repo.

    Raises ValueError for files the pins would miss: a subdirectory, anything
    but chat_template.jinja under chat_templates/<name>/, or a chat template
    in both places. Hidden files such as .DS_Store are ignored.
    """
    files = {}
    for path in visible_entries(artifact_dir(name)):
        if not path.is_file():
            raise ValueError(f"unexpected subdirectory {path}")
        files[path.name] = path
    for path in visible_entries(REPO_ROOT / "chat_templates" / name):
        if path.name != CHAT_TEMPLATE or CHAT_TEMPLATE in files:
            raise ValueError(f"unexpected chat template file {path}")
        files[CHAT_TEMPLATE] = path
    return files


def visible_entries(directory: Path) -> list[Path]:
    """The non-hidden entries of ``directory``, sorted; none if it is absent."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if not p.name.startswith("."))


def load_recipe(entry: Derived) -> Callable[..., object]:
    """Import the callable a derived entry's ``recipe`` refers to."""
    module, _, function = entry.recipe.partition(":")
    return getattr(importlib.import_module(module), function)
