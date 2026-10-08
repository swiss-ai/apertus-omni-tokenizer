"""Registry invariants: pinned imports, byte-exact rebuilds, no strays.

Every check is driven by omnitok/registry.py, so a newly registered
tokenizer is covered without touching this file.
"""

import hashlib

import pytest
from tokenizers import Tokenizer

from omnitok import registry
from omnitok.registry import Derived, Imported

ROOT = registry.REPO_ROOT
IMPORTED = [e for e in registry.TOKENIZERS if isinstance(e, Imported)]
DERIVED = [e for e in registry.TOKENIZERS if isinstance(e, Derived)]
MANIFESTS = sorted((ROOT / "validation").glob("*.md5"))


def test_every_artifact_directory_is_registered():
    names = {entry.name for entry in registry.TOKENIZERS}
    assert len(names) == len(registry.TOKENIZERS), "duplicate registry names"
    tokenizer_dirs = registry.visible_entries(ROOT / "tokenizers")
    template_dirs = registry.visible_entries(ROOT / "chat_templates")
    assert all(p.is_dir() for p in tokenizer_dirs + template_dirs), "stray files"
    assert {p.name for p in tokenizer_dirs} == names
    assert {p.name for p in template_dirs} <= names
    assert {p.stem for p in MANIFESTS} <= names


def test_parents_are_listed_before_their_children():
    seen = set()
    for entry in registry.TOKENIZERS:
        if isinstance(entry, Derived):
            assert entry.parent in seen, f"{entry.name}: parent {entry.parent!r}"
        seen.add(entry.name)


@pytest.mark.parametrize("entry", IMPORTED, ids=lambda e: e.name)
def test_imported_files_match_their_pins(entry):
    files = registry.committed_files(entry.name)
    assert set(files) == set(entry.sha256)
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry.sha256[name], name


@pytest.mark.parametrize("entry", DERIVED, ids=lambda e: e.name)
def test_derived_tokenizer_json_survives_a_resave(entry, tmp_path):
    """Checkpoints often re-save their tokenizer; a tokenizers load-and-save
    must reproduce the committed bytes, or validation and bindings break."""
    path = registry.artifact_dir(entry.name) / "tokenizer.json"
    Tokenizer.from_file(str(path)).save(str(tmp_path / "resaved.json"), pretty=True)
    assert (tmp_path / "resaved.json").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("entry", DERIVED, ids=lambda e: e.name)
def test_derived_rebuilds_byte_for_byte(entry, tmp_path):
    output = tmp_path / entry.name
    registry.load_recipe(entry)(registry.artifact_dir(entry.parent), output)
    built = {p.name: p for p in output.iterdir()}
    expected = registry.committed_files(entry.name)
    assert set(built) == set(expected)
    for name, path in expected.items():
        assert built[name].read_bytes() == path.read_bytes(), name


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.stem)
def test_deployment_manifest_covers_the_committed_files(manifest):
    files = registry.committed_files(manifest.stem)
    pinned = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split()
        pinned[name] = digest
    # generation_config.json differs per checkpoint; validate_model.sh checks
    # its fields instead of its checksum.
    assert set(pinned) == set(files) - {"generation_config.json"}
    for name, digest in pinned.items():
        assert hashlib.md5(files[name].read_bytes()).hexdigest() == digest, name
