"""The Apertus 1.5 recipe: parents it accepts and the base it refuses.

The byte-for-byte rebuild of tokenizers/Apertus_1p5 from its registered
parent (tokenizers/Apertus_1_base) is checked in tests/test_registry.py.
"""

from pathlib import Path

import pytest

from omnitok import prepare_apertus_1p5_text_base, registry
from omnitok.recipes.apertus_1p5 import build_from_parent

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_wrong_base_is_rejected(tmp_path):
    """Rebuilding on anything but the pinned Apertus 1 layout fails loudly
    instead of producing a differently-shaped artifact (e.g. a base where the
    reasoning/tool slots were already renamed)."""
    with pytest.raises(ValueError, match="not the expected Apertus 1 base"):
        prepare_apertus_1p5_text_base(
            str(REPO_ROOT / "tokenizers" / "Apertus_1p5"),
            str(tmp_path / "never_finished"),
        )


def test_rebuild_from_the_instruct_mirror_matches(tmp_path):
    """The 1.0 instruct tokenizer (Apertus_1) builds the same bytes as the
    registered base parent; the recipe is parent-agnostic between them."""
    output = tmp_path / "Apertus_1p5"
    build_from_parent(registry.artifact_dir("Apertus_1"), output)
    expected = registry.committed_files("Apertus_1p5")
    assert {p.name for p in output.iterdir()} == set(expected)
    for name, path in expected.items():
        assert (output / name).read_bytes() == path.read_bytes(), name
