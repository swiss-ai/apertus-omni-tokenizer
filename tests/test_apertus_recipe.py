"""The Apertus 1.5 build recipe refuses the wrong base.

The byte-for-byte rebuild of tokenizers/Apertus_1p5 from its registered
parent (tokenizers/Apertus_1_base) is checked in tests/test_registry.py.
"""

from pathlib import Path

import pytest

from omnitok import prepare_apertus_1p5_text_base

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
