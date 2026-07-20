"""The Apertus 1.5 build recipe reproduces the canonical artifact.

The whole point of this repo: running `build_apertus_1p5` on the Apertus 1
base yields the canonical Apertus 1.5 tokenizer (apertus-ai/Apertus-v1.5-8B-RC)
byte-for-byte, as pinned by the md5 manifest under validation/. The build runs
offline from the checked-in copy of the base under tokenizers/Apertus_1.
"""

import hashlib
from pathlib import Path

import pytest

from omnitok import build_apertus_1p5, prepare_apertus_1p5_text_base

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "validation" / "Apertus_1p5.md5"
BASE_DIR = REPO_ROOT / "tokenizers" / "Apertus_1"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def built_1p5(tmp_path_factory):
    out = tmp_path_factory.mktemp("apertus_1p5_build") / "Apertus_1p5"
    build_apertus_1p5(str(out), base_tokenizer_path=str(BASE_DIR))
    return out


def test_build_matches_canonical_manifest(built_1p5):
    """Every file pinned in validation/Apertus_1p5.md5 is reproduced exactly."""
    manifest = {}
    for line in MANIFEST.read_text().splitlines():
        digest, name = line.split()
        manifest[name] = digest
    assert manifest, "empty manifest"

    mismatches = {
        name: (_md5(built_1p5 / name), digest)
        for name, digest in manifest.items()
        if _md5(built_1p5 / name) != digest
    }
    assert not mismatches, (
        f"Built files diverge from the canonical Apertus 1.5: {mismatches}"
    )


def test_wrong_base_is_rejected(tmp_path):
    """Rebuilding on anything but the pinned Apertus 1 layout fails loudly
    instead of producing a differently-shaped artifact (e.g. a base where the
    reasoning/tool slots were already renamed)."""
    with pytest.raises(ValueError, match="not the expected Apertus 1 base"):
        prepare_apertus_1p5_text_base(
            str(REPO_ROOT / "tokenizers" / "Apertus_1p5"),
            str(tmp_path / "never_finished"),
        )
