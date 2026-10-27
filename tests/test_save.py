"""save_tokenizer_files, the last step of every recipe.

The compatible mode writes the committed bytes on every supported version
(the full rebuilds in tests/test_registry.py check it end to end). The
current_version mode is the installed transformers' own save_pretrained; it
is only checked within one version, saved and loaded by the same release,
because other major versions need not load it.
"""

import json
import shutil
import sys

import pytest
from transformers import AutoTokenizer

from omnitok import cli, registry
from omnitok.io import TOKENIZER_FILES, load_backend_state, save_tokenizer_files
from omnitok.recipes import apertus_1p5, apertus_2
from omnitok.registry import Derived

DERIVED = [e.name for e in registry.TOKENIZERS if isinstance(e, Derived)]


def _inputs(name):
    """The committed artifact as save_tokenizer_files takes it."""
    files = registry.committed_files(name)
    config = json.loads(files["tokenizer_config.json"].read_text(encoding="utf-8"))
    if "chat_template.jinja" in files:
        config["chat_template"] = files["chat_template.jinja"].read_text(encoding="utf-8")
    return load_backend_state(str(registry.artifact_dir(name))), config, files


@pytest.mark.parametrize("name", DERIVED)
def test_compatible_save_reproduces_the_committed_files(name, tmp_path):
    state, config, files = _inputs(name)
    save_tokenizer_files(tmp_path, state, config)
    expected = {f: p for f, p in files.items() if f in TOKENIZER_FILES}
    assert {p.name for p in tmp_path.iterdir()} == set(expected)
    for f, path in expected.items():
        assert (tmp_path / f).read_bytes() == path.read_bytes(), f


@pytest.fixture(scope="module")
def current_version_save(tmp_path_factory):
    """Apertus 1.5 saved in current_version mode, over a stale file."""
    state, config, _ = _inputs("Apertus_1p5")
    ours = tmp_path_factory.mktemp("current_version")
    (ours / "special_tokens_map.json").write_text("stale")  # removed first
    save_tokenizer_files(ours, state, config, mode="current_version")
    return ours


def test_current_version_save_is_the_installed_save_pretrained(
    current_version_save, tmp_path
):
    deployed = tmp_path / "deployed"
    deployed.mkdir()
    for f, path in registry.committed_files("Apertus_1p5").items():
        shutil.copy(path, deployed / f)
    plain = tmp_path / "plain"
    AutoTokenizer.from_pretrained(deployed).save_pretrained(plain)

    ours = current_version_save
    assert {p.name for p in ours.iterdir()} == {p.name for p in plain.iterdir()}
    for path in plain.iterdir():
        assert (ours / path.name).read_bytes() == path.read_bytes(), path.name


def test_current_version_save_loads_back_in_the_same_version(current_version_save):
    tok = AutoTokenizer.from_pretrained(current_version_save)
    for literal, token_id in apertus_1p5._VERIFY_ENCODINGS.items():
        assert tok.encode(literal, add_special_tokens=False) == [token_id], literal
    assert tok.eos_token == "</s>"
    for attr, token in apertus_1p5.EXTRA_SPECIAL_TOKENS.items():
        assert getattr(tok, attr) == token, attr
    template = registry.committed_files("Apertus_1p5")["chat_template.jinja"]
    assert tok.chat_template == template.read_text(encoding="utf-8")


def test_unknown_save_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="save mode"):
        save_tokenizer_files(tmp_path, {}, {}, mode="native")


def test_instruct_is_saved_only_in_compatible_mode(tmp_path):
    with pytest.raises(ValueError, match="compatible"):
        apertus_2.build_instruct(
            registry.artifact_dir("Apertus_2"), tmp_path / "out",
            save_mode="current_version",
        )
    assert not (tmp_path / "out").exists()


def test_cli_passes_the_save_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(apertus_1p5, "build_apertus_1p5", lambda **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", [
        "omnitok", "build-apertus-1p5", "--output-path", "out",
        "--save-mode", "current_version",
    ])
    cli.main()
    assert calls[0]["save_mode"] == "current_version"


@pytest.mark.parametrize("mode", ["compatible", "current_version"])
def test_apertus_2_named_roles_survive_save_load(mode, tmp_path):
    state, config, _ = _inputs("Apertus_2_instruct")
    save_tokenizer_files(tmp_path, state, config, mode=mode)
    tok = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    for role, glyph in apertus_2.EXTRA_SPECIAL_TOKENS.items():
        assert getattr(tok, role) == glyph
        assert getattr(tok, role + "_id") == apertus_2.CONTROLS[glyph]
    assert (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) == (1, 12, 3)
    assert tok.encode("Hello") == [36971]
    assert {i for i, t in tok.added_tokens_decoder.items() if t.special} == set(range(124))
