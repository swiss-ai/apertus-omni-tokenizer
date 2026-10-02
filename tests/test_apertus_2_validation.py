"""Deployment checks preserve checkpoint-specific generation settings."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def deployed(tmp_path):
    target = tmp_path / "model"
    shutil.copytree(ROOT / "tokenizers" / "Apertus_2_instruct", target)
    return target


def validate(target, fix=False, model="Apertus_2_instruct"):
    return subprocess.run(
        [
            "bash",
            str(ROOT / "validate_model.sh"),
            *(["--fix"] if fix else []),
            str(target),
            model,
        ],
        env={**os.environ, "BASE_URL": ROOT.as_uri(), "BRANCH": "."},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("eos", [45, [45]])
def test_generation_settings_preserved_with_extra_fields(deployed, eos):
    config = {
        "eos_token_id": eos,
        "pad_token_id": 46,
        "temperature": 0.7,
        "transformers_version": "checkpoint-specific",
    }
    generation = deployed / "generation_config.json"
    generation.write_text(json.dumps(config, indent=2))
    before = generation.read_bytes()
    for fix in (False, True):
        result = validate(deployed, fix)
        assert result.returncode == 0, result.stdout + result.stderr
        assert generation.read_bytes() == before
    assert not list(deployed.glob("generation_config.json.bak*"))


@pytest.mark.parametrize("eos,expected", [([2, 68, 72], 0), ([2, 68], 1)])
def test_apertus_1p5_retains_existing_eos_checks(tmp_path, eos, expected):
    target = tmp_path / "legacy"
    shutil.copytree(ROOT / "tokenizers" / "Apertus_1p5", target)
    shutil.copyfile(
        ROOT / "chat_templates" / "Apertus_1p5" / "chat_template.jinja",
        target / "chat_template.jinja",
    )
    (target / "generation_config.json").write_text(json.dumps({"eos_token_id": eos}))
    result = validate(target, model="Apertus_1p5")
    assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"eos_token_id": 45},
        {"pad_token_id": 46},
        {"eos_token_id": 43, "pad_token_id": 46},
        {"eos_token_id": [43, 45], "pad_token_id": 46},
        {"eos_token_id": [45, 45], "pad_token_id": 46},
        {"eos_token_id": "45", "pad_token_id": 46},
        {"eos_token_id": 45.0, "pad_token_id": 46},
        {"eos_token_id": 45, "pad_token_id": "46"},
        {"eos_token_id": 45, "pad_token_id": 3},
        {"nested": {"eos_token_id": 45, "pad_token_id": 46}},
        {"eos_token_id": 45, "nested": {"pad_token_id": 46}},
        [{"eos_token_id": 45, "pad_token_id": 46}],
        {"eos_token_id": [45.0], "pad_token_id": 46},
        {"eos_token_id": 45, "pad_token_id": 46, "temperature": float("nan")},
    ],
)
def test_invalid_eos_or_padding_rejected_without_replacement(deployed, config):
    generation = deployed / "generation_config.json"
    if config is None:
        generation.unlink()
        before = None
    else:
        generation.write_text(json.dumps(config))
        before = generation.read_bytes()
    result = validate(deployed, fix=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "requires eos_token_id 45 or [45]" in result.stdout
    assert (generation.read_bytes() if generation.exists() else None) == before
    assert not list(deployed.glob("generation_config.json.bak*"))


def test_malformed_generation_json_is_rejected_without_replacement(deployed):
    generation = deployed / "generation_config.json"
    invalid = '{"eos_token_id":45,"pad_token_id":46, broken}'
    generation.write_text(invalid)
    result = validate(deployed, fix=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert generation.read_text() == invalid
