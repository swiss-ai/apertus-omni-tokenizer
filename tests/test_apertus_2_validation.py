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


@pytest.mark.parametrize("eos", [[12, 2], [2, 12]])
def test_generation_settings_preserved_with_extra_fields(deployed, eos):
    config = {
        "eos_token_id": eos,
        "pad_token_id": 3,
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


# Each case breaks one rule of the check; None removes the file and a str is
# written verbatim.
INVALID_GENERATION_CONFIGS = {
    "missing": None,
    "malformed_json": '{"eos_token_id":[12,2],"pad_token_id":3, broken}',
    "no_pad": {"eos_token_id": [12, 2]},
    "no_eos": {"pad_token_id": 3},
    "wrong_eos": {"eos_token_id": [10, 2], "pad_token_id": 3},
    "old_layout": {"eos_token_id": 45, "pad_token_id": 46},
    "wrong_pad": {"eos_token_id": [12, 2], "pad_token_id": 46},
    "wait_only_scalar": {"eos_token_id": 12, "pad_token_id": 3},
    "wait_only_list": {"eos_token_id": [12], "pad_token_id": 3},
    "end_only": {"eos_token_id": [2], "pad_token_id": 3},
    "duplicate_eos": {"eos_token_id": [12, 12], "pad_token_id": 3},
    "extra_eos": {"eos_token_id": [12, 2, 10], "pad_token_id": 3},
    "float_eos": {"eos_token_id": 12.0, "pad_token_id": 3},
    "float_eos_in_list": {"eos_token_id": [12.0, 2], "pad_token_id": 3},
    "bool_eos_in_list": {"eos_token_id": [12, True], "pad_token_id": 3},
    "nested_eos": {"eos_token_id": [[12], 2], "pad_token_id": 3},
    "float_pad": {"eos_token_id": [12, 2], "pad_token_id": 3.0},
    "bool_pad": {"eos_token_id": [12, 2], "pad_token_id": True},
    "nested_pad": {"eos_token_id": [12, 2], "nested": {"pad_token_id": 3}},
    "non_finite": {"eos_token_id": [12, 2], "pad_token_id": 3, "temperature": float("nan")},
}


@pytest.mark.parametrize(
    "config", INVALID_GENERATION_CONFIGS.values(), ids=INVALID_GENERATION_CONFIGS
)
def test_invalid_eos_or_padding_rejected_without_replacement(deployed, config):
    generation = deployed / "generation_config.json"
    if config is None:
        generation.unlink()
    else:
        generation.write_text(config if isinstance(config, str) else json.dumps(config))
    before = generation.read_bytes() if generation.exists() else None
    result = validate(deployed, fix=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "requires eos_token_id [12, 2]" in result.stdout
    assert (generation.read_bytes() if generation.exists() else None) == before
    assert not list(deployed.glob("generation_config.json.bak*"))
