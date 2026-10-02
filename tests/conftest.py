"""Shared fixtures for omnitok tests."""

import pytest

from omnitok import add_modality, registry

# The checked-in Apertus 1 base: pinned and offline.
BASE_TOKENIZER = str(registry.artifact_dir("Apertus_1_base"))
SMALL_VOCAB = 32


@pytest.fixture(scope="session")
def vision_tokenizer(tmp_path_factory):
    """Vision-only tokenizer with small vocab, created once per session."""
    out = str(tmp_path_factory.mktemp("vision"))
    add_modality(BASE_TOKENIZER, out, "vision", SMALL_VOCAB)
    return out


@pytest.fixture(scope="session")
def audio_tokenizer(tmp_path_factory):
    """Audio-only tokenizer with small vocab, created once per session."""
    out = str(tmp_path_factory.mktemp("audio"))
    add_modality(BASE_TOKENIZER, out, "audio", SMALL_VOCAB)
    return out


@pytest.fixture(scope="session")
def stacked_tokenizer(tmp_path_factory, vision_tokenizer):
    """Vision + audio stacked tokenizer, created once per session."""
    stacked_out = str(tmp_path_factory.mktemp("stacked_both"))
    add_modality(vision_tokenizer, stacked_out, "audio", SMALL_VOCAB)
    return stacked_out
