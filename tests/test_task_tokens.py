"""Verify that all audio task tokens expected by the tokenization pipeline
are defined in the modality config and resolve to real IDs after building."""

import pytest
from transformers import AutoTokenizer

from omnitok.modalities import AUDIO

# Token names that the benchmark-audio-tokenizer pipeline relies on.
# If any of these are missing from AUDIO.structure_tokens, the pipeline
# will silently get unk IDs and produce corrupt tokenized data.
REQUIRED_TASK_TOKENS = [
    "<|stt_transcribe|>",
    "<|stt_continue|>",
    "<|stt_translate|>",
    "<|audio_annotate|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio|>",
]


class TestTaskTokensInConfig:
    """Check that required tokens are declared in AUDIO.structure_tokens."""

    def test_all_required_tokens_declared(self):
        declared = {r.target_name for r in AUDIO.structure_tokens}
        for token in REQUIRED_TASK_TOKENS:
            assert token in declared, (
                f"{token} is required by the tokenization pipeline but missing "
                f"from AUDIO.structure_tokens in modalities.py"
            )

    def test_no_slot_collisions(self):
        slots = [r.reserved_index for r in AUDIO.structure_tokens]
        assert len(slots) == len(set(slots)), (
            f"Duplicate reserved_index in AUDIO.structure_tokens: {slots}"
        )


class TestTaskTokensResolve:
    """Check that required tokens resolve to real IDs (not unk) in a built tokenizer."""

    def test_task_tokens_not_unk(self, stacked_tokenizer):
        tok = AutoTokenizer.from_pretrained(stacked_tokenizer)
        unk = tok.unk_token_id
        for token in REQUIRED_TASK_TOKENS:
            tid = tok.convert_tokens_to_ids(token)
            assert tid != unk, (
                f"{token} resolved to unk (ID {unk}) in the built tokenizer. "
                f"This will cause silent data corruption during tokenization."
            )

    def test_task_tokens_unique_ids(self, stacked_tokenizer):
        tok = AutoTokenizer.from_pretrained(stacked_tokenizer)
        ids = {}
        for token in REQUIRED_TASK_TOKENS:
            tid = tok.convert_tokens_to_ids(token)
            assert tid not in ids.values(), (
                f"{token} (ID {tid}) collides with another task token"
            )
            ids[token] = tid
