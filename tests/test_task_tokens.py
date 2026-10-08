"""The audio task tokens the tokenization pipeline relies on are declared.

A token missing from AUDIO.structure_tokens would silently tokenize to unk.
That every declared structure token resolves after a build is checked in
tests/test_builder.py.
"""

from omnitok.modalities import AUDIO

# Token names that the benchmark-audio-tokenizer pipeline relies on.
REQUIRED_TASK_TOKENS = [
    "<|stt_transcribe|>",
    "<|stt_continue|>",
    "<|stt_translate|>",
    "<|audio_annotate|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio|>",
]


def test_all_required_tokens_declared():
    declared = {r.target_name for r in AUDIO.structure_tokens}
    for token in REQUIRED_TASK_TOKENS:
        assert token in declared, (
            f"{token} is required by the tokenization pipeline but missing "
            f"from AUDIO.structure_tokens in modalities.py"
        )
