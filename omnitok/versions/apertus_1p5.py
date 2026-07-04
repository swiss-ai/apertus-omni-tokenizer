"""Apertus 1.0 / 1.5 omni build recipe (frozen).

The shipped artifacts under tokenizers/ are canonical and never rebuilt;
this records the inputs that produced them: append a 200-slot RESERVED_OMNI
block on top of the text base, rename slots into structure tokens,
append content tokens.
"""

from __future__ import annotations

BASE = "swiss-ai/Apertus-8B-2509"
NUM_RESERVED_TOKENS = 200
VISION_VOCAB_SIZE = 131_072
AUDIO_VOCAB_SIZE = 4_096


def build(input_tokenizer_path: str = BASE, output_path: str = "./omni_vision_audio"):
    """Rebuild a 1.5-style omni tokenizer (for reproduction tests, not shipping)."""
    from ..builder import add_modality

    add_modality(
        input_tokenizer_path, output_path, "vision", VISION_VOCAB_SIZE,
        num_reserved_tokens=NUM_RESERVED_TOKENS,
    )
    return add_modality(
        output_path, output_path, "audio", AUDIO_VOCAB_SIZE,
        num_reserved_tokens=NUM_RESERVED_TOKENS,
    )
