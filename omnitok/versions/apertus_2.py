"""Apertus 2 omni build recipe.

Base: ``preliminary_mul_200k`` (cmeister/apertus_v2_tokenizer, decided in
swiss-ai/apertus-program#429 on 2026-06-30) — 200,064 text tokens with
pre-baked ``<|image|>``/``<|audio|>`` and a ``<SPECIAL_*>`` reserve pool.

The tables below are the spec: the build renames pool slots in place
(ids never move), reuses the pre-baked placeholders, and appends only
content tokens. The build asserts the base matches and fails loudly on
drift; ids verified against the published omni_mul200k_vision_audio_335232
artifact.
"""

from __future__ import annotations

from ..builder import add_modality_in_place

BASE_VOCAB_SIZE = 200_064
VISION_VOCAB_SIZE = 131_072
AUDIO_VOCAB_SIZE = 4_096
TOTAL_VOCAB_SIZE = 335_232

REUSED_IDS = {"<|image|>": 18, "<|audio|>": 19}

VISION_RENAMES = {
    "<SPECIAL_27>": "<|img_start|>",
    "<SPECIAL_28>": "<|img_end|>",
    "<SPECIAL_29>": "<|img_token_start|>",
    "<SPECIAL_30>": "<|img_end_of_row|>",
    "<SPECIAL_31>": "<|img_end_of_frame|>",
    "<SPECIAL_32>": "<|img_generation_start|>",
}

AUDIO_RENAMES = {
    "<SPECIAL_33>": "<|audio_start|>",
    "<SPECIAL_34>": "<|audio_end|>",
    "<SPECIAL_35>": "<|stt_transcribe|>",
    "<SPECIAL_36>": "<|stt_continue|>",
    "<SPECIAL_37>": "<|tts_continue|>",
    "<SPECIAL_38>": "<|stt_translate|>",
    "<SPECIAL_39>": "<|audio_annotate|>",
}


def build(input_tokenizer_path: str, output_path: str):
    """Build the Apertus 2 omni tokenizer from the pinned text base."""
    add_modality_in_place(
        input_tokenizer_path, output_path, "vision", VISION_VOCAB_SIZE,
        renames=VISION_RENAMES,
        reused_ids={"<|image|>": REUSED_IDS["<|image|>"]},
        expected_base_vocab_size=BASE_VOCAB_SIZE,
        publish_structure_ids=True,
    )
    tokenizer, stats = add_modality_in_place(
        output_path, output_path, "audio", AUDIO_VOCAB_SIZE,
        renames=AUDIO_RENAMES,
        reused_ids={"<|audio|>": REUSED_IDS["<|audio|>"]},
        expected_base_vocab_size=BASE_VOCAB_SIZE,
        publish_structure_ids=True,
    )
    if len(tokenizer) != TOTAL_VOCAB_SIZE:
        raise ValueError(
            f"built {len(tokenizer):,} tokens, expected {TOTAL_VOCAB_SIZE:,}"
        )
    return tokenizer, stats
