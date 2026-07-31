"""Apertus 2 omni build recipe.

Base: ``preliminary_mul_200k`` (cmeister/apertus_v2_tokenizer, decided in
swiss-ai/apertus-program#429 on 2026-06-30) — 200,064 text tokens with
pre-baked ``<|image|>``/``<|audio|>`` and a ``<SPECIAL_*>`` reserve pool.

The tables below are the spec: the build renames pool slots in place
(ids never move), reuses the pre-baked placeholders, and appends only
content tokens. The build asserts the base matches and fails loudly on
drift; ids verified against the published omni_mul200k_vision_audio_335232
artifact.

Base tokenizer only.
The instruct stage is not defined yet -- see ``build_instruct``.
"""

from __future__ import annotations

from ..builder import add_modality_in_place
from ..io import _rewrite_backend_state, finalize_tokenizer_config

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


# An explicit allowlist: the artifact must not inherit save_pretrained's shape.
# Absent on purpose: the added_tokens_decoder mirror (24MB of duplication),
# and the backend/is_local/local_files_only fossils that 4.x cannot load.
CARRIED_CONFIG_KEYS = (
    "add_prefix_space", "added_tokens_count", "base_vocab_size", "bos_token",
    "clean_up_tokenization_spaces", "eos_token", "model_input_names",
    "model_max_length", "omnimodal_config", "pad_token", "padding_side",
    "unk_token",
)


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

    # save_pretrained re-adds an empty TemplateProcessing on transformers 5.x,
    # so the strip has to be reasserted on the written file.
    # BOS is template-owned, and nothing may auto-append EOS to a prompt.
    def _drop_post_processor(state):
        state["post_processor"] = None

    _rewrite_backend_state(output_path, _drop_post_processor)

    finalize_tokenizer_config(
        output_path,
        carried_keys=CARRIED_CONFIG_KEYS,
        overrides={
            "tokenizer_class": "PreTrainedTokenizerFast",
            "vocab_size": TOTAL_VOCAB_SIZE,
        },
    )
    return tokenizer, stats


def build_instruct(input_tokenizer_path: str, output_path: str):
    """Not implemented: the Apertus 2 instruct conventions are undecided."""
    raise NotImplementedError(
        "The Apertus 2 instruct tokenizer is not defined yet: no chat template "
        "under chat_templates/Apertus_2/, and the eos convention and SFT "
        "begin/end sequences have not been decided. build() produces the base "
        "omni tokenizer; add the instruct stage here once those are settled."
    )
