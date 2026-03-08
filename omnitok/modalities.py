"""Modality configurations for omni-tokenizers.

Adding a new modality = adding one ModalityConfig here. No other code changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRename:
    """A RESERVED_OMNI slot claim with optional alias."""

    reserved_index: int
    target_name: str
    alias: str | None = None


@dataclass(frozen=True)
class ModalityConfig:
    """Everything needed to add one modality to an omni-tokenizer."""

    name: str
    content_token_format: str  # Python format string with {i}
    mapping_file: str
    offset_key: str
    vocab_size_key: str
    start_token: str
    end_token: str
    structure_tokens: tuple[TokenRename, ...]
    config_section_name: str | None = None


# ── Built-in modality configs ────────────────────────────────────────────────
#
#  Slot allocation:
#    0       boundary marker (never renamed)
#    1-7     vision
#    8-14    audio
#    15-199  reserved for future modalities

VISION = ModalityConfig(
    name="vision",
    content_token_format="<|visual token {i}|>",
    mapping_file="vision_token_mapping.json",
    offset_key="vision_token_offset",
    vocab_size_key="visual_vocab_size",
    start_token="<|img_start|>",
    end_token="<|img_end|>",
    config_section_name="vision_tokenizer",
    structure_tokens=(
        TokenRename(1, "<|img_start|>"),
        TokenRename(2, "<|img_end|>"),
        TokenRename(3, "<|img_token_start|>"),
        TokenRename(4, "<|img_end_of_row|>"),
        TokenRename(5, "<|img_end_of_frame|>"),
        TokenRename(6, "<|img_generation_start|>"),
        TokenRename(7, "<|image|>", alias="<image>"),
    ),
)

AUDIO = ModalityConfig(
    name="audio",
    content_token_format="<|audio token {i}|>",
    mapping_file="audio_token_mapping.json",
    offset_key="audio_token_offset",
    vocab_size_key="audio_vocab_size",
    start_token="<|audio_start|>",
    end_token="<|audio_end|>",
    config_section_name="audio_tokenizer",
    structure_tokens=(
        TokenRename(8, "<|audio_start|>"),
        TokenRename(9, "<|audio_end|>"),
        TokenRename(10, "<|stt_transcribe|>"),
        TokenRename(11, "<|stt_continue|>"),
        TokenRename(12, "<|tts_synthesize|>"),
        TokenRename(13, "<|tts_continue|>"),
        TokenRename(14, "<|audio|>", alias="<audio>"),
    ),
)

# ── Registry ─────────────────────────────────────────────────────────────────

MODALITY_REGISTRY: dict[str, ModalityConfig] = {
    "vision": VISION,
    "audio": AUDIO,
}
