"""Modality configurations for omni-tokenizers.

Adding a new modality = adding one ModalityConfig here. No other code changes.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    start_token: str
    end_token: str
    structure_tokens: tuple[TokenRename, ...]
    config_section_name: str | None = None
    # HF named-attribute mapping (attribute name -> token string) emitted into
    # tokenizer_config.json's `extra_special_tokens`; transformers (>= 4.47)
    # exposes each entry as a tokenizer attribute (tokenizer.image_token, ...)
    # used by processors and chat templates. Every value must be one of this
    # modality's tokens (validated by hf_extra_special_tokens).
    hf_named_tokens: tuple[tuple[str, str], ...] = ()


# ── Built-in modality configs ────────────────────────────────────────────────
#
#  Slot allocation:
#    0       boundary marker (never renamed)
#    1-7     vision
#    8-15    audio
#    16-199  reserved for future modalities

VISION = ModalityConfig(
    name="vision",
    content_token_format="<|visual token {i}|>",
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
    hf_named_tokens=(
        ("image_token", "<|image|>"),
        ("boi_token", "<|img_start|>"),
        ("eoi_token", "<|img_end|>"),
        ("image_wrapper_token", "<|img_token_start|>"),
        ("eol_token", "<|img_end_of_row|>"),
    ),
)

AUDIO = ModalityConfig(
    name="audio",
    content_token_format="<|audio token {i}|>",
    start_token="<|audio_start|>",
    end_token="<|audio_end|>",
    config_section_name="audio_tokenizer",
    structure_tokens=(
        TokenRename(8, "<|audio_start|>"),
        TokenRename(9, "<|audio_end|>"),
        TokenRename(10, "<|stt_transcribe|>"),
        TokenRename(11, "<|stt_continue|>"),
        TokenRename(12, "<|tts_continue|>"),
        TokenRename(13, "<|audio|>", alias="<audio>"),
        TokenRename(14, "<|stt_translate|>"),
        TokenRename(15, "<|audio_annotate|>"),
    ),
    hf_named_tokens=(
        ("audio_token", "<|audio|>"),
        # boa/eoa follow the boi/eoi image-token convention
        ("boa_token", "<|audio_start|>"),
        ("eoa_token", "<|audio_end|>"),
    ),
)

# ── Registry ─────────────────────────────────────────────────────────────────

MODALITY_REGISTRY: dict[str, ModalityConfig] = {
    "vision": VISION,
    "audio": AUDIO,
}


def hf_extra_special_tokens(modalities: Iterable[ModalityConfig]) -> dict[str, str]:
    """Merged HF named-token mapping for the given modalities.

    Returns e.g. {"image_token": "<|image|>", ...} for writing into
    tokenizer_config.json's `extra_special_tokens`. Validates that every named
    token is one of its modality's tokens and that names are unique across
    modalities, so the mapping cannot drift from the modality definitions.
    """
    named: dict[str, str] = {}
    for mc in modalities:
        modality_tokens = {mc.start_token, mc.end_token}
        modality_tokens.update(t.target_name for t in mc.structure_tokens)
        # instruct.py writes SFT id-lists under these top-level keys; an attribute
        # of the same name would overwrite them on a save_pretrained round trip
        reserved_names = {f"{mc.name}_begin_token", f"{mc.name}_end_token"}
        for name, token in mc.hf_named_tokens:
            if name in reserved_names or name.startswith("sft_"):
                raise ValueError(
                    f"{mc.name}: named special token {name!r} collides with a reserved "
                    "tokenizer_config key (SFT metadata written by instruct.py)"
                )
            if token not in modality_tokens:
                raise ValueError(
                    f"{mc.name}: named special token {name}={token!r} is not "
                    f"one of the modality's tokens"
                )
            if name in named:
                raise ValueError(f"Duplicate named special token: {name}")
            named[name] = token
    return named
