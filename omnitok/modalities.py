"""Modality configurations for omni-tokenizers.

Adding a new modality = adding one ModalityConfig here. No other code changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRename:
    """A structure-token claim with an optional alias.

    The ``source`` discriminator selects how the token's id is obtained:

    - ``"reserved"`` (append mode): claim ``<|RESERVED_OMNI_{reserved_index:03d}|>``
      from the appended reserved block, then rename it to ``target_name``.
    - ``"reuse"`` (in-place mode): the token already exists in the base vocab
      under ``existing_name`` (e.g. ``<|image|>``); wire its id, no rename,
      no pool slot consumed.
    - ``"pool"`` (in-place mode): auto-allocate the next free reserve-pool slot
      (matching ``ModalityConfig.reserve_pool_pattern``) and rename it.
    - ``"explicit"`` (in-place mode): claim the reserve-pool slot ``explicit_slot`` --
      a pool ordinal (``40`` => ``<SPECIAL_40>``) or a full reserve-token name
      (``"<SPECIAL_40>"``) -- and rename it.

    Backward compat: legacy positional entries ``TokenRename(1, "<|img_start|>")``
    set ``reserved_index`` and ``source`` is inferred as ``"reserved"``.
    """

    reserved_index: int | None = None
    target_name: str = ""
    alias: str | None = None
    source: str | None = None
    existing_name: str | None = None
    explicit_slot: int | str | None = None

    def __post_init__(self) -> None:
        if not self.target_name:
            raise ValueError("TokenRename requires a target_name")

        src = self.source
        if src is None:
            # Backward compat: a bare reserved_index implies the append-mode source.
            src = "reserved" if self.reserved_index is not None else None
            object.__setattr__(self, "source", src)

        if src == "reserved":
            if self.reserved_index is None:
                raise ValueError(
                    f"{self.target_name}: source='reserved' requires reserved_index"
                )
            if self.existing_name is not None or self.explicit_slot is not None:
                raise ValueError(
                    f"{self.target_name}: reserved source forbids existing_name/explicit_slot"
                )
        elif src == "reuse":
            if self.existing_name is None:
                raise ValueError(
                    f"{self.target_name}: source='reuse' requires existing_name"
                )
            if self.reserved_index is not None or self.explicit_slot is not None:
                raise ValueError(
                    f"{self.target_name}: reuse source forbids reserved_index/explicit_slot"
                )
        elif src == "pool":
            if (
                self.reserved_index is not None
                or self.existing_name is not None
                or self.explicit_slot is not None
            ):
                raise ValueError(
                    f"{self.target_name}: pool source forbids "
                    f"reserved_index/existing_name/explicit_slot"
                )
        elif src == "explicit":
            if self.explicit_slot is None:
                raise ValueError(
                    f"{self.target_name}: source='explicit' requires explicit_slot"
                )
            if self.existing_name is not None:
                raise ValueError(
                    f"{self.target_name}: explicit source forbids existing_name"
                )
        else:
            raise ValueError(f"{self.target_name}: unknown TokenRename source {src!r}")


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
    # In-place mode only: regex (one capture group = the slot ordinal) matching
    # the base tokenizer's pre-baked reserve pool, e.g. r"^<SPECIAL_(\d+)>$".
    # None => append-mode config (no pool).
    reserve_pool_pattern: str | None = None


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
        TokenRename(12, "<|tts_continue|>"),
        TokenRename(13, "<|audio|>", alias="<audio>"),
        TokenRename(14, "<|stt_translate|>"),
        TokenRename(15, "<|audio_annotate|>"),
    ),
)

# ── Registry ─────────────────────────────────────────────────────────────────

MODALITY_REGISTRY: dict[str, ModalityConfig] = {
    "vision": VISION,
    "audio": AUDIO,
}


# ── In-place modality configs ──────────────────────────────────────────────────
#
#  For tokenizers that PRE-BAKE special tokens inside the base vocab (e.g. the
#  200k tokenizer: <|image|>=18, <|audio|>=19, and a free pool <SPECIAL_27>..).
#  Structure tokens are not appended; image/audio are REUSED and the remaining
#  tokens are RENAMED from the <SPECIAL_*> pool (auto or explicit). Content
#  tokens are still appended at the end. VISION/AUDIO above are left untouched so
#  append-mode output (Apertus 1.5) stays byte-identical.

_RESERVE_POOL_PATTERN = r"^<SPECIAL_(\d+)>$"

VISION_INPLACE = ModalityConfig(
    name="vision",
    content_token_format="<|visual token {i}|>",
    mapping_file="vision_token_mapping.json",
    offset_key="vision_token_offset",
    vocab_size_key="visual_vocab_size",
    start_token="<|img_start|>",
    end_token="<|img_end|>",
    config_section_name="vision_tokenizer",
    reserve_pool_pattern=_RESERVE_POOL_PATTERN,
    structure_tokens=(
        TokenRename(target_name="<|img_start|>", source="pool"),
        TokenRename(target_name="<|img_end|>", source="pool"),
        TokenRename(target_name="<|img_token_start|>", source="pool"),
        TokenRename(target_name="<|img_end_of_row|>", source="pool"),
        TokenRename(target_name="<|img_end_of_frame|>", source="pool"),
        TokenRename(target_name="<|img_generation_start|>", source="pool"),
        TokenRename(
            target_name="<|image|>",
            source="reuse",
            existing_name="<|image|>",
            alias="<image>",
        ),
    ),
)

AUDIO_INPLACE = ModalityConfig(
    name="audio",
    content_token_format="<|audio token {i}|>",
    mapping_file="audio_token_mapping.json",
    offset_key="audio_token_offset",
    vocab_size_key="audio_vocab_size",
    start_token="<|audio_start|>",
    end_token="<|audio_end|>",
    config_section_name="audio_tokenizer",
    reserve_pool_pattern=_RESERVE_POOL_PATTERN,
    structure_tokens=(
        TokenRename(target_name="<|audio_start|>", source="pool"),
        TokenRename(target_name="<|audio_end|>", source="pool"),
        TokenRename(target_name="<|stt_transcribe|>", source="pool"),
        TokenRename(target_name="<|stt_continue|>", source="pool"),
        TokenRename(target_name="<|tts_continue|>", source="pool"),
        TokenRename(target_name="<|stt_translate|>", source="pool"),
        TokenRename(target_name="<|audio_annotate|>", source="pool"),
        TokenRename(
            target_name="<|audio|>",
            source="reuse",
            existing_name="<|audio|>",
            alias="<audio>",
        ),
    ),
)

INPLACE_REGISTRY: dict[str, ModalityConfig] = {
    "vision": VISION_INPLACE,
    "audio": AUDIO_INPLACE,
}
