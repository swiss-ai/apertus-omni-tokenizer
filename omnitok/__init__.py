"""omnitok — Create omnimodal tokenizers."""

from .builder import add_modality
from .instruct import create_instruct_tokenizer
from .io import detect_existing_modalities, get_content_token_id, load_modality_mapping
from .modalities import (
    AUDIO,
    AUDIO_INPLACE,
    INPLACE_REGISTRY,
    MODALITY_REGISTRY,
    VISION,
    VISION_INPLACE,
    ModalityConfig,
    TokenRename,
)

__all__ = [
    "add_modality",
    "create_instruct_tokenizer",
    "detect_existing_modalities",
    "load_modality_mapping",
    "get_content_token_id",
    "ModalityConfig",
    "TokenRename",
    "VISION",
    "AUDIO",
    "MODALITY_REGISTRY",
    "VISION_INPLACE",
    "AUDIO_INPLACE",
    "INPLACE_REGISTRY",
]
