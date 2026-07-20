"""omnitok — Create omnimodal tokenizers."""

from .apertus import build_apertus_1p5, prepare_apertus_1p5_text_base
from .builder import add_modality
from .instruct import create_instruct_tokenizer
from .io import detect_existing_modalities, get_content_token_id, load_modality_mapping
from .modalities import AUDIO, MODALITY_REGISTRY, VISION, ModalityConfig, TokenRename

__all__ = [
    "add_modality",
    "build_apertus_1p5",
    "prepare_apertus_1p5_text_base",
    "create_instruct_tokenizer",
    "detect_existing_modalities",
    "load_modality_mapping",
    "get_content_token_id",
    "ModalityConfig",
    "TokenRename",
    "VISION",
    "AUDIO",
    "MODALITY_REGISTRY",
]
