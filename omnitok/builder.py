"""The single engine for adding modalities to tokenizers.

Replaces both create_base_tokenizer (vision) and add_audio_tokens (audio)
with one modality-agnostic function.
"""

from __future__ import annotations

import json
import os
from typing import Any

from tokenizers import AddedToken
from transformers import AutoTokenizer

from .io import (
    add_token_alias,
    build_omnimodal_config,
    copy_modality_mapping_files,
    detect_existing_modalities,
    mark_tokens_non_special,
    rename_reserved_token,
    save_tokenizer,
    strip_bos_from_post_processor,
    write_tokenizer_config,
)
from .modalities import MODALITY_REGISTRY, ModalityConfig


def add_modality(
    input_tokenizer_path: str,
    output_path: str,
    modality: str | ModalityConfig,
    vocab_size: int,
    *,
    num_reserved_tokens: int = 200,
    extra_config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Add a modality to a tokenizer.

    Works for both fresh text tokenizers and existing omni-tokenizers.
    Idempotent: if the modality already exists with matching vocab_size, skips.

    Args:
        input_tokenizer_path: Path or HF model ID for input tokenizer.
        output_path: Where to save the result.
        modality: Modality name (from registry) or a ModalityConfig instance.
        vocab_size: Number of content tokens (codebook size).
        num_reserved_tokens: RESERVED_OMNI slots (default 200).
        extra_config: Optional metadata for tokenizer_config.json
                      (e.g. {"type": "Emu3.5", "path": "/path/to/model"}).

    Returns:
        (tokenizer, stats) tuple.
    """
    # Resolve modality config
    mc = _resolve_modality(modality)

    print("=" * 60)
    print(f"ADDING MODALITY: {mc.name}")
    print("=" * 60)

    # Detect existing state
    existing = detect_existing_modalities(input_tokenizer_path)

    print(f"\nInput tokenizer: {input_tokenizer_path}")
    print(f"Modality: {mc.name}")
    print(f"Vocab size: {vocab_size:,}")
    if existing["modalities"]:
        print(f"Existing modalities: {list(existing['modalities'].keys())}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_path, use_fast=True)
    current_vocab_size = len(tokenizer)
    base_vocab_size = existing["base_vocab_size"] or current_vocab_size
    print(f"Current vocab size: {current_vocab_size:,}")
    print(f"Base vocab size (text-only): {base_vocab_size:,}")

    stats = {
        "input_tokenizer": input_tokenizer_path,
        "modality": mc.name,
        "base_vocab_size": base_vocab_size,
        "original_vocab_size": current_vocab_size,
        "reserved_tokens_added": 0,
        "content_tokens_added": 0,
        "final_vocab_size": 0,
        "existing_modalities": list(existing["modalities"].keys()),
    }

    # Idempotency check
    if mc.name in existing["modalities"]:
        existing_vs = existing["modalities"][mc.name].get("vocab_size")
        if existing_vs and existing_vs != vocab_size:
            raise ValueError(
                f"{mc.name} already exists with vocab_size={existing_vs}, "
                f"but requested {vocab_size}. Start from a tokenizer without "
                f"{mc.name} tokens to use a different vocab size."
            )
        print(f"\n{mc.name} tokens already exist. Skipping.")
        tokenizer.save_pretrained(output_path)
        copy_modality_mapping_files(existing, input_tokenizer_path, output_path)
        omnimodal_config = build_omnimodal_config(
            output_path, base_vocab_size, tokenizer
        )
        write_tokenizer_config(
            output_path,
            tokenizer,
            base_vocab_size,
            omnimodal_config=omnimodal_config,
        )
        mark_tokens_non_special(output_path)
        strip_bos_from_post_processor(output_path)
        stats["final_vocab_size"] = current_vocab_size
        tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)
        return tokenizer, stats

    # Validate num_reserved_tokens covers all claimed slots
    max_slot = max(r.reserved_index for r in mc.structure_tokens)
    if num_reserved_tokens <= max_slot:
        raise ValueError(
            f"num_reserved_tokens={num_reserved_tokens} is too small: "
            f"{mc.name} uses slot {max_slot}. "
            f"Must be at least {max_slot + 1}."
        )

    # Collect tokens to add
    existing_vocab = tokenizer.get_vocab()

    # 1. RESERVED_OMNI tokens
    reserved = _collect_reserved_tokens(existing_vocab, num_reserved_tokens)
    stats["reserved_tokens_added"] = len(reserved)

    # 2. Content tokens
    content = _collect_content_tokens(existing_vocab, vocab_size, mc)
    stats["content_tokens_added"] = len(content)
    print(f"\nAdding {len(reserved)} reserved + {len(content):,} content tokens...")

    # Add all tokens
    all_tokens = reserved + content
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": all_tokens}
    )
    print(f"Added {num_added:,} new tokens")

    stats["final_vocab_size"] = len(tokenizer)
    print(f"New vocab size: {stats['final_vocab_size']:,}")

    # Save
    save_tokenizer(
        tokenizer,
        output_path,
        base_vocab_size,
        extra_config=extra_config,
        config_section_name=mc.config_section_name,
    )

    # Copy existing modality mapping files
    copy_modality_mapping_files(existing, input_tokenizer_path, output_path)

    # Rename structure tokens
    print(f"\nRenaming RESERVED_OMNI tokens to {mc.name} structure tokens...")
    for rename in mc.structure_tokens:
        old = f"<|RESERVED_OMNI_{rename.reserved_index:03d}|>"
        rename_reserved_token(output_path, tokenizer, old, rename.target_name)

    # Reload so returned tokenizer has renames applied.
    tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)

    # Batch alias mutations in-memory, then save once.
    aliases = [r for r in mc.structure_tokens if r.alias]
    for rename in aliases:
        add_token_alias(output_path, rename.target_name, rename.alias,
                        tokenizer=tokenizer, save=False)
    if aliases:
        tokenizer.save_pretrained(output_path)

    # Mapping must be written before omnimodal_config is rebuilt from disk.
    _save_modality_mapping(output_path, mc, tokenizer, vocab_size, stats)
    omnimodal_config = build_omnimodal_config(output_path, base_vocab_size, tokenizer)
    write_tokenizer_config(
        output_path,
        tokenizer,
        base_vocab_size,
        extra_config=extra_config,
        config_section_name=mc.config_section_name,
        omnimodal_config=omnimodal_config,
    )

    # Keep the reasoning delimiters non-special so a reasoning parser can find
    # them in the detokenized output under the default skip_special_tokens=True
    # (apertus-omni-tokenizer #5).
    mark_tokens_non_special(output_path)

    # Drop the BOS from the post-processor: the chat template already emits it,
    # so auto-adding it on the /completions path double-BOSes a templated prompt
    # (apertus-program #420).
    strip_bos_from_post_processor(output_path)

    # Reload after all file mutations so the returned tokenizer matches disk.
    tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)

    # Verification
    _print_verification(tokenizer, mc, vocab_size)

    return tokenizer, stats


# ── Private helpers ──────────────────────────────────────────────────────────


def _resolve_modality(modality: str | ModalityConfig) -> ModalityConfig:
    if isinstance(modality, ModalityConfig):
        return modality
    if modality not in MODALITY_REGISTRY:
        available = list(MODALITY_REGISTRY.keys())
        raise ValueError(f"Unknown modality '{modality}'. Available: {available}")
    return MODALITY_REGISTRY[modality]


def _collect_reserved_tokens(
    existing_vocab: dict[str, int], num_reserved: int
) -> list:
    """Collect RESERVED_OMNI tokens, skipping slots already in vocab."""
    # Build set of all claimed target names across all modalities
    all_renamed = {}
    for mc in MODALITY_REGISTRY.values():
        for r in mc.structure_tokens:
            all_renamed[r.reserved_index] = r

    tokens = []
    for i in range(num_reserved):
        original = f"<|RESERVED_OMNI_{i:03d}|>"
        rename = all_renamed.get(i)

        # Skip if already present (original or renamed form)
        if original in existing_vocab:
            continue
        if rename and rename.target_name in existing_vocab:
            continue

        # Tokens with aliases need normalized=True
        if rename and rename.alias:
            tokens.append(AddedToken(original, normalized=True, special=True))
        else:
            tokens.append(original)

    return tokens


def _collect_content_tokens(
    existing_vocab: dict[str, int],
    vocab_size: int,
    mc: ModalityConfig,
) -> list:
    """Generate content tokens, skipping any that already exist."""
    tokens = []
    for i in range(vocab_size):
        token = mc.content_token_format.format(i=i)
        if token not in existing_vocab:
            tokens.append(token)
    return tokens


def _save_modality_mapping(
    output_path: str,
    mc: ModalityConfig,
    tokenizer,
    vocab_size: int,
    stats: dict[str, Any],
) -> None:
    """Save the modality's token mapping JSON."""
    mapping = {}
    for i in range(vocab_size):
        token = mc.content_token_format.format(i=i)
        mapping[i] = tokenizer.convert_tokens_to_ids(token)

    structure_tokens = {}
    for rename in mc.structure_tokens:
        key = rename.target_name.removeprefix("<|").removesuffix("|>")
        structure_tokens[key] = tokenizer.convert_tokens_to_ids(rename.target_name)

    data = {
        mc.vocab_size_key: vocab_size,
        f"{mc.name}_token_format": mc.content_token_format.replace("{i}", "N"),
        mc.offset_key: mapping[0],
        "vocab_size": stats["final_vocab_size"],
        "structure_tokens": structure_tokens,
        f"{mc.name}_token_ids": mapping,
    }

    mapping_path = os.path.join(output_path, mc.mapping_file)
    with open(mapping_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {mc.mapping_file}")


def _print_verification(
    tokenizer, mc: ModalityConfig, vocab_size: int
) -> None:
    """Print verification summary."""
    print("\n" + "=" * 60)
    print(f"VERIFICATION - {mc.name}")
    print("=" * 60)

    print(f"\n{mc.name} structure tokens:")
    for rename in mc.structure_tokens:
        tid = tokenizer.convert_tokens_to_ids(rename.target_name)
        if tid != tokenizer.unk_token_id:
            suffix = f" (alias: {rename.alias})" if rename.alias else ""
            print(f"  {rename.target_name}: ID {tid}{suffix}")

    print(f"\n{mc.name} content tokens (sample):")
    for idx in [0, vocab_size // 2, vocab_size - 1]:
        token = mc.content_token_format.format(i=idx)
        tid = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID {tid}")

    print("=" * 60)
