"""The modality engine: adds vision/audio tokens to any text tokenizer.

Two entry points for the two allocation strategies, both converging on the
shared _assemble tail:

    add_modality           append a RESERVED_OMNI pool, rename it, append content
    add_modality_in_place  rename a pre-baked <SPECIAL_*> pool, append content
"""

from __future__ import annotations

import json
from typing import Any

from tokenizers import AddedToken
from transformers import AutoTokenizer

from .io import (
    _resolve_tokenizer_path,
    _rewrite_backend_state,
    assert_droppable_post_processor,
    add_token_alias,
    build_omnimodal_config,
    detect_existing_modalities,
    rename_reserved_tokens,
    save_tokenizer,
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
    revision: str | None = None,
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
        revision: Hub commit to pin when input_tokenizer_path is a repo ID.
                  Hub repos are mutable (tokens can be renamed upstream after
                  release), so reproducible builds should pin one.

    Returns:
        (tokenizer, stats) tuple.
    """
    # Resolve modality config
    mc = _resolve_modality(modality)
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    print("=" * 60)
    print(f"ADDING MODALITY: {mc.name}")
    print("=" * 60)

    input_tokenizer_path = _resolve_tokenizer_path(input_tokenizer_path, revision=revision)

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

    stats = _init_stats(input_tokenizer_path, mc, base_vocab_size,
                        current_vocab_size, existing)

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
        omnimodal_config = build_omnimodal_config(base_vocab_size, tokenizer)
        write_tokenizer_config(
            output_path,
            tokenizer,
            base_vocab_size,
            omnimodal_config=omnimodal_config,
        )
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

    renames = {
        f"<|RESERVED_OMNI_{r.reserved_index:03d}|>": r.target_name
        for r in mc.structure_tokens
    }
    tokenizer = _assemble(
        output_path, tokenizer, mc, vocab_size, base_vocab_size,
        renames, extra_config,
    )
    return tokenizer, stats


def add_modality_in_place(
    input_tokenizer_path: str,
    output_path: str,
    modality: str | ModalityConfig,
    vocab_size: int,
    *,
    renames: dict[str, str],
    reused_ids: dict[str, int],
    expected_base_vocab_size: int | None = None,
    publish_structure_ids: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Add a modality by renaming the base's pre-baked reserve slots.

    For bases that ship their own special-token pool (Apertus 2): the pool
    slots in ``renames`` are renamed in place, the placeholders in
    ``reused_ids`` (name -> pinned id) are reused as-is, and only content
    tokens are appended. Raises if the base does not match the recipe.

    One-shot per modality: rebuild from the base rather than re-running
    over an output. ``publish_structure_ids`` records each structure
    token's id in omnimodal_config.

    Returns:
        (tokenizer, stats) tuple.
    """
    mc = _resolve_modality(modality)
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    print("=" * 60)
    print(f"ADDING MODALITY IN PLACE: {mc.name}")
    print("=" * 60)

    existing = detect_existing_modalities(input_tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_path, use_fast=True)
    base_vocab_size = existing["base_vocab_size"] or len(tokenizer)
    if expected_base_vocab_size and base_vocab_size != expected_base_vocab_size:
        raise ValueError(
            f"base vocab is {base_vocab_size:,}, "
            f"expected {expected_base_vocab_size:,}"
        )
    print(f"\nInput tokenizer: {input_tokenizer_path}")
    print(f"Base vocab size (text-only): {base_vocab_size:,}")

    _strip_post_processor(tokenizer)
    _assert_in_place_base(tokenizer, renames, reused_ids)

    stats = _init_stats(input_tokenizer_path, mc, base_vocab_size,
                        len(tokenizer), existing)

    content = _collect_content_tokens(tokenizer.get_vocab(), vocab_size, mc)
    stats["content_tokens_added"] = len(content)
    print(f"\nAdding {len(content):,} content tokens...")
    tokenizer.add_tokens(content, special_tokens=True)

    stats["final_vocab_size"] = len(tokenizer)
    print(f"New vocab size: {stats['final_vocab_size']:,}")

    tokenizer = _assemble(
        output_path, tokenizer, mc, vocab_size, base_vocab_size,
        renames, extra_config=None,
        publish_structure_ids=publish_structure_ids,
    )
    return tokenizer, stats


# ── Private helpers ──────────────────────────────────────────────────────────


def _init_stats(input_tokenizer_path, mc, base_vocab_size, original_vocab_size,
                existing) -> dict[str, Any]:
    """The stats skeleton both entry points fill in as they go."""
    return {
        "input_tokenizer": input_tokenizer_path,
        "modality": mc.name,
        "base_vocab_size": base_vocab_size,
        "original_vocab_size": original_vocab_size,
        "reserved_tokens_added": 0,
        "content_tokens_added": 0,
        "final_vocab_size": 0,
        "existing_modalities": list(existing["modalities"].keys()),
    }


def _strip_post_processor(tokenizer) -> None:
    """Drop the base's BOS/EOS-injecting post-processor.

    BOS is template-owned and nothing may auto-append EOS to prompts
    (apertus-program#420); SFT packing needs exact encoding.
    See assert_droppable_post_processor for the shapes this refuses.
    """
    backend = tokenizer.backend_tokenizer
    pp = backend.post_processor
    if pp is None:
        return
    declared = {tokenizer.bos_token, tokenizer.eos_token} - {None}
    assert_droppable_post_processor(json.loads(pp.__getstate__()), declared)
    backend.post_processor = None
    print("Stripped base BOS/EOS post-processor (specials are template-owned)")


def _assert_in_place_base(
    tokenizer,
    renames: dict[str, str],
    reused_ids: dict[str, int],
) -> None:
    """The base must carry the recipe's pool slots and placeholders."""
    vocab = tokenizer.get_vocab()
    added = {t.content for t in tokenizer.added_tokens_decoder.values()}
    missing = [s for s in renames if s not in added]
    if missing:
        raise ValueError(f"base is missing reserve slots: {missing}")
    taken = [t for t in renames.values() if t in vocab]
    if taken:
        raise ValueError(f"rename targets already exist in the base: {taken}")
    for name, expected in reused_ids.items():
        if name not in added:
            raise ValueError(f"reused token {name} is not an added token")
        if vocab[name] != expected:
            raise ValueError(f"{name} has id {vocab[name]}, expected {expected}")


def _assemble(
    output_path: str,
    tokenizer,
    mc: ModalityConfig,
    vocab_size: int,
    base_vocab_size: int,
    renames: dict[str, str],
    extra_config: dict[str, Any] | None,
    publish_structure_ids: bool = False,
) -> Any:
    """Save, rename, alias, and write omnimodal metadata.

    Re-strips the post-processor after the last write when the caller had
    already dropped it: every save_pretrained re-adds an empty
    TemplateProcessing on transformers 5.x.
    Returns the reloaded tokenizer.
    """
    had_post_processor = tokenizer.backend_tokenizer.post_processor is not None
    save_tokenizer(
        tokenizer,
        output_path,
        base_vocab_size,
        extra_config=extra_config,
        config_section_name=mc.config_section_name,
    )

    # Rename structure tokens
    print(f"\nRenaming reserved tokens to {mc.name} structure tokens...")
    rename_reserved_tokens(output_path, tokenizer, renames)

    # Reload so returned tokenizer has renames applied.
    tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)

    # Batch alias mutations in-memory, then save once.
    aliases = [r for r in mc.structure_tokens if r.alias]
    for rename in aliases:
        add_token_alias(output_path, rename.target_name, rename.alias,
                        tokenizer=tokenizer, save=False)
    if aliases:
        tokenizer.save_pretrained(output_path)

    # build_omnimodal_config verifies content-id contiguity for every modality.
    omnimodal_config = build_omnimodal_config(
        base_vocab_size, tokenizer, publish_structure_ids=publish_structure_ids
    )
    built = next(
        (m for m in omnimodal_config.get("modalities", []) if m["name"] == mc.name),
        None,
    )
    built_size = built["vocab_size"] if built else 0
    if built_size != vocab_size:
        raise ValueError(
            f"{mc.name} ended up with {built_size} content tokens, "
            f"expected {vocab_size}"
        )
    write_tokenizer_config(
        output_path,
        tokenizer,
        base_vocab_size,
        extra_config=extra_config,
        config_section_name=mc.config_section_name,
        omnimodal_config=omnimodal_config,
    )

    if not had_post_processor:
        # Every save_pretrained re-adds an empty TemplateProcessing on
        # transformers 5.x, so a strip made before the saves is reasserted
        # after the last of them.
        declared = {tokenizer.bos_token, tokenizer.eos_token} - {None}

        def _drop(state):
            assert_droppable_post_processor(state.get("post_processor"), declared)
            state["post_processor"] = None

        _rewrite_backend_state(output_path, _drop)

    # Reload after all file mutations so the returned tokenizer matches disk.
    tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)

    # Verification
    _print_verification(tokenizer, mc, vocab_size)

    return tokenizer


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
