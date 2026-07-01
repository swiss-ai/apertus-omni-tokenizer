"""The single engine for adding modalities to tokenizers.

Replaces both create_base_tokenizer (vision) and add_audio_tokens (audio)
with one modality-agnostic function.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from tokenizers import AddedToken
from transformers import AutoTokenizer

from .io import (
    add_token_alias,
    build_omnimodal_config,
    copy_modality_mapping_files,
    detect_existing_modalities,
    flip_token_normalized,
    rename_reserved_token,
    save_tokenizer,
    write_tokenizer_config,
)
from .modalities import INPLACE_REGISTRY, MODALITY_REGISTRY, ModalityConfig


def add_modality(
    input_tokenizer_path: str,
    output_path: str,
    modality: str | ModalityConfig,
    vocab_size: int,
    *,
    allocation: str = "append",
    slot_assignments: dict[str, int | str] | None = None,
    reserve_pool_pattern: str | None = None,
    allow_existing: bool = True,
    dry_run: bool = False,
    num_reserved_tokens: int = 200,
    extra_config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Add a modality to a tokenizer.

    Works for both fresh text tokenizers and existing omni-tokenizers.
    Idempotent: if the modality already exists with matching vocab_size, skips.

    Args:
        input_tokenizer_path: Path or HF model ID for input tokenizer.
        output_path: Where to save the result.
        modality: Modality name (from the registry for the chosen ``allocation``)
                  or a ModalityConfig instance.
        vocab_size: Number of content tokens (codebook size).
        allocation: ``"append"`` (default) appends a RESERVED_OMNI block on top of
                  the base vocab. ``"in_place"`` reuses pre-baked specials and the
                  ``<SPECIAL_*>`` reserve pool already present in the base vocab.
        slot_assignments: in-place only -- explicit ``{target_name: slot}`` overrides,
                  where ``slot`` is a pool ordinal (the N in ``<SPECIAL_N>``) or a full
                  reserve-token name, e.g. ``{"<|img_start|>": 40}`` or
                  ``{"<|img_start|>": "<SPECIAL_40>"}``.
        reserve_pool_pattern: in-place only -- override the ModalityConfig's reserve-pool
                  regex; one capture group = the ordinal (e.g. a pattern matching
                  ``<extra_id_0>``, ``<extra_id_1>``, ...).
        allow_existing: if a token to be added already exists, skip it (always);
                  when False, raise on any *unexpected* pre-existing token
                  (declared reuse tokens are exempt). Default True.
        dry_run: resolve and report the full plan without writing anything.
        num_reserved_tokens: RESERVED_OMNI slots for append mode (default 200).
        extra_config: Optional metadata for tokenizer_config.json
                      (e.g. {"type": "Emu3.5", "path": "/path/to/model"}).

    Returns:
        (tokenizer, stats) tuple. ``tokenizer`` is None when ``dry_run=True``.
        ``stats["report"]`` holds the human-readable change report.
    """
    # Resolve modality config
    mc = _resolve_modality(modality, allocation)

    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")

    print("=" * 60)
    print(f"ADDING MODALITY: {mc.name} (allocation={allocation})")
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
    print(f"Base vocab size: {base_vocab_size:,}")

    stats = {
        "input_tokenizer": input_tokenizer_path,
        "modality": mc.name,
        "allocation": allocation,
        "base_vocab_size": base_vocab_size,
        "original_vocab_size": current_vocab_size,
        "reserved_tokens_added": 0,
        "content_tokens_added": 0,
        "final_vocab_size": 0,
        "existing_modalities": list(existing["modalities"].keys()),
    }

    # Idempotency check (full re-run of an already-present modality)
    if mc.name in existing["modalities"]:
        existing_vs = existing["modalities"][mc.name].get("vocab_size")
        if existing_vs and existing_vs != vocab_size:
            raise ValueError(
                f"{mc.name} already exists with vocab_size={existing_vs}, "
                f"but requested {vocab_size}. Start from a tokenizer without "
                f"{mc.name} tokens to use a different vocab size."
            )
        print(f"\n{mc.name} tokens already exist. Skipping.")
        if dry_run:
            stats["final_vocab_size"] = current_vocab_size
            stats["dry_run"] = True
            return None, stats
        registry = INPLACE_REGISTRY if allocation == "in_place" else MODALITY_REGISTRY
        tokenizer.save_pretrained(output_path)
        copy_modality_mapping_files(existing, input_tokenizer_path, output_path)
        omnimodal_config = build_omnimodal_config(
            output_path, base_vocab_size, tokenizer,
            registry=registry, allocation=allocation,
        )
        write_tokenizer_config(
            output_path,
            tokenizer,
            base_vocab_size,
            omnimodal_config=omnimodal_config,
        )
        # Re-assert reused-alias normalized flags (idempotent).
        if allocation == "in_place":
            for r in mc.structure_tokens:
                if r.source == "reuse" and r.alias:
                    flip_token_normalized(output_path, r.target_name, True)
        stats["final_vocab_size"] = current_vocab_size
        tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)
        return tokenizer, stats

    # RESOLVE (read-only): compute + validate the full plan.
    plan = _plan_modality(
        tokenizer, mc, vocab_size,
        allocation=allocation,
        slot_assignments=slot_assignments,
        reserve_pool_pattern=reserve_pool_pattern,
        allow_existing=allow_existing,
        num_reserved_tokens=num_reserved_tokens,
    )
    stats["reserved_tokens_added"] = len(plan["reserved_tokens"])
    stats["content_tokens_added"] = len(plan["content_new"])
    # Resolved plan: identical in dry-run and real runs (callers can preview ids).
    stats["planned"] = {
        "structure_ids": dict(plan["structure_ids"]),
        "content_offset": plan["content_offset"],
        "content_count": len(plan["content_new"]),
        "special_region_offset": plan["special_region_offset"],
        "special_region_count": plan["special_region_count"],
        "reused_special_ids": list(plan["reused_special_ids"]),
        "projected_vocab_size": plan["projected_vocab_size"],
    }

    if dry_run:
        report = _build_report(plan, mc, allocation, True, input_tokenizer_path,
                               base_vocab_size)
        print("\n" + report)
        stats["report"] = report
        stats["final_vocab_size"] = plan["projected_vocab_size"]
        stats["dry_run"] = True
        return None, stats

    # APPLY (writes).
    tokenizer = _apply_modality(
        plan, tokenizer, output_path, input_tokenizer_path, existing, mc,
        vocab_size, base_vocab_size, extra_config, allocation,
    )
    stats["final_vocab_size"] = len(tokenizer)

    report = _build_report(plan, mc, allocation, False, input_tokenizer_path,
                           base_vocab_size, len(tokenizer))
    stats["report"] = report
    print("\n" + report)

    _print_verification(tokenizer, mc, vocab_size)
    return tokenizer, stats


# ── Private helpers ──────────────────────────────────────────────────────────


def _resolve_modality(
    modality: str | ModalityConfig, allocation: str = "append"
) -> ModalityConfig:
    if isinstance(modality, ModalityConfig):
        return modality
    registry = INPLACE_REGISTRY if allocation == "in_place" else MODALITY_REGISTRY
    if modality not in registry:
        available = list(registry.keys())
        raise ValueError(
            f"Unknown modality '{modality}' for allocation={allocation}. "
            f"Available: {available}"
        )
    return registry[modality]


def _partition_content(
    existing_vocab: dict[str, int], vocab_size: int, mc: ModalityConfig
) -> tuple[list[str], list[str]]:
    """Split content tokens into (new, already-present), preserving order."""
    new, preexisting = [], []
    for i in range(vocab_size):
        token = mc.content_token_format.format(i=i)
        (preexisting if token in existing_vocab else new).append(token)
    return new, preexisting


def _resolve_inplace_slots(
    mc: ModalityConfig,
    slot_assignments: dict[str, int | str] | None,
    vocab: dict[str, int],
    pool_pattern: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve every in-place structure token to a concrete id + action.

    ``vocab`` is the tokenizer's ``get_vocab()`` dict (passed in to avoid rebuilding
    it). Presence/id are resolved by dict membership -- NOT via
    ``convert_tokens_to_ids``, which returns ``unk_token_id`` for missing tokens and
    so cannot distinguish "absent" from "present at the unk id" (e.g. id 0).

    Returns a list of entries:
        {rename, target_name, id, rename_from, reuse, unexpected}

    ``reuse`` is True when the final-name token already exists (declared reuse, or
    an unexpected pre-existing pool/explicit target -> ``unexpected=True``).
    Otherwise a free ``<SPECIAL_n>`` slot is picked and ``rename_from`` is set.

    ``slot_assignments``/``explicit_slot`` values are pool ordinals (the N in
    ``<SPECIAL_N>``) OR full reserve-token names (``"<SPECIAL_40>"``) -- not token ids.
    ``pool_pattern`` overrides ``mc.reserve_pool_pattern`` (one capture group = the ordinal).

    Stacking works without bookkeeping: once a slot is renamed it no longer matches
    the reserve-pool pattern, so a later modality's discovery never re-picks it.
    """
    pattern = pool_pattern or mc.reserve_pool_pattern
    if not pattern:
        raise ValueError(
            f"{mc.name}: in_place mode requires a reserve-pool pattern "
            f"(pass reserve_pool_pattern or set it on the ModalityConfig)."
        )
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        raise ValueError(
            f"{mc.name}: invalid reserve_pool_pattern {pattern!r}: {e}"
        ) from e
    if compiled.groups != 1:
        raise ValueError(
            f"{mc.name}: reserve_pool_pattern {pattern!r} must have exactly one "
            f"capture group (the pool ordinal)."
        )
    slot_assignments = slot_assignments or {}

    # slot_assignments must name a structure token of this modality (catch typos),
    # and cannot target a reused token (which claims no pool slot).
    structure_targets = {r.target_name for r in mc.structure_tokens}
    reuse_targets = {r.target_name for r in mc.structure_tokens if r.source == "reuse"}
    for name in slot_assignments:
        if name not in structure_targets:
            raise ValueError(
                f"{mc.name}: slot_assignments key {name!r} is not a structure token "
                f"of this modality"
            )
        if name in reuse_targets:
            raise ValueError(
                f"{mc.name}: slot_assignments cannot target reused token {name!r} "
                f"(it has no reserve-pool slot)"
            )

    # Discover the free reserve pool: ordinal -> token string (+ reverse map).
    # fullmatch so a custom pattern must span the whole token, not just its prefix.
    pool: dict[int, str] = {}
    for tok in vocab:
        m = compiled.fullmatch(tok)
        if m:
            pool[int(m.group(1))] = tok
    pool_ordinals = sorted(pool)
    name_to_ordinal = {tok: o for o, tok in pool.items()}

    def _to_ordinal(value):
        # A requested slot is either a pool ordinal (int) or a full reserve-token
        # name (str, e.g. "<SPECIAL_40>").
        if isinstance(value, str):
            if value not in name_to_ordinal:
                raise ValueError(
                    f"{mc.name}: slot {value!r} is not a free reserve-pool token "
                    f"(pattern {pattern})"
                )
            return name_to_ordinal[value]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{mc.name}: slot {value!r} must be a pool ordinal (int) or a "
                f"reserve-token name (str)"
            )
        return value

    def _requested(rn):
        # explicit override (slot_assignments) > config explicit_slot > auto (None)
        if rn.target_name in slot_assignments:
            return _to_ordinal(slot_assignments[rn.target_name])
        return _to_ordinal(rn.explicit_slot) if rn.source == "explicit" else None

    # Pass 1: reserve explicitly-requested ordinals up-front so an auto pick can
    # never steal a slot that a later explicit token needs.
    claimed: set[int] = set()
    for r in mc.structure_tokens:
        if r.source == "reuse" or r.target_name in vocab:
            continue
        ordinal = _requested(r)
        if ordinal is None:
            continue
        if ordinal not in pool:
            raise ValueError(
                f"{mc.name}: slot {ordinal} for {r.target_name} is not a free "
                f"reserve-pool token (pool pattern {pattern})"
            )
        if ordinal in claimed:
            raise ValueError(
                f"{mc.name}: slot {ordinal} requested by more than one token"
            )
        claimed.add(ordinal)

    # Pass 2: assemble in declaration order; auto fills the lowest still-free slot.
    resolved: list[dict[str, Any]] = []
    for r in mc.structure_tokens:
        if r.source == "reuse":
            if r.existing_name not in vocab:
                raise ValueError(
                    f"{mc.name}: reuse target {r.existing_name} not found in vocab"
                )
            resolved.append({
                "rename": r, "target_name": r.target_name, "id": vocab[r.existing_name],
                "rename_from": None, "reuse": True, "unexpected": False,
            })
            continue

        # A pool/explicit target name that already exists -> reuse it.
        if r.target_name in vocab:
            resolved.append({
                "rename": r, "target_name": r.target_name, "id": vocab[r.target_name],
                "rename_from": None, "reuse": True, "unexpected": True,
            })
            continue

        ordinal = _requested(r)
        if ordinal is None:  # auto
            free = [o for o in pool_ordinals if o not in claimed]
            if not free:
                raise ValueError(
                    f"{mc.name}: reserve pool exhausted "
                    f"({len(pool_ordinals)} {pattern} slots, all claimed)"
                )
            ordinal = free[0]
            claimed.add(ordinal)
        resolved.append({
            "rename": r, "target_name": r.target_name, "id": vocab[pool[ordinal]],
            "rename_from": pool[ordinal], "reuse": False, "unexpected": False,
        })

    return resolved


def _plan_modality(
    tokenizer,
    mc: ModalityConfig,
    vocab_size: int,
    *,
    allocation: str,
    slot_assignments: dict[str, int | str] | None,
    reserve_pool_pattern: str | None = None,
    allow_existing: bool,
    num_reserved_tokens: int,
) -> dict[str, Any]:
    """Compute (and validate) everything needed to add the modality. No writes."""
    vocab = tokenizer.get_vocab()
    current_vocab_size = len(tokenizer)

    content_new, content_pre = _partition_content(vocab, vocab_size, mc)

    plan: dict[str, Any] = {
        "allocation": allocation,
        "current_vocab_size": current_vocab_size,
        "reserved_tokens": [],
        "content_new": content_new,
        "content_preexisting": content_pre,
        "renames": [],
        "reused": [],
        "structure_preexisting": [],
        "flips": [],
        "aliases": [],
        "structure_ids": {},
        "special_region_offset": None,
        "special_region_count": 0,
        "reused_special_ids": [],
    }

    if allocation == "in_place":
        resolved = _resolve_inplace_slots(
            mc, slot_assignments, vocab, pool_pattern=reserve_pool_pattern
        )
        for e in resolved:
            r = e["rename"]
            plan["structure_ids"][e["target_name"]] = e["id"]
            if e["reuse"]:
                bucket = "structure_preexisting" if e["unexpected"] else "reused"
                plan[bucket].append({"name": e["target_name"], "id": e["id"]})
            else:
                plan["renames"].append(
                    {"from": e["rename_from"], "to": e["target_name"], "id": e["id"]}
                )
            if r.alias:
                plan["aliases"].append({"alias": r.alias, "target": e["target_name"]})
                if e["reuse"]:
                    plan["flips"].append(e["target_name"])
        pool_ids = sorted(e["id"] for e in resolved if not e["reuse"])
        plan["special_region_offset"] = pool_ids[0] if pool_ids else None
        plan["special_region_count"] = len(pool_ids)
        plan["reused_special_ids"] = sorted(e["id"] for e in resolved if e["reuse"])
    else:
        max_slot = max(r.reserved_index for r in mc.structure_tokens)
        if num_reserved_tokens <= max_slot:
            raise ValueError(
                f"num_reserved_tokens={num_reserved_tokens} is too small: "
                f"{mc.name} uses slot {max_slot}. Must be at least {max_slot + 1}."
            )
        plan["reserved_tokens"] = _collect_reserved_tokens(vocab, num_reserved_tokens)
        for r in mc.structure_tokens:
            plan["renames"].append(
                {"from": f"<|RESERVED_OMNI_{r.reserved_index:03d}|>",
                 "to": r.target_name, "id": None}
            )
            if r.alias:
                plan["aliases"].append({"alias": r.alias, "target": r.target_name})

    # content_offset = id of codebook index 0: pre-existing id if present, else the append start.
    plan["content_append_start"] = current_vocab_size + len(plan["reserved_tokens"])
    token0 = mc.content_token_format.format(i=0)
    plan["content_offset"] = (
        vocab[token0] if token0 in vocab else plan["content_append_start"]
    )

    # allow_existing: only *unexpected* pre-existing tokens are violations.
    unexpected = list(content_pre) + [s["name"] for s in plan["structure_preexisting"]]
    if unexpected and not allow_existing:
        head = unexpected[:8]
        raise ValueError(
            f"{mc.name}: {len(unexpected)} token(s) already exist and "
            f"allow_existing=False: {head}{'...' if len(unexpected) > 8 else ''}"
        )

    plan["projected_vocab_size"] = (
        current_vocab_size + len(plan["content_new"]) + len(plan["reserved_tokens"])
    )
    return plan


def _apply_modality(
    plan: dict[str, Any],
    tokenizer,
    output_path: str,
    input_tokenizer_path: str,
    existing: dict[str, Any],
    mc: ModalityConfig,
    vocab_size: int,
    base_vocab_size: int,
    extra_config: dict[str, Any] | None,
    allocation: str,
):
    """Write the resolved plan to ``output_path``. Returns the reloaded tokenizer."""
    # 1. Add tokens via add_tokens(special_tokens=True) -- both the appended RESERVED_OMNI
    #    block (append mode) and the content tokens are atomic and stripped by
    #    skip_special_tokens, but NOT enrolled in the additional_special_tokens named role.
    #    So special_tokens_map.json stays clean (bos/eos/pad/unk only) and content ids stay
    #    out of all_special_ids/masks, in both modes and on any transformers version.
    to_add = plan["reserved_tokens"] + plan["content_new"]
    if to_add:
        num_added = tokenizer.add_tokens(to_add, special_tokens=True)
        print(f"Added {num_added:,} new tokens")

    # 2. Save tokenizer + base metadata.
    save_tokenizer(
        tokenizer, output_path, base_vocab_size,
        extra_config=extra_config, config_section_name=mc.config_section_name,
    )

    # 3. Preserve existing modality mapping files (stacking).
    copy_modality_mapping_files(existing, input_tokenizer_path, output_path)

    # 4. Rename structure placeholders -> final names (reuse entries have none).
    for rn in plan["renames"]:
        rename_reserved_token(output_path, tokenizer, rn["from"], rn["to"])

    tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)

    # 5. Aliases (batched in-memory, then one save_pretrained -- the LAST one).
    aliases = plan["aliases"]
    for a in aliases:
        add_token_alias(output_path, a["target"], a["alias"],
                        tokenizer=tokenizer, save=False)
    if aliases:
        tokenizer.save_pretrained(output_path)

    # 6. In-place: flip reused alias targets to normalized=True AFTER the final
    #    save_pretrained (see flip_token_normalized for why).
    if allocation == "in_place" and plan["flips"]:
        for name in plan["flips"]:
            flip_token_normalized(output_path, name, True)
        tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)

    # 7. Mapping file (must precede the omnimodal_config rebuild).
    _save_modality_mapping(
        output_path, mc, tokenizer, vocab_size, len(tokenizer), allocation, plan
    )

    # 8. omnimodal_config + full tokenizer_config write.
    registry = INPLACE_REGISTRY if allocation == "in_place" else MODALITY_REGISTRY
    omnimodal_config = build_omnimodal_config(
        output_path, base_vocab_size, tokenizer,
        registry=registry, allocation=allocation,
    )
    write_tokenizer_config(
        output_path, tokenizer, base_vocab_size,
        extra_config=extra_config, config_section_name=mc.config_section_name,
        omnimodal_config=omnimodal_config,
    )

    return AutoTokenizer.from_pretrained(output_path, use_fast=True)


def _build_report(
    plan: dict[str, Any],
    mc: ModalityConfig,
    allocation: str,
    dry_run: bool,
    input_path: str,
    base_vocab_size: int,
    final_vocab_size: int | None = None,
) -> str:
    """Format the verbose existing-vs-changed change report."""
    L = ["=" * 60,
         f"ADD MODALITY REPORT — {mc.name}   (allocation={allocation}, dry_run={dry_run})",
         "=" * 60,
         f"Input: {input_path}    base_vocab_size={base_vocab_size}"]

    if plan["reused"]:
        L.append("\nReused existing tokens (not re-added):")
        for r in plan["reused"]:
            flip = "   normalized → true" if r["name"] in plan["flips"] else ""
            L.append(f"  {r['name']}   id {r['id']}   [declared reuse]{flip}")
    if plan["structure_preexisting"]:
        L.append("\nPre-existing structure targets reused (allow_existing):")
        for r in plan["structure_preexisting"]:
            L.append(f"  {r['name']}   id {r['id']}")
    if plan["renames"]:
        label = "Renamed reserve-pool slots:" if allocation == "in_place" \
            else "Renamed RESERVED_OMNI slots:"
        L.append("\n" + label)
        for rn in plan["renames"]:
            idtxt = f"   id {rn['id']}" if rn["id"] is not None else ""
            L.append(f"  {rn['from']} → {rn['to']}{idtxt}")
    if plan["aliases"]:
        L.append("\nAliases added:")
        for a in plan["aliases"]:
            L.append(f"  {a['alias']} → {a['target']}")

    L.append("\nContent tokens:")
    n_new = len(plan["content_new"])
    if n_new:
        start = plan["content_append_start"]
        L.append(f"  added:   {n_new:,}   ids {start}..{start + n_new - 1}")
    else:
        L.append("  added:   0")
    L.append(f"  skipped: {len(plan['content_preexisting'])} already existed")

    if allocation == "in_place":
        L.append(
            f"\nMetadata: special_region_offset={plan.get('special_region_offset')}  "
            f"special_region_count={plan.get('special_region_count')}  "
            f"reused_special_ids={plan.get('reused_special_ids')}"
        )

    fv = final_vocab_size if final_vocab_size is not None else plan["projected_vocab_size"]
    added = n_new + len(plan["reserved_tokens"])
    L.append(f"\nFinal vocab size: {fv:,}   (+{added:,})")
    if dry_run:
        L.append("NO FILES WRITTEN (dry run)")
    L.append("=" * 60)
    return "\n".join(L)


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


def _save_modality_mapping(
    output_path: str,
    mc: ModalityConfig,
    tokenizer,
    vocab_size: int,
    final_vocab_size: int,
    allocation: str = "append",
    plan: dict[str, Any] | None = None,
) -> None:
    """Save the modality's token mapping JSON.

    Append-mode output is unchanged. In-place mode also records the claimed
    reserve-pool region (special_region_offset/count) and reused_special_ids.
    """
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
        "vocab_size": final_vocab_size,
        "structure_tokens": structure_tokens,
        f"{mc.name}_token_ids": mapping,
    }

    if allocation == "in_place" and plan is not None:
        data["special_region_offset"] = plan.get("special_region_offset")
        data["special_region_count"] = plan.get("special_region_count")
        data["reused_special_ids"] = plan.get("reused_special_ids", [])

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
