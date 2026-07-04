"""Low-level tokenizer file I/O utilities.

Handles renaming tokens, adding aliases, saving configs, detecting modalities.
No modality-specific knowledge -- operates on ModalityConfig generically.

Token layout after add_modality("vision", vocab_size=V):

    [0 .. base-1]       text tokens (unchanged)
    [base]              <|RESERVED_OMNI_000|>  (boundary marker, never renamed)
    [base+1 .. base+7]  vision structure tokens (img_start, img_end, ..., image)
    [base+8 .. base+199] remaining reserved slots (for future modalities)
    [base+200 .. end]   V vision content tokens (<|visual token 0|> .. <|visual token V-1|>)

Vocab size gotcha -- three ways to query, different answers:

    tokenizer.vocab_size          # base model vocab only, NEVER includes added tokens
    len(tokenizer)                # base + added tokens (correct total)
    len(tokenizer.get_vocab())    # base + added tokens (correct total)

After adding modality tokens, ``tokenizer.vocab_size`` still returns the
original text-only size (e.g. 131072).  Always use ``len(tokenizer)`` or
``len(tokenizer.get_vocab())`` to get the true total.

Call flow (driven by builder.add_modality):

    save_tokenizer()              # save HF tokenizer + write base metadata
    rename_reserved_tokens()      # e.g. <|RESERVED_OMNI_001|> -> <|img_start|>
    add_token_alias()             # e.g. <image> encodes to same ID as <|image|>
    build_omnimodal_config()      # derive omnimodal metadata from the tokenizer
    write_tokenizer_config()      # write all custom tokenizer_config.json fields
"""

from __future__ import annotations

import json
import os
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .modalities import MODALITY_REGISTRY, ModalityConfig


def _resolve_tokenizer_path(tokenizer_path: str) -> str:
    """Resolve a tokenizer path to a local directory.

    If ``tokenizer_path`` is already a local directory, returns it as-is.
    Otherwise treats it as a HuggingFace Hub model ID and downloads/resolves
    it to the local cache.

    Returns:
        Absolute path to a local directory containing the tokenizer files.

    Raises:
        FileNotFoundError: If the path is neither a local dir nor a valid Hub ID.
    """
    if os.path.isdir(tokenizer_path):
        return tokenizer_path

    try:
        return snapshot_download(tokenizer_path)
    except (HFValidationError, RepositoryNotFoundError, OSError) as e:
        raise FileNotFoundError(
            f"'{tokenizer_path}' is not a local directory and could not be "
            f"resolved as a HuggingFace Hub model ID: {e}"
        ) from e


# ── Token manipulation ───────────────────────────────────────────────────────


def rename_reserved_tokens(
    save_path: str, tokenizer, renames: dict[str, str]
) -> None:
    """Rename tokens in the saved tokenizer files on disk, in one pass.

    Replaces each quoted old token in tokenizer.json and each exactly-matching
    string value in tokenizer_config.json; ids never move.
    Old tokens missing from the vocabulary are skipped.

    Used to turn placeholders like <|RESERVED_OMNI_001|> into real names
    like <|img_start|>.

    Note: modifies files on disk, not the in-memory tokenizer object.
    Reload from disk after renaming to get the updated vocabulary.
    """
    if set(renames) & set(renames.values()):
        raise ValueError("rename sources and targets overlap")
    vocab = tokenizer.get_vocab()
    present = {}
    for old, new in renames.items():
        if old in vocab:
            present[old] = new
        else:
            print(f"  {old} not found, skipping rename to {new}")
    if not present:
        return

    tokenizer_json_path = os.path.join(save_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            content = f.read()
        for old, new in present.items():
            content = content.replace(f'"{old}"', f'"{new}"')
        with open(tokenizer_json_path, "w", encoding="utf-8") as f:
            f.write(content)

    config_path = os.path.join(save_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        def _replace(obj):
            if isinstance(obj, dict):
                return {k: _replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_replace(item) for item in obj]
            elif isinstance(obj, str):
                return present.get(obj, obj)
            return obj

        config = _replace(config)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    for old, new in present.items():
        print(f"  Renamed {old} -> {new} (ID {vocab[old]})")


def add_token_alias(
    save_path: str,
    token: str,
    alias: str,
    tokenizer: AutoTokenizer | None = None,
    save: bool = True,
) -> None:
    """Make ``alias`` encode to the same token ID as ``token``.

    Prepends a Replace(alias, token) rule to the tokenizer's normalizer chain,
    and switches the target token to normalized=True in the same rebuild:
    the rewritten alias only exists after normalization, so the target must match normalized input.
    Raises ValueError if ``token`` is not an added token.

    The normalizer is rebuilt as a single flat Sequence from the tokenizer's serialized state;
    nested Sequences lose their child rules on some tokenizers versions.

    This eliminates the need for manual .replace("<image>", "<|image|>")
    calls in data loaders and conversation transforms.

    If ``tokenizer`` is provided, modifies it in-memory instead of
    loading from ``save_path``.  Set ``save=False`` to skip the
    save_pretrained call (useful when batching multiple aliases).

    Example::

        add_token_alias(path, "<|image|>", "<image>")
        # Now tokenizer.encode("<image>") == tokenizer.encode("<|image|>")
    """
    if tokenizer is None and not save:
        raise ValueError("save=False without tokenizer= would discard the alias")
    tok = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(save_path)
    state = json.loads(tok.backend_tokenizer.to_str())

    target = next((t for t in state.get("added_tokens", []) if t["content"] == token), None)
    if target is None:
        raise ValueError(f"Alias target {token!r} is not an added token")
    target["normalized"] = True

    existing = state.get("normalizer")
    if existing is None:
        chain = []
    elif existing["type"] == "Sequence":
        chain = existing["normalizers"]
    else:
        chain = [existing]
    rule = {"type": "Replace", "pattern": {"String": alias}, "content": token}
    if rule not in chain:
        chain.insert(0, rule)
    state["normalizer"] = {"type": "Sequence", "normalizers": chain}

    # transformers exposes no setter for backend_tokenizer
    tok._tokenizer = Tokenizer.from_str(json.dumps(state))
    if save:
        tok.save_pretrained(save_path)
    print(f"  Added alias {alias} -> {token}")


# ── Tokenizer config ─────────────────────────────────────────────────────────


def write_tokenizer_config(
    save_path: str,
    tokenizer,
    base_vocab_size: int,
    *,
    extra_config: dict[str, Any] | None = None,
    config_section_name: str | None = None,
    omnimodal_config: dict[str, Any] | None = None,
) -> None:
    """Write derived/custom fields into ``tokenizer_config.json``.

    Hugging Face's ``save_pretrained()`` cannot faithfully persist this
    project's custom metadata on its own:

    - ``vocab_size`` gets overwritten with the base-model vocab size
      because ``tokenizer.vocab_size`` excludes added tokens.
    - ``base_vocab_size`` and ``added_tokens_count`` are project-specific.
    - ``omnimodal_config`` is derived metadata the tokenizer object
      itself does not carry.

    This helper centralizes those post-save corrections in one place.
    """
    config_path = os.path.join(save_path, "tokenizer_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    actual_vocab_size = len(tokenizer.get_vocab())
    config["vocab_size"] = actual_vocab_size
    config["base_vocab_size"] = config.get("base_vocab_size", base_vocab_size)
    config["added_tokens_count"] = actual_vocab_size - base_vocab_size

    if extra_config and config_section_name:
        config[config_section_name] = extra_config

    if omnimodal_config is not None:
        if omnimodal_config:
            config["omnimodal_config"] = omnimodal_config
        else:
            config.pop("omnimodal_config", None)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_tokenizer(
    tokenizer,
    save_path: str,
    base_vocab_size: int,
    extra_config: dict[str, Any] | None = None,
    config_section_name: str | None = None,
) -> None:
    """Save tokenizer to disk and write base metadata to tokenizer_config.json.

    Writes to tokenizer_config.json:
        vocab_size:         total size (text + reserved + content)
        base_vocab_size:    original text-only size (set once, never overwritten)
        added_tokens_count: vocab_size - base_vocab_size

    If extra_config and config_section_name are provided, also writes
    extra_config under that key (e.g. vision_tokenizer: {type: "Emu3.5"}).
    """
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    write_tokenizer_config(
        save_path,
        tokenizer,
        base_vocab_size,
        extra_config=extra_config,
        config_section_name=config_section_name,
    )


# ── Modality detection ────────────────────────────────────────────────────────


def _read_tokenizer_config(local_path: str) -> dict[str, Any]:
    config_path = os.path.join(local_path, "tokenizer_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_existing_modalities(tokenizer_path: str) -> dict[str, Any]:
    """Detect which modalities already exist in a tokenizer directory.

    Accepts both local paths and HuggingFace Hub model IDs (e.g.
    ``"swiss-ai/Apertus-8B-2509"``).  Hub IDs are resolved to the local
    cache automatically.

    Reads base_vocab_size and omnimodal_config from tokenizer_config.json.

    Returns::

        {
            "base_vocab_size": 131072,
            "modalities": {
                "vision": {"vocab_size": 131072},
                "audio":  {"vocab_size": 4096},
            }
        }

    Used by builder.add_modality() for idempotency checks and stacking.
    """
    config = _read_tokenizer_config(_resolve_tokenizer_path(tokenizer_path))
    result: dict[str, Any] = {
        "base_vocab_size": config.get("base_vocab_size"),
        "modalities": {},
    }
    for entry in config.get("omnimodal_config", {}).get("modalities", []):
        if (name := entry.get("name")) is not None:
            result["modalities"][name] = {"vocab_size": entry.get("vocab_size")}
    return result


# ── Omnimodal config ──────────────────────────────────────────────────────────


def read_modality_info(mc: ModalityConfig, vocab: dict[str, int]) -> dict[str, Any] | None:
    """Derive a single modality's summary from the tokenizer vocabulary.

    Returns {name, offset, vocab_size, start_token, end_token} or None
    if the modality's content or structure tokens are not in the vocabulary.
    Walks the content tokens from index 0, verifying id = offset + index as it counts them —
    every config this feeds is only written for contiguous ids,
    which is what lets consumers look tokens up by offset arithmetic.

    Raises:
        ValueError: If the content ids have a gap — a renumbered or missing
            token (corrupted or hand-edited tokenizer).
    """
    offset = vocab.get(mc.content_token_format.format(i=0))
    start_id = vocab.get(mc.start_token)
    end_id = vocab.get(mc.end_token)
    if offset is None or start_id is None or end_id is None:
        return None

    count = 0
    while (token_id := vocab.get(mc.content_token_format.format(i=count))) is not None:
        if token_id != offset + count:
            raise ValueError(
                f"{mc.content_token_format.format(i=count)} has id {token_id}, "
                f"expected {offset + count}: {mc.name} content ids are not contiguous"
            )
        count += 1
    prefix, suffix = mc.content_token_format.split("{i}")
    total = sum(1 for t in vocab if t.startswith(prefix) and t.endswith(suffix))
    if total != count:
        raise ValueError(
            f"{mc.name} has {total} content tokens in the vocab but only "
            f"{count} are contiguous from index 0"
        )

    return {
        "name": mc.name,
        "offset": offset,
        "vocab_size": count,
        "start_token": start_id,
        "end_token": end_id,
        "structure_token_ids": {
            r.target_name: vocab[r.target_name]
            for r in mc.structure_tokens
            if r.target_name in vocab
        },
    }


def build_omnimodal_config(
    base_vocab_size: int,
    tokenizer,
    registry: dict[str, ModalityConfig] | None = None,
) -> dict[str, Any]:
    """Build omnimodal_config for every registered modality present in the tokenizer.

    Assembles the summaries sorted by content token offset::

        {
            "omni_special_token_offset": 131072,
            "modalities": [
                {"name": "vision", "offset": 131272, "vocab_size": 131072, ...},
                {"name": "audio",  "offset": 262344, "vocab_size": 4096,   ...},
            ]
        }

    Returns empty dict if no modalities are detected.
    """
    if registry is None:
        registry = MODALITY_REGISTRY

    vocab = tokenizer.get_vocab()
    modalities = [
        info
        for mc in registry.values()
        if (info := read_modality_info(mc, vocab)) is not None
    ]

    if not modalities:
        return {}

    modalities.sort(key=lambda m: m["offset"])
    return {
        "omni_special_token_offset": base_vocab_size,
        "modalities": modalities,
    }


# ── Mapping utilities ─────────────────────────────────────────────────────────


def load_modality_mapping(tokenizer_path: str, modality_name: str) -> dict:
    """Return a modality's entry from the tokenizer's omnimodal_config.

    Accepts both local paths and HuggingFace Hub model IDs.

    Returns the entry dict ({name, offset, vocab_size, start_token, end_token}).
    Content ids are contiguous, so id = offset + index.

    Raises:
        ValueError: If the modality is not present in the tokenizer.
    """
    config = _read_tokenizer_config(_resolve_tokenizer_path(tokenizer_path))
    entry = next(
        (
            m
            for m in config.get("omnimodal_config", {}).get("modalities", [])
            if m.get("name") == modality_name
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"{modality_name} not present in {tokenizer_path}")
    return entry


def get_content_token_id(
    index: int,
    tokenizer_path: str | None = None,
    modality_name: str = "vision",
    mapping: dict | None = None,
) -> int:
    """Convert a codebook index to its token ID (offset + index).

    Pass tokenizer_path for one-off lookups, or pass a pre-loaded mapping
    dict for batch lookups to avoid repeated file reads::

        # One-off
        tid = get_content_token_id(42, tokenizer_path="/path/to/tokenizer")

        # Batch
        mapping = load_modality_mapping("/path/to/tokenizer", "vision")
        tids = [get_content_token_id(i, mapping=mapping) for i in indices]

    Raises:
        ValueError: If the index is out of the codebook range.
    """
    if mapping is None:
        mapping = load_modality_mapping(tokenizer_path, modality_name)
    vocab_size = mapping["vocab_size"]
    if not 0 <= index < vocab_size:
        raise ValueError(
            f"{modality_name} index {index} not found. "
            f"Valid range: 0-{vocab_size - 1}"
        )
    return mapping["offset"] + index
