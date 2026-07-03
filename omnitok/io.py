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
    copy_modality_mapping_files() # preserve existing mapping JSONs when stacking
    rename_reserved_token()       # e.g. <|RESERVED_OMNI_001|> -> <|img_start|>
    add_token_alias()             # e.g. <image> encodes to same ID as <|image|>
    build_omnimodal_config()      # derive omnimodal metadata from mapping files
    write_tokenizer_config()      # write all custom tokenizer_config.json fields
"""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
from tokenizers import normalizers
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


def rename_reserved_token(
    save_path: str, tokenizer, old_token: str, new_token: str
) -> None:
    """Rename a token in the saved tokenizer files on disk.

    String-replaces all occurrences of old_token with new_token in both
    tokenizer.json and tokenizer_config.json. Skips if old_token is not
    in the vocabulary.

    Used to turn placeholders like <|RESERVED_OMNI_001|> into real names
    like <|img_start|>.

    Note: modifies files on disk, not the in-memory tokenizer object.
    Reload from disk after all renames to get the updated vocabulary.
    """
    token_id = tokenizer.convert_tokens_to_ids(old_token)
    if token_id == tokenizer.unk_token_id:
        print(f"  {old_token} not found, skipping rename to {new_token}")
        return

    tokenizer_json_path = os.path.join(save_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(f'"{old_token}"', f'"{new_token}"')
        content = content.replace(old_token, new_token)
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
                return obj.replace(old_token, new_token)
            return obj

        config = _replace(config)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    print(f"  Renamed {old_token} -> {new_token} (ID {token_id})")


def add_token_alias(
    save_path: str,
    token: str,
    alias: str,
    tokenizer: AutoTokenizer | None = None,
    save: bool = True,
) -> None:
    """Make ``alias`` encode to the same token ID as ``token``.

    Prepends a normalizers.Replace(alias, token) to the tokenizer's
    normalizer chain. The target token must have been created with
    AddedToken(normalized=True) so the matcher checks normalized input
    (where the alias has already been rewritten).

    This eliminates the need for manual .replace("<image>", "<|image|>")
    calls in data loaders and conversation transforms.

    If ``tokenizer`` is provided, modifies it in-memory instead of
    loading from ``save_path``.  Set ``save=False`` to skip the
    save_pretrained call (useful when batching multiple aliases).

    Example::

        add_token_alias(path, "<|image|>", "<image>")
        # Now tokenizer.encode("<image>") == tokenizer.encode("<|image|>")
    """
    tok = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(save_path)
    backend = tok.backend_tokenizer
    replace = normalizers.Replace(alias, token)
    existing = backend.normalizer
    backend.normalizer = (
        normalizers.Sequence([replace, existing]) if existing else replace
    )
    if save:
        tok.save_pretrained(save_path)
    print(f"  Added alias {alias} -> {token}")


# Reasoning-delimiter tokens across both known tokenizer schemes. The canonical
# repo build carries <|inner_prefix|>/<|inner_suffix|> at the emitted ids; some
# deployed builds register <think>/</think> there instead. We flip whichever are
# present, so this is safe to run on either scheme (apertus-omni-tokenizer #5).
REASONING_DELIMITER_TOKENS = (
    "<|inner_prefix|>",
    "<|inner_suffix|>",
    "<think>",
    "</think>",
)


def mark_tokens_non_special(
    save_path: str, tokens: tuple[str, ...] = REASONING_DELIMITER_TOKENS
) -> list[str]:
    """Flip ``special`` to ``false`` for ``tokens`` in the saved tokenizer files.

    A vLLM reasoning parser locates the end-of-reasoning delimiter in the
    *detokenized* string. When the delimiter is a **special** token, the default
    ``skip_special_tokens=True`` strips it before the parser runs, so the whole
    deliberation block leaks into ``content`` and the ``reasoning`` channel stays
    empty (apertus-omni-tokenizer #5). Registering the delimiters as non-special
    keeps them in the decoded output for every client -- no ``skip_special_tokens``
    override required -- while their ids (hence the streaming parser and any
    encoding of the literal string) are unchanged.

    Edits ``tokenizer.json`` (``added_tokens``) and, when present,
    ``tokenizer_config.json`` (``added_tokens_decoder``), and drops the tokens
    from ``additional_special_tokens`` in ``special_tokens_map.json``. Only tokens
    actually present are touched. Returns the list of tokens that were flipped.

    Note: modifies files on disk, not any in-memory tokenizer object.
    """
    targets = set(tokens)
    flipped: set[str] = set()

    # tokenizer.json is up to tens of MB; a full json round-trip would reformat
    # the whole file (and risk serializer drift). Flip the one boolean in place
    # with a surgical text substitution that leaves every other byte untouched.
    # The same added-token object shape ({"content": TOK, ..., "special": true})
    # appears in tokenizer.json's `added_tokens` list and in
    # tokenizer_config.json's `added_tokens_decoder` map, so one regex covers
    # both. Added-token objects contain no nested braces, so [^{}] stays inside
    # the object and reaches only that token's own `special` flag.
    def _flip_in_text(path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        changed = 0
        for tok in targets:
            pattern = re.compile(
                r'("content":\s*"' + re.escape(tok) + r'"[^{}]*?"special":\s*)true',
                re.DOTALL,
            )
            text, n = pattern.subn(r"\1false", text)
            if n:
                flipped.add(tok)
                changed += n
        # Avoid a multi-MB no-op rewrite (and mtime churn) when nothing changed.
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

    _flip_in_text(os.path.join(save_path, "tokenizer.json"))
    _flip_in_text(os.path.join(save_path, "tokenizer_config.json"))

    stm_path = os.path.join(save_path, "special_tokens_map.json")
    if os.path.exists(stm_path):
        with open(stm_path, "r", encoding="utf-8") as f:
            stm = json.load(f)
        ast = stm.get("additional_special_tokens")
        if isinstance(ast, list):
            def _content(t):
                return t.get("content") if isinstance(t, dict) else t

            removed = {_content(t) for t in ast if _content(t) in targets}
            if removed:
                stm["additional_special_tokens"] = [
                    t for t in ast if _content(t) not in targets
                ]
                flipped.update(removed)  # only tokens actually present/removed
                with open(stm_path, "w", encoding="utf-8") as f:
                    json.dump(stm, f, ensure_ascii=False, indent=2)

    for tok in sorted(flipped):
        print(f"  Marked {tok} non-special")
    if not flipped:
        print("  No reasoning delimiters found to mark non-special")
    return sorted(flipped)


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
    - ``omnimodal_config`` is derived from mapping JSONs written outside
      the tokenizer object itself.

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


def detect_existing_modalities(
    tokenizer_path: str,
    registry: dict[str, ModalityConfig] | None = None,
) -> dict[str, Any]:
    """Detect which modalities already exist in a tokenizer directory.

    Accepts both local paths and HuggingFace Hub model IDs (e.g.
    ``"swiss-ai/Apertus-8B-2509"``).  Hub IDs are resolved to the local
    cache automatically.

    Checks for modality mapping files (e.g. vision_token_mapping.json)
    and reads base_vocab_size from tokenizer_config.json.

    Returns::

        {
            "base_vocab_size": 131072,
            "modalities": {
                "vision": {"mapping_file": "...", "vocab_size": 131072},
                "audio":  {"mapping_file": "...", "vocab_size": 4096},
            }
        }

    Used by builder.add_modality() for idempotency checks and stacking.
    """
    if registry is None:
        registry = MODALITY_REGISTRY

    local_path = _resolve_tokenizer_path(tokenizer_path)
    result: dict[str, Any] = {"base_vocab_size": None, "modalities": {}}

    config_path = os.path.join(local_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        result["base_vocab_size"] = config.get("base_vocab_size")

    for name, mc in registry.items():
        mapping_path = os.path.join(local_path, mc.mapping_file)
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result["modalities"][name] = {
                "mapping_file": mc.mapping_file,
                "vocab_size": data.get(mc.vocab_size_key),
            }

    return result


def copy_modality_mapping_files(
    existing_modalities: dict[str, Any],
    input_path: str,
    output_path: str,
) -> None:
    """Copy existing modality mapping files from input to output.

    When stacking (e.g. adding audio on top of vision), the vision
    mapping file must be preserved in the output directory.

    Accepts Hub model IDs as ``input_path`` — resolved to local cache
    before copying.
    """
    local_input = _resolve_tokenizer_path(input_path)
    modalities = existing_modalities.get("modalities") or {}
    for info in modalities.values():
        src = os.path.join(local_input, info["mapping_file"])
        dst = os.path.join(output_path, info["mapping_file"])
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)
            print(f"  Copied {info['mapping_file']}")


# ── Omnimodal config ──────────────────────────────────────────────────────────


def read_modality_info(
    output_path: str, mc: ModalityConfig, tokenizer
) -> dict[str, Any] | None:
    """Read a single modality's summary from its mapping file.

    Returns {name, offset, vocab_size, start_token, end_token} or None
    if the mapping file is missing or the structure tokens are not in
    the vocabulary yet.
    """
    mapping_path = os.path.join(output_path, mc.mapping_file)
    if not os.path.exists(mapping_path):
        return None

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if mc.offset_key not in data:
        return None

    start_id = tokenizer.convert_tokens_to_ids(mc.start_token)
    end_id = tokenizer.convert_tokens_to_ids(mc.end_token)
    if start_id == tokenizer.unk_token_id or end_id == tokenizer.unk_token_id:
        return None

    return {
        "name": mc.name,
        "offset": data[mc.offset_key],
        "vocab_size": data.get(mc.vocab_size_key),
        "start_token": start_id,
        "end_token": end_id,
    }


def build_omnimodal_config(
    output_path: str,
    base_vocab_size: int,
    tokenizer,
    registry: dict[str, ModalityConfig] | None = None,
) -> dict[str, Any]:
    """Build omnimodal_config from all present modality mapping files.

    Scans for each registered modality's mapping file, reads its summary,
    and assembles them sorted by content token offset::

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

    modalities = [
        info
        for mc in registry.values()
        if (info := read_modality_info(output_path, mc, tokenizer)) is not None
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
    """Load a modality's token mapping JSON (e.g. vision_token_mapping.json).

    Accepts both local paths and HuggingFace Hub model IDs.

    Returns the full mapping dict containing token IDs, offset, vocab size.
    Used by downstream code to convert codebook indices to token IDs.

    Raises:
        ValueError: If modality_name is not in the registry.
        FileNotFoundError: If the mapping file does not exist.
    """
    if modality_name not in MODALITY_REGISTRY:
        raise ValueError(f"Unknown modality: {modality_name}")
    mc = MODALITY_REGISTRY[modality_name]
    local_path = _resolve_tokenizer_path(tokenizer_path)
    mapping_path = os.path.join(local_path, mc.mapping_file)
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")
    with open(mapping_path, "r") as f:
        return json.load(f)


def get_content_token_id(
    index: int,
    tokenizer_path: str | None = None,
    modality_name: str = "vision",
    mapping: dict | None = None,
) -> int:
    """Convert a codebook index to its token ID.

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
    mc = MODALITY_REGISTRY[modality_name]
    token_ids = mapping.get(f"{mc.name}_token_ids", {})
    key = str(index)
    if key in token_ids:
        return token_ids[key]
    raise ValueError(
        f"{modality_name} index {index} not found. "
        f"Valid range: 0-{mapping.get(mc.vocab_size_key, '?')}"
    )
