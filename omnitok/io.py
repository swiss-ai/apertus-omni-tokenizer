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

Call flow (driven by builder._assemble, shared by both entry points):

    save_tokenizer()              # save HF tokenizer + write base metadata
    rename_reserved_tokens()      # e.g. <|RESERVED_OMNI_001|> -> <|img_start|>
    add_token_alias()             # e.g. <image> encodes to same ID as <|image|>
    build_omnimodal_config()      # derive omnimodal metadata from the tokenizer
    write_tokenizer_config()      # write all custom tokenizer_config.json fields
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Sequence

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .modalities import MODALITY_REGISTRY, ModalityConfig


def _resolve_tokenizer_path(tokenizer_path: str, revision: str | None = None) -> str:
    """Resolve a tokenizer path to a local directory.

    If ``tokenizer_path`` is already a local directory, returns it as-is.
    Otherwise treats it as a HuggingFace Hub model ID and downloads/resolves
    it to the local cache. ``revision`` pins the Hub commit to fetch; Hub
    repos are mutable, so reproducible builds should always pin one. It is
    ignored for local directories.

    Returns:
        Absolute path to a local directory containing the tokenizer files.

    Raises:
        FileNotFoundError: If the path is neither a local dir nor a valid Hub ID.
    """
    if os.path.isdir(tokenizer_path):
        return tokenizer_path

    try:
        return snapshot_download(
            tokenizer_path,
            revision=revision,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"],
        )
    except (HFValidationError, RepositoryNotFoundError, OSError) as e:
        raise FileNotFoundError(
            f"'{tokenizer_path}' is not a local directory and could not be "
            f"resolved as a HuggingFace Hub model ID: {e}"
        ) from e


# ── Token manipulation ───────────────────────────────────────────────────────


def _rewrite_backend_state(
    save_path: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    """Round-trip tokenizer.json through the backend, applying ``mutate``.

    The state dict is parsed back by the backend on save, so a malformed
    edit raises instead of writing a corrupt file, and the serialization
    stays identical to what the backend itself writes.
    """
    path = os.path.join(save_path, "tokenizer.json")
    state = json.loads(Tokenizer.from_file(path).to_str())
    mutate(state)
    Tokenizer.from_str(json.dumps(state)).save(path, pretty=True)


def assert_droppable_post_processor(state: dict[str, Any], declared: set[str]) -> None:
    """Refuse post-processor shapes that are not safe to drop.

    Only a TemplateProcessing over the tokenizer's own declared bos/eos is;
    anything else encodes behaviour the caller did not ask to lose.
    """
    if state is None:
        return
    if state.get("type") != "TemplateProcessing" or not (
        set(state.get("special_tokens", {})) <= declared
    ):
        raise ValueError(f"unrecognized post-processor: {state.get('type')}")


def rename_reserved_tokens(
    save_path: str, tokenizer, renames: dict[str, str]
) -> None:
    """Rename tokens in the saved tokenizer files on disk; ids never move.

    tokenizer.json is rewritten structurally (vocab keys and added-token
    contents); the tokenizer_config.json and special_tokens_map.json mirrors
    swap exactly-equal string values. Old tokens missing from the vocabulary
    are skipped.

    Used to turn placeholders like <|RESERVED_OMNI_001|> into real names
    like <|img_start|>.

    Note: modifies files on disk, not the in-memory tokenizer object.
    Reload from disk after renaming to get the updated vocabulary.
    """
    if set(renames) & set(renames.values()):
        raise ValueError("rename sources and targets overlap")
    targets = list(renames.values())
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate rename targets")
    present = {}
    token_ids = {}
    for old, new in renames.items():
        token_id = tokenizer.backend_tokenizer.token_to_id(old)
        if token_id is None:
            print(f"  {old} not found, skipping rename to {new}")
            continue
        if tokenizer.backend_tokenizer.token_to_id(new) is not None:
            raise ValueError(f"rename target {new} already in the vocabulary")
        present[old] = new
        token_ids[old] = token_id
    if not present:
        return

    def _rename(state):
        vocab = state["model"]["vocab"]
        for old, new in present.items():
            if old in vocab:
                vocab[new] = vocab.pop(old)
        for entry in state["added_tokens"]:
            if entry["content"] in present:
                entry["content"] = present[entry["content"]]

    _rewrite_backend_state(save_path, _rename)

    def _replace(obj):
        if isinstance(obj, dict):
            return {k: _replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_replace(item) for item in obj]
        elif isinstance(obj, str):
            return present.get(obj, obj)
        return obj

    for fname in ("tokenizer_config.json", "special_tokens_map.json"):
        mirror_path = os.path.join(save_path, fname)
        if os.path.exists(mirror_path):
            with open(mirror_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(mirror_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(_replace(data), indent=2))

    for old, new in present.items():
        print(f"  Renamed {old} -> {new} (ID {token_ids[old]})")


def _flat_normalizer_chain(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the tokenizer state's normalizer as a flat list of rules.

    The normalizer is rebuilt as a single flat Sequence from the tokenizer's
    serialized state; nested Sequences lose their child rules on some
    tokenizers versions.
    """
    existing = state.get("normalizer")
    if existing is None:
        return []
    if existing["type"] == "Sequence":
        return existing["normalizers"]
    return [existing]


def _prepend_rules(state: dict[str, Any], rules: Sequence[dict[str, Any]]) -> None:
    """Insert ``rules`` at the front of the state's normalizer chain, first
    rule first; rules already present are not duplicated."""
    chain = _flat_normalizer_chain(state)
    for rule in reversed(rules):
        if rule not in chain:
            chain.insert(0, rule)
    state["normalizer"] = {"type": "Sequence", "normalizers": chain}


def prepend_normalizer_rules(
    save_path: str, rules: Sequence[dict[str, Any]]
) -> None:
    """Insert normalizer ``rules`` at the front of the chain, first rule first.

    Unlike :func:`add_token_alias`, this takes raw serialized rule dicts
    (e.g. ``{"type": "Replace", "pattern": {"Regex": ...}, "content": ...}``),
    so it can express Regex patterns and deletions whose replacement is not an
    added token. Rules already present are not duplicated. Operates directly
    on ``tokenizer.json``.
    """
    _rewrite_backend_state(save_path, lambda state: _prepend_rules(state, rules))
    print(f"  Prepended {len(rules)} normalizer rules")


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

    chain = _flat_normalizer_chain(state)
    rule = {"type": "Replace", "pattern": {"String": alias}, "content": token}
    if rule not in chain:
        chain.insert(0, rule)
    state["normalizer"] = {"type": "Sequence", "normalizers": chain}

    # transformers exposes no setter for backend_tokenizer
    tok._tokenizer = Tokenizer.from_str(json.dumps(state))
    if save:
        tok.save_pretrained(save_path)
    print(f"  Added alias {alias} -> {token}")


def mark_tokens_non_special(
    save_path: str, tokens: tuple[str, ...]
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
    present: set[str] = set()  # delimiters found in the files, at any `special`

    # tokenizer.json is up to tens of MB; a full json round-trip would reformat
    # the whole file (and risk serializer drift). Flip the one boolean in place.
    # An added-token entry is a *flat* JSON object (no nested braces) carrying
    # BOTH a "content" string and a "special" bool -- e.g.
    #   {"id": 32, "content": "<|inner_prefix|>", ..., "special": true}
    # This shape appears in tokenizer.json's `added_tokens` list and in
    # tokenizer_config.json's `added_tokens_decoder` map. Matching flat objects
    # and requiring both fields is what keeps a normalizer `Replace` rule -- which
    # also mentions the token in "content" but has a nested "pattern" object and
    # no "special" -- from being mistaken for a patchable delimiter, and makes the
    # flip independent of the order of the "content"/"special" keys.
    flat_obj = re.compile(r"\{[^{}]*\}", re.DOTALL)

    def _flip_in_text(path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        n_changed = 0

        def _patch(match):
            nonlocal n_changed
            obj = match.group(0)
            content = re.search(r'"content"\s*:\s*"([^"]*)"', obj)
            if content is None or content.group(1) not in targets:
                return obj
            if re.search(r'"special"\s*:\s*(?:true|false)', obj) is None:
                return obj  # not an added-token entry (e.g. a normalizer rule)
            tok = content.group(1)
            present.add(tok)  # present regardless of its `special` value
            new_obj, n = re.subn(r'("special"\s*:\s*)true', r"\1false", obj)
            if n:
                flipped.add(tok)
                n_changed += n
            return new_obj

        new_text = flat_obj.sub(_patch, text)
        # Avoid a multi-MB no-op rewrite (and mtime churn) when nothing changed.
        if n_changed:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_text)

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
            present.update(removed)
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
        if present:
            print("  Reasoning delimiters already non-special; nothing to do")
        else:
            print("  No reasoning delimiters found to mark non-special")
    return sorted(flipped)


# ── Tokenizer config ─────────────────────────────────────────────────────────


def dump_canonical_json(obj: dict[str, Any], path: str) -> None:
    """transformers-style deterministic JSON: sorted keys, indent 2, LF tail."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


_SPECIAL_TOKEN_ROLES = ("bos_token", "eos_token", "pad_token", "unk_token")


def finalize_tokenizer_config(
    save_path: str,
    *,
    carried_keys: Sequence[str],
    overrides: dict[str, Any],
    require_chat_template: bool = False,
) -> dict[str, Any]:
    """Rewrite the saved config into its canonical, version-independent form.

    ``save_pretrained`` emits whatever shape the running transformers prefers.
    5.x writes a TokenizersBackend class plus fossils that 4.x cannot load;
    4.x mirrors every added token into ``added_tokens_decoder``.
    An allowlist plus explicit overrides ties the output to the recipe,
    not to the build environment.

    Writes chat_template.jinja only if the built config carries a template;
    set ``require_chat_template`` where its absence means the instruct stage
    silently did not run.
    Rewrites special_tokens_map.json from whichever role tokens are present.

    Returns the config that was written.
    """
    config_path = os.path.join(save_path, "tokenizer_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        built = json.load(f)

    chat_template = built.pop("chat_template", None)
    if chat_template is None:
        if require_chat_template:
            raise ValueError("Pipeline did not produce a chat template.")
    else:
        with open(os.path.join(save_path, "chat_template.jinja"), "w",
                  encoding="utf-8") as f:
            f.write(chat_template)

    config = {key: built[key] for key in carried_keys}
    config.update(overrides)
    dump_canonical_json(config, config_path)

    dump_canonical_json(
        {
            name: {
                "content": config[name],
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
            }
            for name in _SPECIAL_TOKEN_ROLES
            if name in config
        },
        os.path.join(save_path, "special_tokens_map.json"),
    )
    return config


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

    dump_canonical_json(config, config_path)


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


def read_modality_info(
    mc: ModalityConfig,
    vocab: dict[str, int],
    *,
    publish_structure_ids: bool = False,
) -> dict[str, Any] | None:
    """Derive a single modality's summary from the tokenizer vocabulary.

    Returns {name, offset, vocab_size, start_token, end_token} or None
    if the modality's content or structure tokens are not in the vocabulary.
    ``publish_structure_ids`` adds the structure_token_ids map; the Apertus
    1.5 artifact predates that field, so it stays off unless a recipe asks.
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

    info = {
        "name": mc.name,
        "offset": offset,
        "vocab_size": count,
        "start_token": start_id,
        "end_token": end_id,
    }
    if publish_structure_ids:
        info["structure_token_ids"] = {
            r.target_name: vocab[r.target_name]
            for r in mc.structure_tokens
            if r.target_name in vocab
        }
    return info


def build_omnimodal_config(
    base_vocab_size: int,
    tokenizer,
    registry: dict[str, ModalityConfig] | None = None,
    *,
    publish_structure_ids: bool = False,
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
        if (info := read_modality_info(
            mc, vocab, publish_structure_ids=publish_structure_ids
        )) is not None
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
