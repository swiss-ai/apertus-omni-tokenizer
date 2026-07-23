"""Build recipe for the Apertus 1.5 tokenizer.

Builds the Apertus 1.5 tokenizer from the Apertus 1 instruct tokenizer
(swiss-ai/Apertus-8B-Instruct-2509). The output matches the canonical
artifact (apertus-ai/Apertus-v1.5-8B-RC) except for the deliberate fixes
listed below. The recipe is the audit trail for every delta between the two:

1. Text stage (`prepare_apertus_1p5_text_base`):
   - ``<think>``/``</think>`` (ids 32/33) swap names with
     ``<|inner_prefix|>``/``<|inner_suffix|>`` (ids 69/70), so the reasoning
     delimiters emitted by the model live at 32/33.
   - Ids 69/70 (now ``<think>``/``</think>``) are demoted from added tokens to
     plain vocab entries: the literals ``<think>``/``</think>`` are aliased to
     32/33 via the normalizer, so 69/70 must not be matched as added tokens.
   - ``<SPECIAL_73>``/``<SPECIAL_74>`` become
     ``<|tool_output_start|>``/``<|tool_output_end|>``.
   - Ids 32/33 are marked non-special so the delimiters survive detokenization
     under skip_special_tokens=True (apertus-omni-tokenizer #5).
2. Vision modality: 200 RESERVED_OMNI slots + 131,072 Emu3.5 content tokens.
3. Audio modality: 4,096 WavTokenizer content tokens.
4. Instruct stage: the Apertus 1.5 chat template (maintained in this repo at
   chat_templates/Apertus_1p5/chat_template.jinja) plus SFT begin/end
   sequences.
5. Finalize (`_finalize_apertus_1p5`): prepends the reasoning-trace cleanup
   normalizer rules, applies canonical quirks and fixes (see below), and writes
   tokenizer_config.json / special_tokens_map.json / chat_template.jinja with
   a deterministic serialization so the output does not depend on which
   transformers version performed the build.

Known canonical quirks, reproduced on purpose:

- ``tokenizer_config.json`` says ``vocab_size: 131072`` — the base text vocab,
  not the true total of 266,440 (use ``len(tokenizer)`` for the total).
- ``<|stt_translate|>`` (131086) keeps the stray ``normalized: true`` flag the
  original slot-name swap left behind at audio slot 14.

Deliberate fixes over the canonical artifact (each one means shipping a new
tokenizer release, not a rebuild):

- ``<|audio|>`` (131085) is ``normalized: true``, so the ``<audio>`` alias
  rule in the normalizer resolves to the special token — matching how the
  ``<image>`` alias already worked. The canonical artifact had
  ``normalized: false`` (the slot-name swap moved the ``<|audio|>`` name to
  slot 13 without its alias-ready flag), which made ``<audio>`` encode as
  plain text.
- ``backend: "tokenizers"`` / ``is_local: true`` are dropped from
  tokenizer_config.json: fossils of the environment that produced the
  original artifact, recomputed at load time anyway.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .builder import add_modality
from .instruct import create_instruct_tokenizer
from .io import (
    _flat_normalizer_chain,
    _resolve_tokenizer_path,
    add_token_alias,
    mark_tokens_non_special,
    prepend_normalizer_rules,
)

BASE_REPO = "swiss-ai/Apertus-8B-Instruct-2509"
# Hub repos are mutable (this one had <SPECIAL_73> renamed to <|image|> and
# reverted after release), so the base is pinned to the post-revert commit.
BASE_REVISION = "b946d40447b2b597999b9c86d44bee0b452c919f"

BASE_VOCAB_SIZE = 131072
VISION_VOCAB_SIZE = 131072
AUDIO_VOCAB_SIZE = 4096
FINAL_VOCAB_SIZE = 266440

VISION_EXTRA_CONFIG = {
    "codebook_size": VISION_VOCAB_SIZE,
    "path": "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer",
    "type": "Emu3.5",
}
AUDIO_EXTRA_CONFIG = {
    "codebook_size": AUDIO_VOCAB_SIZE,
    "type": "wavtokenizer",
}

# Base-vocab renames, keyed by token id: id -> (expected old name, new name).
# The old name is asserted before renaming so a rebuild on the wrong base
# (e.g. a Hub snapshot where these slots were already renamed) fails loudly
# instead of producing a differently-shaped artifact.
TEXT_RENAMES: dict[int, tuple[str, str]] = {
    32: ("<think>", "<|inner_prefix|>"),
    33: ("</think>", "<|inner_suffix|>"),
    69: ("<|inner_prefix|>", "<think>"),
    70: ("<|inner_suffix|>", "</think>"),
    73: ("<SPECIAL_73>", "<|tool_output_start|>"),
    74: ("<SPECIAL_74>", "<|tool_output_end|>"),
}

# Dropped from added_tokens (they stay in the plain vocab): with the
# <think>/</think> literals aliased to 32/33, the added-token matcher must not
# grab them at 69/70 first.
DEMOTED_TOKEN_IDS = (69, 70)

# Reasoning-trace cleanup rules, prepended (in this order) in front of the
# alias rules during finalization. They rewrite artifacts of earlier reasoning
# formats (<thought>, <answer>, <|channel|>) into the 1.5 delimiters before
# tokenization.
REASONING_CLEANUP_RULES: tuple[dict[str, Any], ...] = (
    {"type": "Replace", "pattern": {"Regex": r"<\|channel\|?>thought\s*\n"}, "content": "<|inner_prefix|>"},
    {"type": "Replace", "pattern": {"String": "<channel|>"}, "content": "<|inner_suffix|>"},
    {"type": "Replace", "pattern": {"String": "<thought>"}, "content": "<|inner_prefix|>"},
    {"type": "Replace", "pattern": {"String": "</thought>"}, "content": "<|inner_suffix|>"},
    {"type": "Replace", "pattern": {"String": "</answer>"}, "content": ""},
    {"type": "Replace", "pattern": {"String": "<answer>"}, "content": ""},
    {"type": "Replace", "pattern": {"Regex": r"<\|inner_suffix\|>\s+"}, "content": "<|inner_suffix|>"},
)

# The exact key set of the canonical tokenizer_config.json; the finalizer
# asserts it so a pipeline change cannot silently alter the artifact schema.
CANONICAL_CONFIG_KEYS = frozenset({
    "add_prefix_space", "added_tokens_count", "added_tokens_decoder",
    "audio_begin_token", "audio_end_token", "audio_tokenizer",
    "base_vocab_size", "bos_token", "clean_up_tokenization_spaces",
    "eos_token", "extra_special_tokens", "model_input_names",
    "model_max_length", "omnimodal_config", "pad_token", "padding_side",
    "sft_assistant_begin_sequence", "sft_eot_token",
    "sft_user_begin_sequence", "tokenizer_class", "unk_token",
    "vision_begin_token", "vision_end_token", "vision_tokenizer",
    "vocab_size",
})

# Post-build encode pins: literal -> single expected id.
_VERIFY_ENCODINGS = {
    "<think>": 32,
    "</think>": 33,
    "<|inner_prefix|>": 32,
    "<|inner_suffix|>": 33,
    "<|tool_output_start|>": 73,
    "<|tool_output_end|>": 74,
    "<|RESERVED_OMNI_000|>": 131072,
    "<|img_start|>": 131073,
    "<|image|>": 131079,
    "<image>": 131079,
    "<|audio_start|>": 131080,
    "<|audio|>": 131085,
    "<audio>": 131085,
    "<|visual token 0|>": 131272,
    "<|audio token 0|>": 262344,
}


def _default_chat_template_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        repo_root, "chat_templates", "Apertus_1p5", "chat_template.jinja"
    )


def _load_backend_state(save_path: str) -> dict[str, Any]:
    path = os.path.join(save_path, "tokenizer.json")
    return json.loads(Tokenizer.from_file(path).to_str())


def _save_backend_state(state: dict[str, Any], save_path: str) -> None:
    # Round-tripping through the tokenizers backend keeps the serialization
    # identical to what save_pretrained() writes.
    path = os.path.join(save_path, "tokenizer.json")
    Tokenizer.from_str(json.dumps(state)).save(path, pretty=True)


def _dump_canonical_json(obj: dict[str, Any], path: str) -> None:
    """transformers-style deterministic JSON: sorted keys, indent 2, LF tail."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def prepare_apertus_1p5_text_base(
    base_tokenizer_path: str,
    output_path: str,
    *,
    revision: str | None = None,
) -> str:
    """Turn the Apertus 1 instruct tokenizer into the 1.5 text tokenizer.

    Applies the base-vocab renames, demotes ids 69/70 to plain vocab entries,
    aliases the <think>/</think> literals to the reasoning delimiters at
    32/33, and marks those delimiters non-special. No omni tokens yet.
    """
    print("=" * 60)
    print("APERTUS 1.5 TEXT BASE")
    print("=" * 60)
    src = _resolve_tokenizer_path(base_tokenizer_path, revision=revision)
    print(f"Base: {src}")

    os.makedirs(output_path, exist_ok=True)
    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        fpath = os.path.join(src, fname)
        if os.path.exists(fpath):
            shutil.copyfile(fpath, os.path.join(output_path, fname))

    state = _load_backend_state(output_path)

    vocab = state["model"]["vocab"]
    for token_id, (old, _) in TEXT_RENAMES.items():
        if vocab.get(old) != token_id:
            raise ValueError(
                f"Expected {old!r} at id {token_id} in the base vocab, found "
                f"id {vocab.get(old)!r}. {base_tokenizer_path} is not the "
                f"expected Apertus 1 base ({BASE_REPO} @ {BASE_REVISION})."
            )
    # Two phases: several renames swap names between ids, so setting new names
    # while old ones are still present would clobber entries.
    for _, (old, _) in TEXT_RENAMES.items():
        del vocab[old]
    for token_id, (_, new) in TEXT_RENAMES.items():
        vocab[new] = token_id

    added_by_id = {t["id"]: t for t in state["added_tokens"]}
    for token_id, (old, new) in TEXT_RENAMES.items():
        entry = added_by_id.get(token_id)
        if entry is not None and entry["content"] == old:
            entry["content"] = new
    state["added_tokens"] = [
        t for t in state["added_tokens"] if t["id"] not in DEMOTED_TOKEN_IDS
    ]
    _save_backend_state(state, output_path)
    for token_id, (old, new) in TEXT_RENAMES.items():
        print(f"  Renamed id {token_id}: {old} -> {new}")
    print(f"  Demoted ids {DEMOTED_TOKEN_IDS} from added tokens")

    # Keep tokenizer_config.json's added_tokens_decoder consistent, else the
    # next load would resurrect the demoted/renamed entries from it.
    config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    decoder = config.get("added_tokens_decoder")
    if decoder:
        for token_id, (old, new) in TEXT_RENAMES.items():
            entry = decoder.get(str(token_id))
            if entry is not None and entry.get("content") == old:
                entry["content"] = new
        for token_id in DEMOTED_TOKEN_IDS:
            decoder.pop(str(token_id), None)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    # Alias the old literals to the new delimiters. Insertion order gives the
    # canonical chain [<think> -> 32, </think> -> 33].
    tokenizer = AutoTokenizer.from_pretrained(output_path)
    add_token_alias(output_path, "<|inner_suffix|>", "</think>",
                    tokenizer=tokenizer, save=False)
    add_token_alias(output_path, "<|inner_prefix|>", "<think>",
                    tokenizer=tokenizer, save=True)

    mark_tokens_non_special(output_path)
    return output_path


def _finalize_apertus_1p5(output_path: str) -> None:
    """Apply canonical quirks and write the derived files deterministically."""
    state = _load_backend_state(output_path)

    # The original artifact swapped the names of audio slots 13/14 by string
    # replacement, leaving a stray normalized=true flag at slot 14
    # (<|stt_translate|>), reproduced here. Slot 13 (<|audio|>) keeps the
    # builder's alias-ready normalized=true — the canonical artifact lost it
    # in that swap, which broke the <audio> alias (see module docstring).
    for entry in state["added_tokens"]:
        if entry["content"] == "<|stt_translate|>":
            entry["normalized"] = True
    _save_backend_state(state, output_path)
    prepend_normalizer_rules(output_path, REASONING_CLEANUP_RULES)
    state = _load_backend_state(output_path)

    config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        built = json.load(f)

    chat_template = built.pop("chat_template", None)
    if chat_template is None:
        raise ValueError("Pipeline did not produce a chat template.")
    with open(os.path.join(output_path, "chat_template.jinja"), "w",
              encoding="utf-8") as f:
        f.write(chat_template)

    carried_keys = (
        "add_prefix_space", "added_tokens_count", "audio_begin_token",
        "audio_end_token", "audio_tokenizer", "base_vocab_size", "bos_token",
        "clean_up_tokenization_spaces", "eos_token", "model_input_names",
        "model_max_length", "omnimodal_config", "pad_token", "padding_side",
        "sft_assistant_begin_sequence", "sft_eot_token",
        "sft_user_begin_sequence", "unk_token", "vision_begin_token",
        "vision_end_token", "vision_tokenizer",
    )
    config = {key: built[key] for key in carried_keys}
    config.update({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "extra_special_tokens": {},
        # Canonical quirk: the base text vocab size, not the true total.
        "vocab_size": BASE_VOCAB_SIZE,
        # Int keys on purpose: json sorts them numerically ("2" before "10"),
        # matching how transformers serializes added_tokens_decoder.
        "added_tokens_decoder": {
            t["id"]: {
                "content": t["content"],
                "lstrip": t["lstrip"],
                "normalized": t["normalized"],
                "rstrip": t["rstrip"],
                "single_word": t["single_word"],
                "special": t["special"],
            }
            for t in state["added_tokens"]
        },
    })
    if set(config) != CANONICAL_CONFIG_KEYS:
        raise ValueError(
            "Config schema drifted from the canonical artifact: "
            f"missing {sorted(CANONICAL_CONFIG_KEYS - set(config))}, "
            f"extra {sorted(set(config) - CANONICAL_CONFIG_KEYS)}"
        )
    _dump_canonical_json(config, config_path)

    def _token_entry(content: str) -> dict[str, Any]:
        return {
            "content": content,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        }

    _dump_canonical_json(
        {
            name: _token_entry(config[name])
            for name in ("bos_token", "eos_token", "pad_token", "unk_token")
        },
        os.path.join(output_path, "special_tokens_map.json"),
    )


def _verify_apertus_1p5(output_path: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(output_path)
    if len(tokenizer) != FINAL_VOCAB_SIZE:
        raise ValueError(
            f"Built tokenizer has {len(tokenizer)} tokens, "
            f"expected {FINAL_VOCAB_SIZE}"
        )
    for literal, token_id in _VERIFY_ENCODINGS.items():
        ids = tokenizer.encode(literal, add_special_tokens=False)
        if ids != [token_id]:
            raise ValueError(f"{literal!r} encodes to {ids}, expected [{token_id}]")
    for token_id, expected in ((32, "<|inner_prefix|>"), (33, "<|inner_suffix|>")):
        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        if decoded != expected:
            raise ValueError(
                f"id {token_id} decodes to {decoded!r} under "
                f"skip_special_tokens=True, expected {expected!r}"
            )
    print(f"Verified: {FINAL_VOCAB_SIZE:,} tokens, "
          f"{len(_VERIFY_ENCODINGS)} encode pins, non-special delimiters")


def build_apertus_1p5(
    output_path: str,
    *,
    base_tokenizer_path: str = BASE_REPO,
    revision: str | None = BASE_REVISION,
    chat_template_file: str | None = None,
    work_dir: str | None = None,
) -> str:
    """Build the canonical Apertus 1.5 tokenizer from the Apertus 1 base.

    Args:
        output_path: Where to write the final tokenizer.
        base_tokenizer_path: The Apertus 1 instruct tokenizer (local dir or
            Hub ID). Defaults to the pinned canonical base.
        revision: Hub commit for base_tokenizer_path; ignored for local dirs.
        chat_template_file: The Apertus 1.5 chat template. Defaults to
            chat_templates/Apertus_1p5/chat_template.jinja in this repo.
        work_dir: Where to keep the intermediate stage directories (useful for
            debugging). A temporary directory is used and removed by default.

    Returns:
        output_path.
    """
    if chat_template_file is None:
        chat_template_file = _default_chat_template_path()
    if not os.path.exists(chat_template_file):
        raise FileNotFoundError(
            f"Chat template not found: {chat_template_file}. Pass "
            f"chat_template_file= explicitly when running outside the repo."
        )

    stages_tmp = None
    if work_dir is None:
        stages_tmp = tempfile.TemporaryDirectory(prefix="apertus_1p5_build_")
        work_dir = stages_tmp.name
    os.makedirs(work_dir, exist_ok=True)

    try:
        text_dir = os.path.join(work_dir, "text_base")
        vision_dir = os.path.join(work_dir, "vision")
        audio_dir = os.path.join(work_dir, "audio")

        prepare_apertus_1p5_text_base(
            base_tokenizer_path, text_dir, revision=revision
        )
        add_modality(
            text_dir, vision_dir, "vision", VISION_VOCAB_SIZE,
            extra_config=VISION_EXTRA_CONFIG,
        )
        add_modality(
            vision_dir, audio_dir, "audio", AUDIO_VOCAB_SIZE,
            extra_config=AUDIO_EXTRA_CONFIG,
        )
        create_instruct_tokenizer(
            audio_dir, None, output_path, chat_template_file=chat_template_file
        )
        _finalize_apertus_1p5(output_path)
        _verify_apertus_1p5(output_path)
    finally:
        if stages_tmp is not None:
            stages_tmp.cleanup()

    print(f"\nApertus 1.5 tokenizer written to {output_path}")
    return output_path
