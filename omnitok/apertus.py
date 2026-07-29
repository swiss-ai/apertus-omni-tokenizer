"""Build recipe for the Apertus 1.5 tokenizer.

Reproduces the canonical Apertus 1.5 tokenizer (tokenizers/Apertus_1p5, as
pinned by validation/Apertus_1p5.md5) byte-for-byte from the Apertus 1 base
(swiss-ai/Apertus-8B-2509). The recipe is the audit trail for every delta
between the two:

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
   normalizer rules and writes tokenizer_config.json /
   special_tokens_map.json / chat_template.jinja with a deterministic
   serialization so the output does not depend on which transformers version
   performed the build.

The canonical artifact deliberately repairs three defects of the originally
released RC (apertus-ai/Apertus-v1.5-8B-RC); the recipe builds the repaired
form:

- ``<|audio|>`` (131085) is ``normalized: true``, so the ``<audio>`` alias
  rule matches just like ``<image>`` does. (The RC shipped it false, leaving
  bare ``<audio>`` to encode as plain text.)
- ``eos_token`` is ``</s>`` (id 2), the sequence terminator; the turn/tool
  stop tokens live in generation_config, not here. (The RC inherited
  ``<|assistant_end|>`` from the Apertus 1 base.)
- ``tokenizer_config.json`` is the slim shape adopted from the aligned
  integration repos: no ``added_tokens_decoder`` mirror, no
  ``backend``/``is_local`` fossils, and the multimodal role tokens exposed
  via ``extra_special_tokens`` for the Apertus1p5Processor.

One canonical quirk is reproduced as-is: ``vocab_size: 131072`` is the base
text vocab, not the true total of 266,440 (use ``len(tokenizer)``).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any

from transformers import AutoTokenizer

from .builder import add_modality
from .instruct import create_instruct_tokenizer
from .io import (
    _prepend_rules,
    _resolve_tokenizer_path,
    _rewrite_backend_state,
    add_token_alias,
    mark_tokens_non_special,
    prepend_normalizer_rules,
)

BASE_REPO = "swiss-ai/Apertus-8B-2509"
# Hub repos are mutable (the sibling instruct repo had <SPECIAL_73> renamed
# to <|image|> and reverted after release), so the base is pinned.
BASE_REVISION = "3162c99675aa588097cecd4a24b9aa1f712af477"
# The build is parent-agnostic between the two Apertus 1 repos: the instruct
# tokenizer (swiss-ai/Apertus-8B-Instruct-2509 @
# b946d40447b2b597999b9c86d44bee0b452c919f) reproduces the same bytes,
# verified 2026-07-29. Its checked-in mirror (tokenizers/Apertus_1) is the
# offline build source and the validate_model.sh baseline.

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


# Multimodal role tokens surfaced to the Apertus1p5Processor, both as the
# extra_special_tokens dict and as their top-level config mirrors.
EXTRA_SPECIAL_TOKENS = {
    "audio_token": "<|audio|>",
    "boa_token": "<|audio_start|>",
    "boi_token": "<|img_start|>",
    "eoa_token": "<|audio_end|>",
    "eoi_token": "<|img_end|>",
    "eol_token": "<|img_end_of_row|>",
    "image_token": "<|image|>",
    "image_wrapper_token": "<|img_token_start|>",
}

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


def _dump_canonical_json(obj: dict[str, Any], path: str) -> None:
    """transformers-style deterministic JSON: sorted keys, indent 2, LF tail."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _add_think_aliases(save_path: str) -> None:
    """Alias the <think>/</think> literals to the delimiters at 32/33.

    Insertion order gives the canonical chain [<think> -> 32, </think> -> 33];
    add_token_alias flips both targets to normalized=True.
    """
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    add_token_alias(save_path, "<|inner_suffix|>", "</think>",
                    tokenizer=tokenizer, save=False)
    add_token_alias(save_path, "<|inner_prefix|>", "<think>",
                    tokenizer=tokenizer, save=True)


def add_reasoning_aliases(save_path: str) -> None:
    """Retrofit the canonical 1.5 reasoning rewrites onto an existing tokenizer.

    Installs the same rules the build recipe does: <think>/</think> aliased to
    the <|inner_prefix|>/<|inner_suffix|> delimiters (flipping the targets to
    normalized=True), then the REASONING_CLEANUP_RULES prepended in front.
    Idempotent; raises ValueError if a delimiter is not an added token.
    """
    _add_think_aliases(save_path)
    prepend_normalizer_rules(save_path, REASONING_CLEANUP_RULES)


def prepare_apertus_1p5_text_base(
    base_tokenizer_path: str,
    output_path: str,
    *,
    revision: str | None = None,
) -> str:
    """Turn the Apertus 1 tokenizer into the 1.5 text tokenizer.

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

    def _apply_text_renames(state):
        vocab = state["model"]["vocab"]
        for token_id, (old, _) in TEXT_RENAMES.items():
            if vocab.get(old) != token_id:
                raise ValueError(
                    f"Expected {old!r} at id {token_id} in the base vocab, "
                    f"found id {vocab.get(old)!r}. {base_tokenizer_path} is "
                    f"not the expected Apertus 1 base "
                    f"({BASE_REPO} @ {BASE_REVISION})."
                )
        # Two phases: several renames swap names between ids, so setting new
        # names while old ones are still present would clobber entries.
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

    _rewrite_backend_state(output_path, _apply_text_renames)
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

    _add_think_aliases(output_path)
    mark_tokens_non_special(output_path)
    return output_path


def _finalize_apertus_1p5(output_path: str) -> None:
    """Write the derived files in their canonical, deterministic form."""
    # The original artifact swapped the names of audio slots 13/14 by string
    # replacement, leaving the alias-ready normalized=true flag behind at
    # slot 14 (<|stt_translate|>). The stray flag is harmless and kept as-is;
    # the <|audio|> side of the swap was repaired in the canonical artifact
    # (see module docstring), and the pipeline already builds it repaired.
    def _keep_stt_flag_and_prepend_rules(state):
        for entry in state["added_tokens"]:
            if entry["content"] == "<|stt_translate|>":
                entry["normalized"] = True
        _prepend_rules(state, REASONING_CLEANUP_RULES)

    _rewrite_backend_state(output_path, _keep_stt_flag_and_prepend_rules)

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
        "clean_up_tokenization_spaces", "model_input_names",
        "model_max_length", "omnimodal_config", "pad_token", "padding_side",
        "sft_assistant_begin_sequence", "sft_eot_token",
        "sft_user_begin_sequence", "unk_token", "vision_begin_token",
        "vision_end_token", "vision_tokenizer",
    )
    config = {key: built[key] for key in carried_keys}
    config.update(EXTRA_SPECIAL_TOKENS)
    config.update({
        # The base stops on <|assistant_end|>; 1.5 ends sequences with the
        # plain </s> terminator and keeps the turn/tool stops in
        # generation_config.
        "eos_token": "</s>",
        "extra_special_tokens": dict(EXTRA_SPECIAL_TOKENS),
        "processor_class": "Apertus1p5Processor",
        "tokenizer_class": "PreTrainedTokenizerFast",
        # Canonical quirk: the base text vocab size, not the true total.
        "vocab_size": BASE_VOCAB_SIZE,
    })
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
    if tokenizer.eos_token != "</s>":
        raise ValueError(
            f"eos_token is {tokenizer.eos_token!r}, expected '</s>'"
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
        base_tokenizer_path: The Apertus 1 tokenizer (local dir or
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
