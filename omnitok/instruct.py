"""Create instruct omni-tokenizers.

Adds chat template and pre-tokenized SFT sequences to a base omni-tokenizer.
Modality-agnostic: works regardless of which modalities are present.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

from transformers import AutoTokenizer

from .io import detect_existing_modalities
from .modalities import MODALITY_REGISTRY


_LLAMA_MULTIMODAL_RENDERER = """{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content | trim -}}
    {%- elif content is sequence -%}
        {%- for item in content -%}
            {%- if item is string -%}
                {{- item | trim -}}
            {%- elif item is mapping -%}
                {%- if item.type == "text" -%}
                    {{- item.text | trim -}}
                {%- elif item.type == "image" or item.type == "image_url" or item.type == "input_image" -%}
                    {{- "<|image|>" -}}
                {%- elif item.type == "audio" or item.type == "audio_url" or item.type == "input_audio" -%}
                    {{- "<|audio|>" -}}
                {%- else -%}
                    {{- raise_exception("Invalid content item: " + item.type) -}}
                {%- endif -%}
            {%- else -%}
                {{- raise_exception("Invalid content item") -}}
            {%- endif -%}
            {%- if not loop.last -%}
                {{- "\\n" -}}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{- raise_exception("Invalid message content") -}}
    {%- endif -%}
{%- endmacro -%}"""


def _patch_llama_chat_template(chat_template: str) -> str:
    """Teach LLaMA-style templates to render structured image/audio blocks."""
    if "macro render_content(content)" in chat_template:
        return chat_template

    replacements = (
        ("messages[0]['content']|trim", "render_content(messages[0]['content'])"),
        ("messages[0]['content'] | trim", "render_content(messages[0]['content'])"),
        ('messages[0]["content"]|trim', 'render_content(messages[0]["content"])'),
        ('messages[0]["content"] | trim', 'render_content(messages[0]["content"])'),
        ("message['content']|trim", "render_content(message['content'])"),
        ("message['content'] | trim", "render_content(message['content'])"),
        ('message["content"]|trim', 'render_content(message["content"])'),
        ('message["content"] | trim', 'render_content(message["content"])'),
    )

    patched = chat_template
    replaced = False
    for old, new in replacements:
        if old in patched:
            patched = patched.replace(old, new)
            replaced = True

    if not replaced:
        return chat_template

    return _LLAMA_MULTIMODAL_RENDERER + "\n\n" + patched


_APERTUS_OMNI_SYSTEM_PROMPT = (
    "You are Apertus 1.5 Omni, a multimodal assistant developed by the "
    "Swiss AI Initiative. Extended from Apertus 1 via continued "
    "pretraining, you understand images and audio and respond in text."
)


def _patch_apertus_chat_template(chat_template: str) -> str:
    """Extend Apertus chat template for omni SFT.

    - Adds audio rendering to user content parts.
    - Replaces the default no-system-message fallback with a static
      omni-aware system prompt. Drops the dynamic ``strftime_now`` call
      and the stale ``Knowledge cutoff`` line so that samples without an
      explicit system message render deterministically across training
      runs. Explicit system messages in data are still honored.
    """
    if "audio_token = '<|audio|>'" in chat_template:
        return chat_template

    patched = chat_template
    image_token_line = "{%- set image_token = '<|image|>' -%}"
    if image_token_line in patched:
        patched = patched.replace(
            image_token_line,
            image_token_line + "\n{%- set audio_token = '<|audio|>' -%}",
            1,
        )

    image_branch = """{%- elif part.type == "image" -%}
                        {{ image_token }}
                    {%- else -%}
                        {{- raise_exception("Invalid user part: " + part.type) -}}
                    {%- endif -%}"""
    audio_branch = """{%- elif part.type == "image" or part.type == "image_url" or part.type == "input_image" -%}
                        {{ image_token }}
                    {%- elif part.type == "audio" or part.type == "audio_url" or part.type == "input_audio" -%}
                        {{ audio_token }}
                    {%- else -%}
                        {{- raise_exception("Invalid user part: " + part.type) -}}
                    {%- endif -%}"""
    patched = patched.replace(image_branch, audio_branch, 1)

    old_default_expr = (
        "'You are Apertus, a helpful assistant created by the SwissAI "
        "initiative.\\nKnowledge cutoff: 2024-04\\nCurrent date: ' "
        "+ strftime_now('%Y-%m-%d')"
    )
    new_default_expr = "'" + _APERTUS_OMNI_SYSTEM_PROMPT + "'"
    patched = patched.replace(old_default_expr, new_default_expr, 1)

    return patched


def create_instruct_tokenizer(
    base_tokenizer_path: str,
    instruct_tokenizer_path: str,
    output_path: str,
) -> tuple[Any, dict[str, Any]]:
    """Add chat template and SFT sequences to a base omni-tokenizer.

    Args:
        base_tokenizer_path: Path to base omni-tokenizer (with modality tokens).
        instruct_tokenizer_path: Path or HF model ID for chat template source.
        output_path: Where to save the instruct tokenizer.

    Returns:
        (tokenizer, stats) tuple.
    """
    print("=" * 60)
    print("CREATING INSTRUCT OMNI-TOKENIZER")
    print("=" * 60)

    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
    print(f"Loaded base tokenizer: {len(tokenizer):,} tokens")

    # Verify it's an omni-tokenizer
    existing = detect_existing_modalities(base_tokenizer_path)
    if not existing["modalities"]:
        raise ValueError(
            f"No modality mapping files found in {base_tokenizer_path}. "
            f"Create an omni-tokenizer first with `add_modality()`."
        )
    print(f"Detected modalities: {list(existing['modalities'].keys())}")

    # Load chat template
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_tokenizer_path)
    chat_template = instruct_tokenizer.chat_template
    if not chat_template:
        raise ValueError(
            f"No chat template found in {instruct_tokenizer_path}."
        )

    # Copy base to output
    if os.path.abspath(base_tokenizer_path) != os.path.abspath(output_path):
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        shutil.copytree(base_tokenizer_path, output_path)

    # Load config
    config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    stats = {
        "base_tokenizer": base_tokenizer_path,
        "instruct_tokenizer": instruct_tokenizer_path,
        "vocab_size": config.get("vocab_size", len(tokenizer)),
        "modalities": list(existing["modalities"].keys()),
    }

    # Detect chat template style and add SFT sequences
    if "<|start_header_id|>" in chat_template:
        chat_template = _patch_llama_chat_template(chat_template)
        user_header = "<|start_header_id|>user<|end_header_id|>"
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
        eot_token = "<|eot_id|>"
        print("Detected LLaMA-3 style chat template")
    elif "<|user_start|>" in chat_template:
        chat_template = _patch_apertus_chat_template(chat_template)
        user_header = "<|user_start|>"
        assistant_header = "<|assistant_start|>"
        eot_token = "<|assistant_end|>"
        print("Detected Apertus style chat template")
    else:
        raise ValueError(
            "Unsupported chat template. Supported: LLaMA-3, Apertus."
        )

    config["chat_template"] = chat_template

    config["sft_user_begin_sequence"] = tokenizer.encode(
        user_header, add_special_tokens=False
    )
    config["sft_assistant_begin_sequence"] = tokenizer.encode(
        assistant_header, add_special_tokens=False
    )
    config["sft_eot_token"] = tokenizer.encode(
        eot_token, add_special_tokens=False
    )

    print(f"  sft_user_begin: {config['sft_user_begin_sequence']}")
    print(f"  sft_assistant_begin: {config['sft_assistant_begin_sequence']}")
    print(f"  sft_eot_token: {config['sft_eot_token']}")

    # Add modality boundary tokens for each detected modality
    for name, mc in MODALITY_REGISTRY.items():
        if name in existing["modalities"]:
            begin_ids = tokenizer.encode(mc.start_token, add_special_tokens=False)
            end_ids = tokenizer.encode(mc.end_token, add_special_tokens=False)
            config[f"{name}_begin_token"] = begin_ids
            config[f"{name}_end_token"] = end_ids
            print(f"  {name}_begin_token: {begin_ids} ({mc.start_token})")
            print(f"  {name}_end_token: {end_ids} ({mc.end_token})")

    # Save config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Reload from output so returned tokenizer has chat_template set
    tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)

    print(f"\nSaved to {output_path}")
    return tokenizer, stats
