"""Tests for chat-template patching in create_instruct_tokenizer()."""

from __future__ import annotations

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from omnitok.instruct import create_instruct_tokenizer

APERTUS_TEMPLATE = """{{ bos_token }}
{%- set user_token = '<|user_start|>' -%}
{%- set end_user_token = '<|user_end|>' -%}
{%- set image_token = '<|image|>' -%}
{%- set audio_token = '<|audio|>' -%}
{%- for message in messages -%}
    {%- if message.role == 'user' -%}
        {%- if "content" in message -%}
            {{ user_token }}
            {%- if message.content is string -%}
                {{ message.content }}
            {%- elif message.content is mapping and "parts" in message.content -%}
                {%- set parts = message.content.parts -%}
                {%- for part in parts -%}
                    {%- if part.type == "text" -%}
                        {{ part.text }}
                    {%- elif part.type == "image" -%}
                        {{ image_token }}
                    {%- elif part.type == "audio" -%}
                        {{ audio_token }}
                    {%- else -%}
                        {{- raise_exception("Invalid user part: " + part.type) -}}
                    {%- endif -%}
                {%- endfor -%}
            {%- endif -%}
            {{ end_user_token }}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
"""


def _make_tokenizer(chat_template: str | None = None) -> PreTrainedTokenizerFast:
    tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<|audio|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|image|>",
        "<|user_start|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
    ]
    vocab = {token: idx for idx, token in enumerate(tokens)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    if chat_template is not None:
        fast.chat_template = chat_template
    return fast


def _save_tokenizer(path, chat_template: str | None = None) -> None:
    tokenizer = _make_tokenizer(chat_template)
    tokenizer.save_pretrained(path)

    config_path = path / "tokenizer_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["base_vocab_size"] = len(tokenizer.get_vocab())
    config["vocab_size"] = len(tokenizer.get_vocab())
    config["added_tokens_count"] = 0
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _write_omnimodal_config(path) -> None:
    config_path = path / "tokenizer_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["omnimodal_config"] = {
        "omni_special_token_offset": 1000,
        "modalities": [
            {
                "name": "audio",
                "offset": 1000,
                "vocab_size": 1,
                "start_token": 3,
                "end_token": 3,
            }
        ],
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def test_template_missing_modality_placeholder_rejected(tmp_path):
    """A template that cannot render a modality the tokenizer carries fails
    the build instead of shipping silently."""
    base_dir = tmp_path / "base"
    instruct_dir = tmp_path / "instruct"

    _save_tokenizer(base_dir)
    _write_omnimodal_config(base_dir)
    text_only = "{{ bos_token }}<|user_start|>{{ messages[0].content }}<|user_end|>"
    _save_tokenizer(instruct_dir, text_only)

    with pytest.raises(ValueError, match="cannot render audio"):
        create_instruct_tokenizer(
            str(base_dir), str(instruct_dir), str(tmp_path / "out")
        )


def test_create_instruct_tokenizer_saves_audio_aware_chat_template(tmp_path):
    base_dir = tmp_path / "base"
    instruct_dir = tmp_path / "instruct"
    output_dir = tmp_path / "output"

    _save_tokenizer(base_dir)
    _write_omnimodal_config(base_dir)
    _save_tokenizer(instruct_dir, APERTUS_TEMPLATE)

    create_instruct_tokenizer(str(base_dir), str(instruct_dir), str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": {
                    "parts": [
                        {"type": "audio"},
                        {"type": "text", "text": "Transcribe this clip."},
                    ]
                },
            }
        ],
        tokenize=False,
    )

    assert "<|audio|>" in rendered

    with open(output_dir / "tokenizer_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    assert "audio_token = '<|audio|>'" in config["chat_template"]
    assert config["audio_begin_token"] == tokenizer.encode(
        "<|audio_start|>", add_special_tokens=False
    )
    assert config["audio_end_token"] == tokenizer.encode(
        "<|audio_end|>", add_special_tokens=False
    )
