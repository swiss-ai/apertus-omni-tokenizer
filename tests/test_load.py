"""The committed tokenizers, as deployed, load with the installed transformers
and expose everything their files declare.

CI runs this module in every build row and, load-only, on the training and
serving stacks below the build floor (transformers 4.48.2 / tokenizers 0.21.0
and 4.51.1 / 0.21.1).
"""

import json
import shutil

import pytest
from transformers import AutoTokenizer

from omnitok import registry

NAMES = [entry.name for entry in registry.TOKENIZERS]
TEMPLATED = [n for n in NAMES if "chat_template.jinja" in registry.committed_files(n)]

# Not values a tokenizer exposes: the added-token mirror, the class that loads
# it, and the template (checked below). The add_*_token flags only describe
# the post-processor, whose behaviour SPECIALS_ON_ENCODE pins instead:
# transformers 5.x reads Apertus 1's add_bos_token back as False yet inserts BOS.
SKIPPED_CONFIG_KEYS = (
    "added_tokens_decoder", "tokenizer_class", "chat_template",
    "add_bos_token", "add_eos_token",
)

# encode("Hello"), with the default add_special_tokens=True.
SPECIALS_ON_ENCODE = {
    "Apertus_1_base": [1, 22177],
    "Apertus_1": [1, 22177],
    "Apertus_1p5": [1, 22177],
    "Apertus_2": [1, 36971, 2],
    "Apertus_2_instruct": [36971],
}


@pytest.fixture(scope="module")
def load(tmp_path_factory):
    """Load a committed tokenizer as deployed: its files plus its chat template."""
    cache = {}

    def _load(name):
        if name not in cache:
            target = tmp_path_factory.mktemp(name)
            for f, path in registry.committed_files(name).items():
                shutil.copy(path, target / f)
            cache[name] = AutoTokenizer.from_pretrained(target), target
        return cache[name]

    return _load


def test_every_tokenizer_has_an_encode_pin():
    assert set(SPECIALS_ON_ENCODE) == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_every_config_value_is_reachable(name, load):
    """Each config value is an attribute or init kwarg of the loaded tokenizer,
    e.g. the 1.5 role tokens (tokenizer.eoa_token) and its omnimodal_config."""
    tok, target = load(name)
    config = json.loads((target / "tokenizer_config.json").read_text(encoding="utf-8"))
    unreachable = []
    for key, value in config.items():
        if key in SKIPPED_CONFIG_KEYS:
            continue
        if key == "extra_special_tokens" and isinstance(value, dict):
            unreachable += [a for a, t in value.items() if getattr(tok, a, None) != t]
        elif tok.init_kwargs.get(key) != value and getattr(tok, key, None) != value:
            unreachable.append(key)
    assert not unreachable


@pytest.mark.parametrize("name", NAMES)
def test_special_tokens_inserted_on_encode(name, load):
    tok, _ = load(name)
    assert tok.encode("Hello") == SPECIALS_ON_ENCODE[name]


@pytest.mark.parametrize("name", TEMPLATED)
def test_chat_template_renders(name, load):
    tok, target = load(name)
    assert tok.chat_template == (target / "chat_template.jinja").read_text(encoding="utf-8")
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}],
        tokenize=False,
    )
    for delimiter in ("<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>"):
        assert delimiter in rendered, delimiter


def test_apertus_1p5_renders_a_tool_call(load):
    tok, _ = load("Apertus_1p5")
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }}]
    messages = [
        {"role": "user", "content": "Weather in Zurich?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Zurich"}},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "14C light rain"},
    ]
    rendered = tok.apply_chat_template(messages, tools=tools, tokenize=False)
    assert '{"get_weather": {"city": "Zurich"}}' in rendered
    assert "14C light rain" in rendered


def test_apertus_2_named_controls_and_literal_content(load):
    tok, _ = load("Apertus_2_instruct")
    roles = {
        "input_start_token": ("<|in|>", 7),
        "input_end_token": ("<|/in|>", 8),
        "output_start_token": ("<|out|>", 10),
        "output_end_token": ("<|/out|>", 11),
        "header_end_token": ("<|hdr|>", 9),
        "wait_token": ("<|wait|>", 12),
        "pad_token": ("<|pad|>", 3),
    }
    for role, (glyph, token_id) in roles.items():
        assert getattr(tok, role) == glyph
        assert getattr(tok, role + "_id") == token_id
    specials = {i: str(t) for i, t in tok.added_tokens_decoder.items() if t.special}
    assert set(specials) == set(range(124))
    payload = " ".join(specials.values()) + " e\u0301\r\n\x00🙂"
    ids = tok.encode(payload, add_special_tokens=False, split_special_tokens=True)
    assert set(specials).isdisjoint(ids)
    assert tok.decode(ids, clean_up_tokenization_spaces=False) == payload
