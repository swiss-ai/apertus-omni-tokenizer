"""Format and rendering checks for chat template Jinja files.

Checks run on every .jinja file under chat_templates/: apply_chat_template()
renders conversations with the correct structure (role delimiters, content in
the right block, turns in order, a system block). Rendering compiles the
template, so broken syntax and unknown filters fail here too.

Run locally:
    pytest tests/test_chat_template.py -v
"""
from __future__ import annotations

import glob
import os

import jinja2
import pytest
from transformers import PreTrainedTokenizerFast
from transformers.utils.chat_template_utils import render_jinja_template

from tokenizer_factory import make_word_level_tokenizer

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TEMPLATES_DIR = os.path.join(_REPO_ROOT, "chat_templates")


def _jinja_files() -> list[str]:
    """Return all .jinja files under chat_templates/, sorted."""
    return sorted(
        glob.glob(os.path.join(_TEMPLATES_DIR, "**", "*.jinja"), recursive=True)
    )


def _rel(path: str) -> str:
    return os.path.relpath(path, _REPO_ROOT)


def _make_tokenizer(template_source: str) -> PreTrainedTokenizerFast:
    """Minimal tokenizer that can render any chat template with tokenize=False."""
    return make_word_level_tokenizer(
        whitespace=True, bos_eos=True, chat_template=template_source
    )


def _render(path: str, messages: list[dict]) -> str:
    source = open(path, encoding="utf-8").read()
    tok = _make_tokenizer(source)
    return tok.apply_chat_template(messages, tokenize=False)


def test_at_least_one_jinja_file_present():
    """Guards the discovery: with no files, every check below would skip."""
    assert _jinja_files(), f"No .jinja files found under {_TEMPLATES_DIR}"


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_contains_role_delimiters(path):
    """Rendered output must contain the Apertus role-delimiter tokens."""
    rendered = _render(path, [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ])
    assert "<|user_start|>" in rendered
    assert "<|user_end|>" in rendered
    assert "<|assistant_start|>" in rendered
    assert "<|assistant_end|>" in rendered


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_user_content_inside_user_delimiters(path):
    """User content must sit between <|user_start|> and <|user_end|>."""
    marker = "INSIDE_USER_BLOCK"
    rendered = _render(path, [
        {"role": "user", "content": marker},
        {"role": "assistant", "content": "ok"},
    ])
    start = rendered.index("<|user_start|>") + len("<|user_start|>")
    end = rendered.index("<|user_end|>")
    assert marker in rendered[start:end], (
        "User content is not between <|user_start|> and <|user_end|>"
    )


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_includes_system_token(path):
    """Rendered output must include a system block (default or explicit)."""
    rendered = _render(path, [{"role": "user", "content": "Hi"}])
    assert "<|system_start|>" in rendered, (
        "No <|system_start|> found - template must emit a system block"
    )


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_explicit_system_message_honored(path):
    """An explicit system message must appear in the rendered output."""
    marker = "CUSTOM_SYSTEM_PROMPT_99"
    rendered = _render(path, [
        {"role": "system", "content": marker},
        {"role": "user", "content": "Hi"},
    ])
    assert marker in rendered, "Explicit system message content missing from output"


@pytest.mark.parametrize("path", _jinja_files(), ids=_rel)
def test_template_multi_turn_all_turns_present_in_order(path):
    """Every turn of a multi-turn conversation is rendered, in order."""
    markers = ["TURN_1_USER", "TURN_1_ASSISTANT", "TURN_2_USER", "TURN_2_ASSISTANT"]
    roles = ["user", "assistant", "user", "assistant"]
    rendered = _render(path, [
        {"role": role, "content": marker} for role, marker in zip(roles, markers)
    ])
    for marker in markers:
        assert marker in rendered, f"{marker} missing from multi-turn render"
    positions = [rendered.index(marker) for marker in markers]
    assert positions == sorted(positions), "turns rendered out of order"


# ---------------------------------------------------------------------------
# Tool-call rendering: arguments may arrive as a JSON string (HF
# apply_chat_template passes them through verbatim) or as an already-parsed dict
# (vLLM json-parses tool_calls before rendering the prompt). The template must
# render both. Regression test for issue #4: replaying a prior assistant tool
# call under vLLM raised 'can only concatenate str (not "dict") to str'.
# ---------------------------------------------------------------------------

_APERTUS_1P5 = os.path.join(_TEMPLATES_DIR, "Apertus_1p5", "chat_template.jinja")


def _weather_tools() -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


def _multi_turn_messages(arguments) -> list[dict]:
    """A user→assistant(tool_call)→tool conversation, parametrized on the type
    of the replayed ``arguments`` field (str for HF, dict for vLLM)."""
    return [
        {"role": "user", "content": "Weather in Zurich? Use the tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": arguments},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "14C light rain"},
    ]


def _render_with_tools(messages: list[dict]) -> str:
    source = open(_APERTUS_1P5, encoding="utf-8").read()
    tok = _make_tokenizer(source)
    return tok.apply_chat_template(messages, tokenize=False, tools=_weather_tools())


def test_tool_call_arguments_render_the_same_as_string_or_dict():
    """HF passes arguments as a JSON string, vLLM as a parsed dict; both render
    the call identically. Before the issue #4 fix the dict path raised
    'can only concatenate str (not "dict") to str'."""
    as_string = _render_with_tools(_multi_turn_messages('{"city": "Zurich"}'))
    as_dict = _render_with_tools(_multi_turn_messages({"city": "Zurich"}))
    assert '{"get_weather": {"city": "Zurich"}}' in as_string
    assert as_dict == as_string


# ---------------------------------------------------------------------------
# Tool calls in the 1.5 template (swiss-ai/apertus-program#1134): calls without
# arguments, non-string defaults, and no new turn after a tool result.
# ---------------------------------------------------------------------------


def _render_1p5(messages, tools=None, **kwargs) -> str:
    tok = _make_tokenizer(open(_APERTUS_1P5, encoding="utf-8").read())
    return tok.apply_chat_template(messages, tokenize=False, tools=tools or _weather_tools(),
                                   add_generation_prompt=True, **kwargs)


def test_call_without_arguments_renders():
    """A call without arguments reaches the template as {}; it must render, not raise."""
    msgs = _multi_turn_messages({})
    assert '[{"get_weather": {}}]' in _render_1p5(msgs)


def test_enum_parameter_with_integer_default_renders():
    """A non-string default is written as JSON instead of failing on str + int."""
    tools = [{"type": "function", "function": {"name": "set_timer", "description": "Set a timer.", "parameters": {
        "type": "object", "properties": {"minutes": {"type": "integer", "enum": [5, 10, 15], "default": 10}}}}}]
    assert "minutes?: number, // default: 10" in _render_1p5([{"role": "user", "content": "Set a timer."}], tools)


def test_prompt_after_tool_result_continues_the_turn():
    """After a tool result the generation prompt ends at the tool output, so the model continues its own turn."""
    rendered = _render_1p5(_multi_turn_messages('{"city": "Zurich"}'))
    assert rendered.endswith("<|tool_output_start|>14C light rain<|tool_output_end|>")


def test_tool_without_description_renders():
    """A tool with no description (optional in the OpenAI schema) must render, not fail on str + None."""
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    rendered = _render_1p5([{"role": "user", "content": "hi"}], tools)
    assert "type f = " in rendered


def test_prompt_after_tool_outputs_block_continues_the_turn():
    """The block-format assistant content (tool_outputs block) also ends the prompt at the tool output."""
    msgs = [
        {"role": "user", "content": "What is the weather in Zurich?"},
        {"role": "assistant", "content": {"blocks": [
            {"type": "tool_calls", "calls": [{"name": "get_weather", "arguments": {"city": "Zurich"}}]},
            {"type": "tool_outputs", "outputs": [{"output": "14C, light rain"}]},
        ]}},
    ]
    assert _render_1p5(msgs).endswith("<|tool_output_start|>14C, light rain<|tool_output_end|>")


# ---------------------------------------------------------------------------
# Standard OpenAI message shapes in the 1.5 template (follow-up to
# swiss-ai/apertus-program#1134). vLLM with --chat-template-content-format
# string never sends these, but Transformers users call apply_chat_template
# directly: an assistant tool call without content, and text content parts in
# system and assistant messages. Each must render like its string form.
# ---------------------------------------------------------------------------


def _with_assistant_content(messages: list[dict], **content) -> list[dict]:
    """The same conversation with the assistant's ``content`` key replaced, or dropped when not given."""
    out = []
    for m in messages:
        if m["role"] == "assistant":
            m = {k: v for k, v in m.items() if k != "content"} | content
        out.append(m)
    return out


@pytest.mark.parametrize("content", [{}, {"content": None}], ids=["missing", "null"])
def test_tool_call_without_content_renders_like_empty_content(content):
    """A tool call with no content (or content: null) renders like content ""; it must not raise."""
    msgs = _multi_turn_messages({"city": "Zurich"})
    assert _render_1p5(_with_assistant_content(msgs, **content)) == _render_1p5(msgs)


def test_second_tool_call_without_content_closes_the_tool_output():
    """A contentless call right after a tool result first closes that result, as content "" does."""
    msgs = _multi_turn_messages({"city": "Zurich"})
    msgs += [dict(msgs[1], tool_calls=[{"id": "call_2", "type": "function",
                                        "function": {"name": "get_weather", "arguments": {"city": "Bern"}}}]),
             {"role": "tool", "tool_call_id": "call_2", "content": "9C fog"}]
    rendered = _render_1p5(_with_assistant_content(msgs, content=None))
    assert rendered == _render_1p5(msgs)
    assert '14C light rain<|tool_output_end|><|tools_prefix|>[{"get_weather": {"city": "Bern"}}]' in rendered


@pytest.mark.parametrize("message", [{"role": "assistant"}, {"role": "assistant", "content": None}],
                         ids=["missing", "null"])
def test_assistant_message_without_content_or_tool_calls_still_raises(message):
    with pytest.raises(jinja2.exceptions.TemplateError, match="Invalid assistant message"):
        _render_1p5([{"role": "user", "content": "hi"}, message])


def _parts(*texts: str) -> list[dict]:
    return [{"type": "text", "text": t} for t in texts]


def test_system_text_parts_render_like_a_string():
    user = {"role": "user", "content": "hi"}
    as_parts = _render_1p5([{"role": "system", "content": _parts("Be brief. ", "Answer in German.")}, user])
    assert as_parts == _render_1p5([{"role": "system", "content": "Be brief. Answer in German."}, user])


def test_assistant_text_parts_render_like_a_string():
    def convo(content):
        return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": content},
                {"role": "user", "content": "and now?"}]
    assert _render_1p5(convo(_parts("Hello! ", "How can I help?"))) == _render_1p5(convo("Hello! How can I help?"))


def test_assistant_text_parts_with_tool_calls_render_like_a_string():
    msgs = _multi_turn_messages({"city": "Zurich"})
    msgs[1]["content"] = "Let me check."
    assert _render_1p5(_with_assistant_content(msgs, content=_parts("Let me check."))) == _render_1p5(msgs)


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_non_text_parts_outside_user_messages_raise(role):
    """Images and audio are only placed in user turns; elsewhere the part type is named in the error."""
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "image"}]}]
    if role == "system":
        msgs = [{"role": "system", "content": [{"type": "image"}]}, msgs[0]]
    with pytest.raises(jinja2.exceptions.TemplateError, match=f"Invalid {role} part: image"):
        _render_1p5(msgs)


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_empty_text_parts_render_like_an_empty_string(role):
    def convo(content):
        if role == "system":
            return [{"role": "system", "content": content}, {"role": "user", "content": "hi"}]
        return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": content}]
    assert _render_1p5(convo([])) == _render_1p5(convo(""))


# ---------------------------------------------------------------------------
# The tool output section in the 1.5 template. A tool message opens it, or
# joins the open one with ", ". It stays open until the next message that is
# not a tool message, or the end of the conversation, closes it. The cases
# below cover every place that opens, joins or closes it and every check that
# guards it, and pin the exact tokens rendered after the first user turn.
# ---------------------------------------------------------------------------

_A, _AE = "<|assistant_start|>", "<|assistant_end|>"
_U, _UE = "<|user_start|>", "<|user_end|>"
_TO, _TOE = "<|tool_output_start|>", "<|tool_output_end|>"
_INNER = "<|inner_prefix|>"
# With thinking off, a new assistant turn starts with a thinking section that is already closed.
_NO_THINKING = _INNER + "<|inner_suffix|>"


def _calls(city: str) -> str:
    """The rendered tool call for get_weather(city)."""
    return f'<|tools_prefix|>[{{"get_weather": {{"city": "{city}"}}}}]<|tools_suffix|>'


def _assistant(**fields) -> dict:
    return {"role": "assistant", **fields}


def _blocks(*blocks: dict, **fields) -> dict:
    return _assistant(content={"blocks": list(blocks)}, **fields)


def _call(city: str) -> dict:
    return {"id": f"call_{city}", "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": city}}}


def _block_calls(city: str) -> dict:
    return {"type": "tool_calls", "calls": [{"name": "get_weather", "arguments": {"city": city}}]}


def _block_outputs(*outputs: str) -> dict:
    return {"type": "tool_outputs", "outputs": [{"output": o} for o in outputs]}


def _response(text: str) -> dict:
    return {"type": "response", "text": text}


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content}


_THOUGHTS = {"type": "thoughts", "text": "t"}
_Q = {"role": "user", "content": "q"}
_U2 = {"role": "user", "content": "u2"}

# A string-format and a block-format conversation, both ending on an open tool
# output, and what both render after the first user turn up to that point.
_STRING_OPEN = [_Q, _assistant(content="", tool_calls=[_call("Zurich")]), _tool("r1")]
_BLOCKS_OPEN = [_Q, _blocks(_block_calls("Zurich")), _tool("r1")]
_OPEN = _A + _calls("Zurich") + _TO + "r1"


def _tail(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Everything the 1.5 template renders after the first user turn.

    Rendered with render_jinja_template so the template sees the messages as
    given: from Transformers 5 the tokenizer's apply_chat_template drops keys
    whose value is None, so content: null would arrive as a missing key.
    """
    source = open(_APERTUS_1P5, encoding="utf-8").read()
    prompts, _ = render_jinja_template([messages], tools=_weather_tools(), chat_template=source,
                                       add_generation_prompt=add_generation_prompt)
    return prompts[0].split(_UE, 1)[1]


# The next message that is not a tool message closes the tool output before it
# writes anything; so does the end of the conversation.
_CLOSES_TOOL_OUTPUT = {
    "user": (_STRING_OPEN + [_U2], _OPEN + _TOE + _AE + _U + "u2" + _UE),
    "string_content": (_STRING_OPEN + [_assistant(content="done")], _OPEN + _TOE + "done" + _AE),
    "empty_string_content": (_STRING_OPEN + [_assistant(content="")], _OPEN + _TOE + _AE),
    "string_content_with_tool_calls": (
        _STRING_OPEN + [_assistant(content="again", tool_calls=[_call("Bern")])],
        _OPEN + _TOE + "again" + _calls("Bern")),
    "text_parts": (_STRING_OPEN + [_assistant(content=_parts("do", "ne"))], _OPEN + _TOE + "done" + _AE),
    "empty_text_parts": (_STRING_OPEN + [_assistant(content=[])], _OPEN + _TOE + _AE),
    "text_parts_with_tool_calls": (
        _STRING_OPEN + [_assistant(content=_parts("again"), tool_calls=[_call("Bern")])],
        _OPEN + _TOE + "again" + _calls("Bern")),
    "null_content_with_tool_calls": (
        _STRING_OPEN + [_assistant(content=None, tool_calls=[_call("Bern")])], _OPEN + _TOE + _calls("Bern")),
    "missing_content_with_tool_calls": (
        _STRING_OPEN + [_assistant(tool_calls=[_call("Bern")])], _OPEN + _TOE + _calls("Bern")),
    "end_of_string_conversation": (_STRING_OPEN, _OPEN + _TOE + _AE),
    "user_after_blocks": (_BLOCKS_OPEN + [_U2], _OPEN + _TOE + _AE + _U + "u2" + _UE),
    "thoughts_block": (_BLOCKS_OPEN + [_blocks(_THOUGHTS)], _OPEN + _TOE + _INNER + "t" + _AE),
    "tool_calls_block": (_BLOCKS_OPEN + [_blocks(_block_calls("Bern"))], _OPEN + _TOE + _calls("Bern")),
    "response_block": (_BLOCKS_OPEN + [_blocks(_response("done"))], _OPEN + _TOE + "done" + _AE),
    "tool_outputs_block_after_thoughts": (
        _BLOCKS_OPEN + [_blocks(_THOUGHTS, _block_outputs("r2"))],
        _OPEN + _TOE + _INNER + "t" + _TO + "r2" + _TOE + _AE),
    "empty_blocks_then_user": (_BLOCKS_OPEN + [_blocks(), _U2], _OPEN + _TOE + _AE + _U + "u2" + _UE),
    "empty_blocks_with_tool_calls": (
        _BLOCKS_OPEN + [_blocks(tool_calls=[_call("Bern")])], _OPEN + _TOE + _calls("Bern")),
    "end_of_block_conversation": (_BLOCKS_OPEN, _OPEN + _TOE + _AE),
}


@pytest.mark.parametrize("messages, expected", _CLOSES_TOOL_OUTPUT.values(), ids=list(_CLOSES_TOOL_OUTPUT))
def test_next_message_closes_the_tool_output(messages, expected):
    assert _tail(messages) == expected


def test_tool_results_in_a_row_share_one_tool_output():
    assert _tail(_STRING_OPEN + [_tool("r2"), _tool("r3")]) == _OPEN + ", r2, r3" + _TOE + _AE


def test_tool_result_after_a_closed_tool_output_opens_a_new_one():
    msgs = _STRING_OPEN + [_assistant(content="", tool_calls=[_call("Bern")]), _tool("r2")]
    assert _tail(msgs) == _OPEN + _TOE + _calls("Bern") + _TO + "r2" + _TOE + _AE


_TOOL_OUTPUT_ERRORS = {
    "tool_before_any_assistant": ([_Q, _tool("r1")], "Tool message outside of assistant"),
    "tool_after_user_closed_the_turn": (_STRING_OPEN + [_U2, _tool("r2")], "Tool message outside of assistant"),
    "tool_outputs_block_while_tool_output_open": (
        _BLOCKS_OPEN + [_blocks(_block_outputs("r2"))],
        "Cannot have both tool outputs as separate messages and tool outputs as blocks"),
}


@pytest.mark.parametrize("messages, error", _TOOL_OUTPUT_ERRORS.values(), ids=list(_TOOL_OUTPUT_ERRORS))
def test_misplaced_tool_output_raises(messages, error):
    with pytest.raises(jinja2.exceptions.TemplateError, match=error):
        _tail(messages)


# With a generation prompt, a conversation that ends on a tool output stops
# right after it, so the model continues its own turn. Anything written after
# the tool output, including empty content, ends the turn and opens a new one.
# These render with thinking off (the default), so the new turn starts with a
# closed thinking section.
_GENERATION_PROMPT = {
    "after_tool_result": (_STRING_OPEN, _OPEN + _TOE),
    "after_tool_results_in_a_row": (_STRING_OPEN + [_tool("r2")], _OPEN + ", r2" + _TOE),
    "after_tool_result_in_block_conversation": (_BLOCKS_OPEN, _OPEN + _TOE),
    "after_tool_outputs_block": ([_Q, _blocks(_block_calls("Zurich"), _block_outputs("r1"))], _OPEN + _TOE),
    "after_answer": (_STRING_OPEN + [_assistant(content="done")], _OPEN + _TOE + "done" + _AE + _A + _NO_THINKING),
    "after_empty_string_content": (_STRING_OPEN + [_assistant(content="")], _OPEN + _TOE + _AE + _A + _NO_THINKING),
    "after_response_block": (
        [_Q, _blocks(_block_calls("Zurich"), _block_outputs("r1"), _response("done"))],
        _OPEN + _TOE + "done" + _AE + _A + _NO_THINKING),
    "after_user": (_STRING_OPEN + [_U2], _OPEN + _TOE + _AE + _U + "u2" + _UE + _A + _NO_THINKING),
}


@pytest.mark.parametrize("messages, expected", _GENERATION_PROMPT.values(), ids=list(_GENERATION_PROMPT))
def test_generation_prompt_after_tool_output(messages, expected):
    assert _tail(messages, add_generation_prompt=True) == expected


def _prompt_end(enable_thinking=None, messages=None) -> str:
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return _render_1p5(messages or [{"role": "user", "content": "q"}], **kwargs)


@pytest.mark.parametrize("enable_thinking", [None, False], ids=["default", "off"])
def test_new_turn_with_thinking_off_starts_with_a_closed_thinking_section(enable_thinking):
    """The model then answers without thinking, and the server sees that thinking has ended."""
    assert _prompt_end(enable_thinking).endswith(_UE + _A + _NO_THINKING)


def test_new_turn_with_thinking_on_ends_at_the_assistant_token():
    assert _prompt_end(True).endswith(_UE + _A)


@pytest.mark.parametrize("enable_thinking", [None, False, True], ids=["default", "off", "on"])
def test_prompt_after_tool_result_has_no_thinking_section(enable_thinking):
    """The turn continues after a tool result, so nothing is added in either mode."""
    assert _prompt_end(enable_thinking, _multi_turn_messages('{"city": "Zurich"}')).endswith(_TOE)


@pytest.mark.parametrize("enable_thinking", [None, False, True], ids=["default", "off", "on"])
def test_no_thinking_section_without_a_generation_prompt(enable_thinking):
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    tok = _make_tokenizer(open(_APERTUS_1P5, encoding="utf-8").read())
    rendered = tok.apply_chat_template([{"role": "user", "content": "q"}], tokenize=False,
                                       add_generation_prompt=False, **kwargs)
    assert rendered.endswith(_UE)


# An assistant message with an empty block list and no tool calls writes
# nothing, so it leaves the tool output open: the conversation renders as if
# the message were not there. With tool calls it closes the tool output first,
# like any other message that writes something (see empty_blocks_with_tool_calls).
_EMPTY_BLOCKS_LEAVE_TOOL_OUTPUT_OPEN = {
    "at_the_end": (_BLOCKS_OPEN + [_blocks()], False, _OPEN + _TOE + _AE),
    "at_the_end_with_generation_prompt": (_BLOCKS_OPEN + [_blocks()], True, _OPEN + _TOE),
    "between_tool_results": (_BLOCKS_OPEN + [_blocks(), _tool("r2")], False, _OPEN + ", r2" + _TOE + _AE),
    "twice_between_tool_results": (
        _BLOCKS_OPEN + [_blocks(), _blocks(), _tool("r2")], False, _OPEN + ", r2" + _TOE + _AE),
    "before_a_tool_calls_block": (
        _BLOCKS_OPEN + [_blocks(), _blocks(_block_calls("Bern"))], False, _OPEN + _TOE + _calls("Bern")),
    "before_a_tool_outputs_block": (
        _BLOCKS_OPEN + [_blocks(), _blocks(_block_outputs("r2"))], False,
        "Cannot have both tool outputs as separate messages and tool outputs as blocks"),
}


@pytest.mark.parametrize("messages, add_generation_prompt, expected", _EMPTY_BLOCKS_LEAVE_TOOL_OUTPUT_OPEN.values(),
                         ids=list(_EMPTY_BLOCKS_LEAVE_TOOL_OUTPUT_OPEN))
def test_empty_block_list_leaves_the_tool_output_open(messages, add_generation_prompt, expected):
    without_empty = [m for m in messages if m != _blocks()]
    for msgs in (messages, without_empty):
        if expected.startswith("Cannot"):
            with pytest.raises(jinja2.exceptions.TemplateError, match=expected):
                _tail(msgs, add_generation_prompt)
        else:
            assert _tail(msgs, add_generation_prompt) == expected


# One conversation uses one assistant format: the string form (strings, text
# parts, tool calls without content) or blocks. Mixing them raises, and so does
# an assistant message with nothing to render.
_ASSISTANT_FORMAT_ERRORS = {
    "null_content_and_empty_tool_calls": ([_Q, _assistant(content=None, tool_calls=[])], "Invalid assistant message"),
    "contentless_tool_call_after_blocks": (
        [_Q, _blocks(_response("hi")), _U2, _assistant(tool_calls=[_call("Bern")])], "Invalid assistant message"),
    "string_after_blocks": ([_Q, _blocks(_response("hi")), _U2, _assistant(content="x")], "Invalid assistant content"),
    "text_parts_after_blocks": (
        [_Q, _blocks(_response("hi")), _U2, _assistant(content=_parts("x"))], "Invalid assistant content"),
    "blocks_after_string": ([_Q, _assistant(content="x"), _U2, _blocks(_response("hi"))], "Invalid assistant content"),
    "blocks_after_text_parts": (
        [_Q, _assistant(content=_parts("x")), _U2, _blocks(_response("hi"))], "Invalid assistant content"),
    "blocks_after_contentless_tool_call": (
        [_Q, _assistant(tool_calls=[_call("Bern")]), _tool("r"), _U2, _blocks(_response("hi"))],
        "Invalid assistant content"),
    "image_among_text_parts": (
        [_Q, _assistant(content=_parts("x") + [{"type": "image"}])], "Invalid assistant part: image"),
}


@pytest.mark.parametrize("messages, error", _ASSISTANT_FORMAT_ERRORS.values(), ids=list(_ASSISTANT_FORMAT_ERRORS))
def test_invalid_assistant_message_raises(messages, error):
    with pytest.raises(jinja2.exceptions.TemplateError, match=error):
        _tail(messages)


# ---------------------------------------------------------------------------
# Meta-tests: guard the guards.
#
# Deliberately broken templates (tests/examples/) are fed through the SAME
# checks the suite runs on real templates; each check must report a failure. If
# someone weakens a check so it passes everything, the matching meta-test below
# starts failing. These live under tests/ (not chat_templates/) so the
# parametrized suite above never picks them up.
# ---------------------------------------------------------------------------

_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


def _example(name: str) -> str:
    return os.path.join(_EXAMPLES_DIR, name)


def test_meta_good_example_passes_the_checks():
    """Sanity check: the unbroken baseline example passes the real checks, so a
    failure below is attributable to the breakage and not to the baseline."""
    path = _example("good_chat_template.jinja")
    test_template_contains_role_delimiters(path)
    test_template_user_content_inside_user_delimiters(path)


def test_meta_broken_syntax_is_reported():
    """broken_syntax.jinja (an unterminated tag) fails as soon as it renders."""
    with pytest.raises(jinja2.TemplateSyntaxError):
        test_template_contains_role_delimiters(_example("broken_syntax.jinja"))


def test_meta_unknown_filter_is_reported():
    """bad_filter.jinja (a typo'd ``|jion``) fails as soon as it renders:
    rendering compiles the template, which resolves filters."""
    with pytest.raises(jinja2.TemplateAssertionError):
        test_template_contains_role_delimiters(_example("bad_filter.jinja"))


def test_meta_corrupted_delimiters_are_reported():
    """corrupted_delimiters.jinja (the djlint --reformat failure mode: a stray
    space inside every special token) must be caught by the delimiter and
    content-placement checks."""
    path = _example("corrupted_delimiters.jinja")
    with pytest.raises(AssertionError):
        test_template_contains_role_delimiters(path)
    # Content-placement check locates the token with str.index, which raises
    # ValueError when the (now corrupted) token is absent.
    with pytest.raises((AssertionError, ValueError)):
        test_template_user_content_inside_user_delimiters(path)


def test_meta_dropped_user_content_is_reported():
    """dropped_user_content.jinja (discards the user message) must fail the
    user-block check."""
    with pytest.raises(AssertionError):
        test_template_user_content_inside_user_delimiters(
            _example("dropped_user_content.jinja")
        )