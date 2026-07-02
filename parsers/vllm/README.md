# Apertus vLLM parsers

Drop-in **vLLM plugins** for Apertus tool calling and reasoning. They live here
until they are upstreamed into vLLM itself (SGLang's equivalents are already in
`sgl-project/sglang`: `apertus2509_detector.py` and the `apertus2509` reasoning
detector).

| File | Registers | vLLM flag |
|------|-----------|-----------|
| `apertus_tool_parser.py` | tool parser `apertus` | `--tool-call-parser apertus --tool-parser-plugin .../apertus_tool_parser.py` |
| `apertus_reasoning_parser.py` | reasoning parser `apertus` | `--reasoning-parser apertus --reasoning-parser-plugin .../apertus_reasoning_parser.py` |

## vLLM version compatibility

`apertus_tool_parser.py` targets the **refactored** vLLM module layout and is
verified to load unmodified across **v0.19 – v0.24+**:

| vLLM range | Layout | This plugin |
|---|---|---|
| **v0.19 – v0.24+** | `vllm.tool_parsers` + split `protocol.py` + `vllm.tokenizers` | ✅ runs unmodified |
| v0.22 – v0.24 | upstream ships `vllm/tool_parsers/apertus_tool_parser.py` (byte-identical to this, minus the plugin `register_module` line) | ✅ |
| **≤ v0.10.2** (pre-refactor, e.g. the 70B image) | `vllm.entrypoints.openai.tool_parsers` + single `protocol.py` + `AnyTokenizer`; base `__init__(tokenizer)` takes no `tools` arg | ❌ needs a separate backport |

Verified by resolving every imported symbol against the actual vLLM source at
each tag; the `ToolParser` base `__init__(tokenizer, tools)` contract is stable
across the supported range.

Enable reasoning generation (Apertus's own chat-template switch) with
`--default-chat-template-kwargs.enable_thinking true`.

## What they parse

- **Tool calls** — `<|tools_prefix|>[{"tool_name": {..args..}}, ...]<|tools_suffix|>`
- **Reasoning** — the deliberation block between the reasoning start/end tokens
  (ids 32/33), emitted as `<|inner_prefix|>…<|inner_suffix|>` **or**
  `<think>…</think>` depending on the tokenizer build (see below).

Both cover **both serving modes**:

| Parser | Non-streaming | Streaming |
|--------|---------------|-----------|
| tool | `extract_tool_calls` | `extract_tool_calls_streaming` |
| reasoning | `extract_reasoning` | `extract_reasoning_streaming` (token-id based) |

## Tokenizer-scheme note (why the reasoning parser is "handle both")

The model always emits reasoning-start/end at **ids 32/33**, but the *string*
those ids decode to differs across tokenizer builds:

| | `<\|inner_prefix\|>` | `<think>` |
|---|---|---|
| this repo (`tokenizers/Apertus_1p5`) | id 32 (canonical) | — |
| some deployed builds | id 69 | id 32 (emitted) |

`BaseThinkingReasoningParser` resolves the token id from the *string*, so
`apertus_reasoning_parser.py` picks whichever candidate pair the loaded tokenizer
exposes at the lower (emitted) id — working under either scheme with no edits.

## Caveat (apertus-omni-tokenizer #5)

The reasoning delimiters are registered as **special** tokens, so on the
**non-streaming** path they are stripped from the detokenized string unless
`skip_special_tokens=false` (the tool parser forces this when tools are active;
clients can pass it otherwise). **Streaming is unaffected** (it keys on token
ids). Registering the delimiters as *non-special* in the tokenizer builder would
remove this caveat entirely.
