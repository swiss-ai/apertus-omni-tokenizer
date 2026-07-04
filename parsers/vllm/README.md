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

### Why use this plugin instead of the parser built into vLLM v0.22+?

On v0.22 – v0.24 the upstream `--tool-call-parser apertus` is the **same code**
(byte-identical apart from this file's `@ToolParserManager.register_module`
line, which upstream replaces with an entry in the lazy-import table in
`vllm/tool_parsers/__init__.py`) — so for plain production serving on those
versions, just use the built-in and skip the plugin. Reach for this copy when:

- **You're on v0.19 – v0.21** — the module layout supports the plugin but the
  Apertus parser hasn't been upstreamed yet. The plugin is the only option.
- **You need to iterate on parser behavior** — this repo exists so you can
  deploy the parser manually for dev: edit the file, restart the server, done.
  No vLLM fork, rebuild, or upgrade required, and fixes can be validated here
  before being sent upstream.
- **You want the parser version pinned independently of the engine** — with the
  built-in, parser behavior silently changes whenever you bump vLLM. Loading
  this file via `--tool-parser-plugin` freezes the parsing logic across engine
  upgrades (within the compatible layout range above).
- **You need to hotfix a deployment** — a parser bug can be patched in place on
  a running deployment without waiting for (or upgrading to) a vLLM release.

Note: the plugin registers under the same name (`apertus`), and on v0.22+ the
plugin's registration takes effect via `--tool-parser-plugin`, overriding the
built-in — you don't need to rename anything to use it on those versions.

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

## Reasoning delimiters are non-special (apertus-omni-tokenizer #5)

The reasoning delimiters are registered as **non-special** tokens, so they
survive detokenization under the default `skip_special_tokens=true` and the
**non-streaming** parser can always find the end-of-reasoning delimiter — no
per-request override needed. **Streaming is unaffected** either way (it keys on
token ids).

Older builds shipped the delimiters as *special* tokens, which the default
`skip_special_tokens=true` stripped before the non-streaming parser ran, leaking
the whole deliberation block into `content`. If you hit that on a deployed model
directory, flip the flag in place:

```
python examples/mark_reasoning_delimiters_nonspecial.py /path/to/served/model
```

The tokenizer builder now applies this automatically (`mark_tokens_non_special`),
so freshly built tokenizers need no fix-up.
