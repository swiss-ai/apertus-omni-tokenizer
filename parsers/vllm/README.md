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
- **Reasoning** — the deliberation block `<|inner_prefix|>…<|inner_suffix|>`
  (ids 32/33 in the Apertus 1.5 tokenizer). `<think>` does not delimit
  reasoning (see below).

Both cover **both serving modes**:

| Parser | Non-streaming | Streaming |
|--------|---------------|-----------|
| tool | `extract_tool_calls` | `extract_tool_calls_streaming` |
| reasoning | `extract_reasoning` | `extract_reasoning_streaming` (token-id based) |

## Tool calls inside thinking (swiss-ai/apertus-program#1134)

Apertus 1.5 calls tools inside its thinking section, so a reply that calls a
tool ends with `<|tools_prefix|>[...]<|tools_suffix|>` and never emits the
reasoning-end token. The reasoning parser treats `<|tools_prefix|>` like the
end token: the text before it is reasoning, the call is content and reaches the
tool parser. This works in both serving modes; before, a streamed call made
without thinking reached the client as raw `<|tools_prefix|>[...]` text. After a
tool result, an answer the model does not close is content.

**Chat-template independent.** The parser and the chat template are rolled out
separately (vLLM releases vs. model repos), so the parser works with every 1.5
template: the released one, the one with the #1134 fixes, and one that opens the
thinking section in the prompt. It learns whether the prompt left a thinking
section open from the prompt **token ids** (`set_prompt_token_ids` in recent
vLLM, `adjust_initial_state_from_prompt` when streaming), never from the request.
Where vLLM passes neither, output without reasoning tokens is content: a thought
may show up as content, but an answer or a call is never hidden in `reasoning`.

**Thinking off.** With `enable_thinking` false or unset, the 1.5 template starts
a new assistant turn with an empty, closed thinking section
(`<|assistant_start|><|inner_prefix|><|inner_suffix|>`). The model then does not
open a thinking section on its own, and vLLM sees from the prompt that reasoning
has ended, so it enforces `response_format` and a forced `tool_choice` from the
first token. With thinking on, the prompt ends at `<|assistant_start|>` as before.

Based on Imanol Schlag's parser in
[swiss-ai/apertus-tool-use-thinking-demo](https://github.com/swiss-ai/apertus-tool-use-thinking-demo).
That version decided from `enable_thinking` in the request that the prompt
opened the thinking section, which is only true for a patched template; with
the released template (or fixes 1-3 only) it returned direct answers as
`reasoning` with empty `content`. His tolerant tool parser (calls written as
`<invoke>` XML or `{"name": ..., "arguments": ...}` JSON) is not included yet.

## `<think>` is not a reasoning delimiter

The parser delimits reasoning with `<|inner_prefix|>`/`<|inner_suffix|>` only.
Ids in this repo's tokenizers:

| Tokenizer | `<\|inner_prefix\|>`/`<\|inner_suffix\|>` | `<think>`/`</think>` |
|---|---|---|
| `Apertus_1p5` | 32/33 | text normalized to 32/33; plain vocab entries at 69/70 |
| `Apertus_1`, `Apertus_1_base` | 69/70 | 32/33 |
| `Apertus_2` | 12/13 | 22/23 |

The parser used to pick whichever pair started at the lower id, so with an
Apertus 1 tokenizer it took `<think>`, although the Apertus 1 chat template
writes `<|inner_prefix|>`. On Apertus 1.5 it always took the inner pair, and
the 8B and 70B served for swiss-ai/vllm#34 opened every reasoning section with
32 and closed it with 33, never 69/70. A tokenizer without the inner pair, such
as the internal `Apertus-v1p5-tool_output_toks-think_toks` build, now makes the
parser fail at startup.

Not fixed: when Apertus 1.5 mentions `<think>` tags in its output, it writes
ids 32/33, which breaks the reasoning split (swiss-ai/apertus-program#1377).

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

```python
from omnitok.io import mark_tokens_non_special
from omnitok.recipes.apertus_1p5 import REASONING_DELIMITER_TOKENS

mark_tokens_non_special("/path/to/served/model", REASONING_DELIMITER_TOKENS)
```

The tokenizer builder now applies this automatically (`mark_tokens_non_special`),
so freshly built tokenizers need no fix-up.

## Tests

`tests/test_parsers.py` checks the plugin files statically on every run. Its
three live tests load the plugins into vLLM, so they run only where vLLM is
importable; CI does not install vLLM and skips them. Run them in an
environment with vLLM:

```bash
pytest tests/test_parsers.py -v
```
