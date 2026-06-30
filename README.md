# omnitok

This package documents the chat templates and core tokenizers used in Apertus, and provides utilities for extending LLaMA-3/Apertus text tokenizers with additional vision and audio modalities.

## Extension Scripts

```bash
# Add vision
python -m omnitok.cli add-modality \
    --input-tokenizer swiss-ai/Apertus-8B-2509 \
    --output-path ./omni_vision \
    --modality vision \
    --vocab-size 131072

# Stack audio on top
python -m omnitok.cli add-modality \
    --input-tokenizer ./omni_vision \
    --output-path ./omni_vision_audio \
    --modality audio \
    --vocab-size 4096

# Add instruct (chat template + SFT sequences)
python -m omnitok.cli add-instruct \
    --base-tokenizer-path ./omni_vision_audio \
    --instruct-tokenizer-path swiss-ai/Apertus-8B-2509-Instruct \
    --output-path ./omni_instruct
```

Or as a library:

```python
from omnitok import add_modality, create_instruct_tokenizer

add_modality("swiss-ai/Apertus-8B-2509", "./omni_vision", "vision", vocab_size=131072)
add_modality("./omni_vision", "./omni_vision_audio", "audio", vocab_size=4096)
create_instruct_tokenizer("./omni_vision_audio", "swiss-ai/Apertus-8B-2509-Instruct", "./omni_instruct")
```

## Token layout

```
[0 .. base-1]           text tokens (unchanged)
[base .. base+199]      200 reserved OMNI slots
  slot 0                  boundary marker (never renamed)
  slots 1-7               vision structure tokens
  slots 8-15              audio structure tokens
  slots 16-199            reserved for future modalities
[base+200 .. ]          content tokens (appended per modality in order added)
```

## Reserved slot allocation

| Slots | Modality | Tokens |
|-------|----------|--------|
| 0 | -- | `<\|RESERVED_OMNI_000\|>` (boundary) |
| 1-7 | Vision | img_start, img_end, img_token_start, img_end_of_row, img_end_of_frame, img_generation_start, image |
| 8-15 | Audio | audio_start, audio_end, stt_transcribe, stt_continue, tts_continue, audio, stt_translate, audio_annotate |
| 16-199 | -- | Reserved |

## Token aliases

`<image>` and `<|image|>` encode to the same token ID. Same for `<audio>` / `<|audio|>`. Handled by the tokenizer's normalizer -- no manual `.replace()` needed in data loaders.

## In-place mode (reuse a pre-baked reserved pool)

The default `add-modality` **appends** a 200-slot `RESERVED_OMNI` block on top of the
base vocab (the layout above). Some base tokenizers instead **pre-bake** the special
tokens *inside* their vocab — e.g. the 200k multilingual tokenizer ships `<|image|>`,
`<|audio|>`, and a free reserve pool `<SPECIAL_27>`…`<SPECIAL_123>`. For those, use
`--allocation in_place`: existing tokens are **reused**, structure tokens are
**renamed from the pool**, and only the content (codebook) tokens are appended.

```bash
# vision: reuse <|image|>; rename <SPECIAL_27..32> -> img_start..img_generation_start; append content
python -m omnitok.cli add-modality --allocation in_place \
    --input-tokenizer ./preliminary_mul_200k --output-path ./omni_vision \
    --modality vision --vocab-size 131072

# stack audio: reuse <|audio|>; rename the next free <SPECIAL_*> slots; append content
python -m omnitok.cli add-modality --allocation in_place \
    --input-tokenizer ./omni_vision --output-path ./omni_vision_audio \
    --modality audio --vocab-size 4096

# preview only -- resolve + print the change report, write nothing
python -m omnitok.cli add-modality --allocation in_place --dry-run \
    --input-tokenizer ./preliminary_mul_200k --output-path ./omni_vision \
    --modality vision --vocab-size 131072
```

```python
from omnitok import add_modality
add_modality("./preliminary_mul_200k", "./omni_vision", "vision", 131072, allocation="in_place")
add_modality("./omni_vision", "./omni_vision_audio", "audio", 4096, allocation="in_place")
```

Layout (content appends after the base vocab; structure tokens stay in place):

```
[0 .. base-1]   text + pre-baked specials, incl. <|image|>, <|audio|>, <SPECIAL_*> pool
  reused          <|image|>, <|audio|>            (kept at their existing ids)
  renamed         img_start..img_generation_start <- <SPECIAL_27..32>
                  audio_start..audio_annotate     <- <SPECIAL_33..39>
[base .. ]      content tokens (appended per modality in order added)
```

Flags:

- `--allocation {append,in_place}` (default `append`).
- `--slot-assignments '{"<|img_start|>": 40}'` — pin a structure token to a pool
  **ordinal** (the N in `<SPECIAL_N>`) instead of auto-filling the lowest free slot.
- `--allow-existing` / `--no-allow-existing` (default allow) — always skip tokens that
  already exist; `--no-allow-existing` errors on any *unexpected* pre-existing token
  (declared-reuse tokens like `<|image|>` are exempt).
- `--dry-run` — print the full change report (reused, renames, aliases, content id
  range) without writing any files.

In-place `omnimodal_config` gains `allocation: "in_place"`, the claimed reserve-pool
range (`omni_special_token_offset` + `omni_special_token_count`), an authoritative
`omni_special_token_ids` union, and per-modality `special_region_offset` /
`special_region_count` / `reused_special_ids` / `structure_token_ids`. Under scattered
`--slot-assignments`, treat `omni_special_token_ids` as authoritative.

## Known codebook sizes

| Tokenizer | Modality | Codebook Size |
|-----------|----------|---------------|
| Emu3.5 (IBQ) | Vision | 131,072 |
| Emu3 | Vision | 32,768 |
| WavTokenizer | Audio | 4,096 |

## Validating a deployed model

To check that a served model directory ships the exact canonical tokenizer (chat
template, tokenizer, config, special tokens), run `validate_model.sh` against it.
It compares md5 checksums against the manifests under `validation/` and exits
non-zero on any mismatch.

```bash
# Remote (no checkout needed): cd into the model dir, then validate it
cd /path/to/served/model
curl -fsSL https://raw.githubusercontent.com/swiss-ai/apertus-omni-tokenizer/main/validate_model.sh \
  | bash

# To validate the 1.0 tokenizer instead, pass the path slot + model name
#   | bash -s -- . Apertus_1

# From a local checkout
bash validate_model.sh /path/to/served/model
```

```
Validating Apertus_1p5 tokenizer in: /path/to/served/model

✔ chat_template.jinja
✔ tokenizer.json
✔ tokenizer_config.json
✔ special_tokens_map.json

OK: Apertus_1p5 tokenizer matches the canonical checksums.
```

Arguments: `validate_model.sh [MODEL_PATH] [MODEL_NAME]` -- `MODEL_PATH` defaults
to the current directory, `MODEL_NAME` defaults to `Apertus_1p5` (pass
`Apertus_1` to validate the 1.0 tokenizer). The canonical checksums are
regenerated by `validation/gen_checksums.sh` and kept in sync by CI.

## Structure

```
apertus-omni-tokenizer/
├── README.md
├── pyproject.toml
├── validate_model.sh        # verify a served model dir matches canonical md5s
├── omnitok/
│   ├── __init__.py      # public API exports
│   ├── modalities.py    # ModalityConfig dataclass, built-in VISION/AUDIO configs
│   ├── builder.py       # add_modality() -- the main engine
│   ├── instruct.py      # create_instruct_tokenizer() -- chat template + SFT sequences
│   ├── io.py            # low-level file I/O (rename, alias, detect, save)
│   └── cli.py           # CLI wrapper (python -m omnitok.cli)
├── tests/
│   ├── conftest.py           # shared fixtures
│   ├── test_alias.py         # token alias tests (<image> == <|image|>)
│   ├── test_builder.py       # add_modality tests
│   ├── test_chat_template.py # add chat template test
│   ├── test_task_tokens.py   # task token contract tests
│   └── test_tokenizers.py    # checked-in tokenizers load + special-token IDs
├── examples/
│   └── rename_tool_tokens.py # Script used to add new special tokens in tool calling parsing
├── tokenizers/
│   ├── Apertus_1/            # Instructed tokenizer used for Apertus 1.0
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── Apertus_1p5/          # Instructed tokenizer used for Apertus 1.5
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── chat_templates/
│   ├── Apertus_1/            # Chat template used for Apertus 1.0
│   │   └── chat_template.jinja
│   └── Apertus_1p5/          # Chat template used for Apertus 1.5
│       └── chat_template.jinja
└── validation/
    ├── gen_checksums.sh      # regenerate the manifests below
    ├── Apertus_1.md5         # canonical md5s for the 1.0 tokenizer
    └── Apertus_1p5.md5       # canonical md5s for the 1.5 tokenizer
```

Adding a new modality = one new `ModalityConfig` entry in `modalities.py`.

## Author

Yixuan Xu (yixuan.xu@ai.ethz.ch)
