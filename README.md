# omnitok

This repo builds the canonical Apertus 1.5 tokenizer (`tokenizers/Apertus_1p5`,
the reference for the [apertus-ai/Apertus-v1.5-8B](https://huggingface.co/apertus-ai/Apertus-v1.5-8B)
release) from the Apertus 1 base
([swiss-ai/Apertus-8B-2509](https://huggingface.co/swiss-ai/Apertus-8B-2509)),
and documents the chat templates and canonical tokenizer files for both releases.

## Where each tokenizer comes from

`omnitok/registry.py` lists every directory under `tokenizers/` and records
where it comes from. An *imported* tokenizer is an unchanged upstream file set,
pinned to a revision and to the SHA-256 of every file. A *derived* tokenizer is
built offline by a recipe from another registered tokenizer, its parent.
`tests/test_registry.py` checks the pins and rebuilds every derived tokenizer
byte-for-byte. It also fails on unregistered directories, so a new tokenizer
needs only a registry entry, plus a recipe `(parent_dir, output_dir)` if it is
derived.

## Apertus 2: text-only base and instruct

`tokenizers/Apertus_2/` is the byte-identical 200,064-token text base from
[`preliminary_mul_200k`](https://github.com/swiss-ai/apertus-tokenizer-development/tree/28ad57a2757f6f72edb0def57ee725b6812f2df3/preliminary_mul_200k).
Its revision and SHA-256 hashes are pinned in `omnitok/registry.py`. It retains the upstream NFC
normalizer and BOS/EOS postprocessor. No vision/audio codebook vocabulary or
modality metadata is added. Existing upstream special tokens, including the
image/audio placeholders, remain at their original IDs.

`tokenizers/Apertus_2_instruct/` derives the same-size conversation tokenizer:

| ID | Reserved name replaced by |
|----|---------------------------|
| 40 | `<\|in\|>`                |
| 41 | `<\|/in\|>`               |
| 42 | `<\|out\|>`               |
| 43 | `<\|/out\|>`              |
| 44 | `<\|hdr\|>`               |
| 45 | `<\|wait\|>`              |
| 46 | `<\|pad\|>`               |

All other vocabulary IDs and BPE merges stay unchanged. The instruct variant
removes normalization and automatic BOS/EOS insertion, uses wait as EOS and
ID 46 for padding, and writes `generation_config.json` and the
`apertus_encoding.json` binding consumed by `apertus-common`. The inherited
model-length setting is unspecified; consumers configure the checkpoint's limit.

There is no Jinja template. `apertus-common` renders messages, inserts controls
and disables special-token recognition while encoding ordinary text. Direct
Hugging Face tokenization alone does not provide that payload isolation.
The binding targets profile revision `85874b84605f2a0452d53fe5874cc5eddac1b7f4`
and `tokenizers>=0.23.2,<0.24` (tested with 0.23.2).

Build into empty, separate output directories; both operations verify all three
upstream source files before writing and refuse source/output overlap:

```sh
# Copy a checkout of the pinned upstream text source without changing its bytes.
python -m omnitok.cli build-apertus-2 base \
  --input-tokenizer /path/to/preliminary_mul_200k --output-path /tmp/Apertus_2

# Rebuild offline from the committed base.
python -m omnitok.cli build-apertus-2 instruct \
  --input-tokenizer tokenizers/Apertus_2 --output-path /tmp/Apertus_2_instruct

pytest tests/test_registry.py tests/test_apertus_2_recipe.py tests/test_apertus_2_validation.py -v
bash validation/gen_checksums.sh
```

`validate_model.sh MODEL_PATH Apertus_2_instruct` checks EOS 45 and padding 46
with Python 3 while preserving checkpoint-specific generation settings. These
fields are validated separately from the tokenizer checksum manifest.

Apertus 1/1.5 artifacts are unchanged. Multimodal support is planned for
Apertus 2.5: `add_modality_in_place` provides the in-place allocation it needs
(see [docs/omnimodal_config.md](docs/omnimodal_config.md)), and the multimodal
Apertus 2 prototype recipe from
[PR #34](https://github.com/swiss-ai/apertus-omni-tokenizer/pull/34) remains in
the history of `omnitok/versions/apertus_2.py`.

## Building the Apertus 1.5 tokenizer

```bash
# From the pinned canonical base on the Hub
python -m omnitok.cli build-apertus-1p5 --output-path ./Apertus_1p5

# Offline, from the checked-in copy of the base
python -m omnitok.cli build-apertus-1p5 --output-path ./Apertus_1p5 --base-tokenizer tokenizers/Apertus_1_base
```

Or as a library:

```python
from omnitok import build_apertus_1p5

build_apertus_1p5("./Apertus_1p5")
```

The output is byte-identical to the canonical artifact (the manifest under
`validation/` pins the md5 of every file, and `tests/test_registry.py`
rebuilds and checks this). The full recipe — reasoning-delimiter renames, tool
output tokens, normalizer alias/cleanup rules, vision + audio modalities, chat
template, SFT sequences — is encoded in `omnitok/apertus.py`; that module's
docstring is the audit trail of every delta between Apertus 1 and 1.5,
including the deliberate repairs over the originally released RC artifact.
Do not hand-edit tokenizer files: change the recipe (or the chat template
under `chat_templates/Apertus_1p5/`) and rebuild.

## Token layout (Apertus 1.5)

```
[0 .. base-1]           text tokens (unchanged)
[base .. base+199]      200 reserved OMNI slots
  slot 0                  boundary marker (never renamed)
  slots 1-7               vision structure tokens
  slots 8-15              audio structure tokens
  slots 16-199            reserved for future modalities
[base+200 .. ]          content tokens (appended per modality in order added)
```

A base that ships a `<SPECIAL_*>` reserve pool, like Apertus 2's, uses
`add_modality_in_place` instead: structure tokens are renamed in place at low
ids inside the text vocab and only content tokens are appended. The shipped
Apertus 2 tokenizers are text-only.
See [docs/omnimodal_config.md](docs/omnimodal_config.md) for both schemes.

## Reserved slot allocation (Apertus 1.5)

| Slots | Modality | Tokens |
|-------|----------|--------|
| 0 | -- | `<\|RESERVED_OMNI_000\|>` (boundary) |
| 1-7 | Vision | img_start, img_end, img_token_start, img_end_of_row, img_end_of_frame, img_generation_start, image |
| 8-15 | Audio | audio_start, audio_end, stt_transcribe, stt_continue, tts_continue, audio, stt_translate, audio_annotate |
| 16-199 | -- | Reserved |

## Token aliases

`<image>`/`<|image|>` and `<audio>`/`<|audio|>` each encode to the same token
ID, handled by the tokenizer's normalizer -- no manual `.replace()` needed in
data loaders. `<think>`/`</think>` are likewise aliased to
`<|inner_prefix|>`/`<|inner_suffix|>` (ids 32/33).

(The originally released RC artifact shipped the `<audio>` alias broken:
`<|audio|>` was left `normalized: false`, so bare `<audio>` encoded as plain
text. The canonical artifact repairs this, and the build produces the
repaired form.)

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

# Repair mode: download the canonical copy of each missing/mismatched file,
# back the old one up as <file>.bak, then re-validate
#   | bash -s -- --fix

# From a local checkout
bash validate_model.sh /path/to/served/model
bash validate_model.sh --fix /path/to/served/model
```

```
Validating Apertus_1p5 tokenizer in: /path/to/served/model

✔ chat_template.jinja
✔ tokenizer.json
✔ tokenizer_config.json
✔ special_tokens_map.json

OK: Apertus_1p5 tokenizer matches the canonical checksums.
```

Arguments: `validate_model.sh [--fix] [MODEL_PATH] [MODEL_NAME]` -- `MODEL_PATH`
defaults to the current directory, `MODEL_NAME` defaults to `Apertus_1p5` (pass
`Apertus_1` to validate the 1.0 tokenizer). With `--fix`, each missing or
mismatched manifest file is re-downloaded from this repo (verified against the
manifest md5 before installing; the existing file is saved as `<file>.bak`,
never overwriting earlier backups — repeat runs write `<file>.bak.1`, `.bak.2`,
...) and the validation is re-run. `generation_config.json` is checked field-level and
is not auto-fixed. The canonical checksums are regenerated by
`validation/gen_checksums.sh` and kept in sync by CI.

## Structure

```
apertus-omni-tokenizer/
├── README.md
├── pyproject.toml
├── validate_model.sh        # verify a served model dir matches canonical md5s
├── omnitok/
│   ├── __init__.py      # public API exports
│   ├── registry.py      # every tokenizer: origin or parent, pins, recipe
│   ├── apertus.py       # build_apertus_1p5() -- the Apertus 1 -> 1.5 recipe
│   ├── modalities.py    # ModalityConfig dataclass, built-in VISION/AUDIO configs
│   ├── builder.py       # add_modality() / add_modality_in_place() -- modality engine
│   ├── versions/
│   │   └── apertus_2.py # text-only Apertus 2 base/instruct recipe
│   ├── instruct.py      # create_instruct_tokenizer() -- chat template + SFT sequences
│   ├── io.py            # low-level file I/O (rename, alias, detect, save)
│   └── cli.py           # CLI wrapper (python -m omnitok.cli)
├── tests/
│   ├── conftest.py             # shared fixtures
│   ├── test_alias.py           # token alias tests (<image> == <|image|>)
│   ├── test_registry.py        # pins, byte-exact rebuilds, no unregistered dirs
│   ├── test_apertus_recipe.py  # the 1.5 recipe rejects the wrong base
│   ├── test_instruct.py        # chat template handling
│   ├── tokenizer_factory.py    # shared synthetic-tokenizer factory
│   ├── test_apertus_2_recipe.py # text base, instruct controls, build guards
│   ├── test_apertus_2_validation.py # deployment EOS/pad checks
│   ├── test_builder.py         # add_modality tests
│   ├── test_chat_template.py   # add chat template test
│   ├── test_task_tokens.py     # task token contract tests
│   └── test_tokenizers.py      # checked-in tokenizers load + special-token IDs
├── tokenizers/
│   ├── Apertus_1_base/       # Apertus 1 base model tokenizer (parent of 1.5)
│   ├── Apertus_1/            # Instructed tokenizer used for Apertus 1.0
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── Apertus_1p5/          # Instructed tokenizer used for Apertus 1.5
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── Apertus_2/            # Byte-identical upstream text base
│   └── Apertus_2_instruct/   # Conversation controls, no multimodal extension
├── chat_templates/
│   ├── Apertus_1/            # Chat template used for Apertus 1.0
│   │   └── chat_template.jinja
│   └── Apertus_1p5/          # Chat template used for Apertus 1.5
│       └── chat_template.jinja
└── validation/
    ├── gen_checksums.sh      # regenerate the manifests below
    ├── Apertus_1.md5         # canonical md5s for the 1.0 tokenizer
    ├── Apertus_1p5.md5       # canonical md5s for the 1.5 tokenizer
    ├── Apertus_2.md5         # canonical md5s for the text base
    └── Apertus_2_instruct.md5 # instruct tokenizer and library binding
```

Adding a new modality = one new `ModalityConfig` entry in `modalities.py`.

## Author

Yixuan Xu (yixuan.xu@ai.ethz.ch)
