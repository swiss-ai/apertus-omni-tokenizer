# omnitok

A repository and Python library that is the central place for Apertus
tokenizers. It stores the canonical tokenizer artifact of each Apertus model,
the logic to verify those artifacts, and the recipes that build them. A recipe
starts either from a base tokenizer kept in this repo (Apertus 1.5 is built
from the Apertus 1 base) or from an external artifact (the Apertus 2 text
tokenizer comes from
[apertus-tokenizer-development](https://github.com/swiss-ai/apertus-tokenizer-development)).

## Contents

- [Supported versions](#supported-versions)
- [Repository structure](#repository-structure)
- [Validating a deployed model](#validating-a-deployed-model)
- [Recipes and the registry](#recipes-and-the-registry)
- [Apertus 1.5](#apertus-15)
- [Apertus 2](#apertus-2)
- [Authors](#authors)

## Supported versions

| Use                                         | transformers               | tokenizers       | Why                                                                                                                        |
|---------------------------------------------|----------------------------|------------------|----------------------------------------------------------------------------------------------------------------------------|
| Load the committed tokenizers               | tested ≥ 4.48.2            | tested ≥ 0.21.0  | all load and encode correctly; below tokenizers 0.22.2 Apertus 1.5 takes 1–3 min to load instead of ~3 s                   |
| Build and save with this package            | ≥ 4.56, ≠ 4.57.2, ≠ 4.57.3 | ≥ 0.22.2         | declared in `pyproject.toml`; 4.56 is the first transformers that accepts tokenizers 0.22.2                                |
| `Apertus_2_instruct` through apertus-common | ≥ 5.16                     | ≥ 0.23.2, < 0.24 | declared in its `apertus_encoding.json`, checked by apertus-common on load; 5.16 is the first transformers that accepts it |

transformers 4.57.2 fails on directories that hold a model `config.json`, such
as Hub snapshots. 4.57.3 queries the Hub on every load by repo id, even with
`HF_HUB_OFFLINE=1`, and fails without network access.

No upper bound is declared. On every pull request and weekly, CI runs the whole
suite, including the byte-for-byte rebuilds, on four combinations: the lower
bounds (Python 3.9, transformers 4.56.0, tokenizers 0.22.2), the newest
transformers 4.x, transformers 5.x with tokenizers 0.22, and the newest
releases. A release that changes the built bytes fails CI instead of shipping
different artifacts.

Save only through the recipes. A plain `save_pretrained` on transformers 5.x
writes `tokenizer_class: TokenizersBackend`, which transformers 4.x cannot load,
and adds a post-processor to `Apertus_2_instruct`, which apertus-common rejects;
on 4.x it inflates the Apertus 1.5 config to 25 MB, which fails validation.
Deploy artifacts by copying their files, and repair a re-saved deployment with
`validate_model.sh --fix` rather than by hand: restoring only `tokenizer_class`
loads on 4.x but loses the role tokens such as `tokenizer.eoa_token`.

## Repository structure

```
apertus-omni-tokenizer/
├── README.md
├── pyproject.toml
├── validate_model.sh               # check a served model dir against validation/
├── omnitok/                        # the library
│   ├── __init__.py                 # public API exports
│   ├── registry.py                 # every tokenizer: origin or parent, pins, recipe
│   ├── recipes/                    # per-model build recipes
│   │   ├── apertus_1p5.py          # Apertus 1 -> 1.5 recipe (build_apertus_1p5)
│   │   └── apertus_2.py            # Apertus 2 base import + instruct recipe
│   ├── builder.py                  # add_modality() / add_modality_in_place()
│   ├── modalities.py               # ModalityConfig, built-in VISION/AUDIO configs
│   ├── instruct.py                 # create_instruct_tokenizer(): chat template + SFT
│   ├── io.py                       # low-level file I/O (rename, alias, finalize, detect)
│   └── cli.py                      # python -m omnitok.cli <subcommand>
├── tokenizers/                     # committed artifacts: tokenizer.json,
│   │                               # tokenizer_config.json, special_tokens_map.json
│   ├── Apertus_1_base/             # Apertus 1 base model tokenizer, parent of 1.5
│   ├── Apertus_1/                  # Apertus 1.0 instruct tokenizer
│   ├── Apertus_1p5/                # Apertus 1.5 tokenizer
│   ├── Apertus_2/                  # Apertus 2 text base, imported unchanged
│   └── Apertus_2_instruct/         # + generation_config.json, apertus_encoding.json
├── chat_templates/
│   ├── Apertus_1/chat_template.jinja
│   └── Apertus_1p5/chat_template.jinja
├── validation/
│   ├── gen_checksums.sh            # regenerate the manifests below
│   ├── Apertus_1.md5               # canonical md5s per served tokenizer
│   ├── Apertus_1p5.md5
│   ├── Apertus_2.md5
│   └── Apertus_2_instruct.md5
├── docs/
│   └── omnimodal_config.md         # omnimodal_config contract, both allocation schemes
├── parsers/vllm/                   # vLLM tool and reasoning parsers for Apertus 1.x
└── tests/
    ├── test_registry.py            # pins, byte-exact rebuilds, no unregistered dirs
    ├── test_tokenizers.py          # committed tokenizers load; pinned special-token ids
    ├── test_apertus_recipe.py      # the 1.5 recipe rejects the wrong base
    ├── test_apertus_2_recipe.py    # Apertus 2 base, instruct controls, build guards
    ├── test_apertus_2_validation.py # validate_model.sh generation-config checks
    ├── test_builder.py             # add_modality
    ├── test_builder_in_place.py    # add_modality_in_place
    ├── test_alias.py               # token aliases (<image> == <|image|>)
    ├── test_instruct.py            # chat template handling
    ├── test_chat_template.py       # chat template rendering
    ├── test_task_tokens.py         # task token contract
    ├── test_parsers.py             # vLLM parser plugins
    ├── conftest.py                 # shared fixtures
    ├── tokenizer_factory.py        # synthetic test tokenizers
    └── examples/                   # chat template fixtures
```

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
`Apertus_1`, `Apertus_2` or `Apertus_2_instruct` to validate another tokenizer). With `--fix`, each missing or
mismatched manifest file is re-downloaded from this repo (verified against the
manifest md5 before installing; the existing file is saved as `<file>.bak`,
never overwriting earlier backups — repeat runs write `<file>.bak.1`, `.bak.2`,
...) and the validation is re-run. `generation_config.json` is checked field-level and
is not auto-fixed. The canonical checksums are regenerated by
`validation/gen_checksums.sh` and kept in sync by CI.

## Recipes and the registry

Every directory under `tokenizers/` has an entry in `omnitok/registry.py`
that records where its files come from. To add a model's tokenizer, decide
which kind it is:

- **Imported:** upstream files used unchanged, e.g. `Apertus_2` from
  apertus-tokenizer-development. Copy them into `tokenizers/<name>/` and add
  an `Imported` entry with the origin (Hub repo or git repository, pinned
  revision) and the SHA-256 of every file. No recipe is needed.
- **Derived:** files built from another registered tokenizer, e.g.
  `Apertus_1p5` from `Apertus_1_base`. Write a recipe, add a `Derived` entry
  naming the parent and the recipe, and commit the recipe's output.

A recipe is a module in `omnitok/recipes/` whose build function takes
`(parent_dir, output_dir)`. The library is the toolbox it is built from:

| Tool                                                                                                          | Purpose                                                                                     |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `add_modality` (`builder.py`)                                                                                 | append reserved structure slots and content tokens for a modality (the Apertus 1.5 scheme)  |
| `add_modality_in_place` (`builder.py`)                                                                        | rename a base's pre-baked reserve slots and append content tokens (planned for Apertus 2.5) |
| `ModalityConfig`, `VISION`, `AUDIO` (`modalities.py`)                                                         | a modality's structure tokens and content-token format                                      |
| `create_instruct_tokenizer` (`instruct.py`)                                                                   | add a chat template and SFT begin/end sequences                                             |
| `rename_reserved_tokens`, `add_token_alias`, `mark_tokens_non_special`, `finalize_tokenizer_config` (`io.py`) | rename tokens, alias literals through the normalizer, write a deterministic config          |

Each recipe documents itself: its module docstring records what it changes
relative to the parent and why, including deliberate quirks (see
`omnitok/recipes/apertus_1p5.py`). The model sections below only show how to
build.

Then add the files to `validation/gen_checksums.sh` if deployments should be
validated, and run `pytest tests/test_registry.py`. It checks every pin,
rebuilds every derived tokenizer byte-for-byte, checks the manifests, and
fails on unregistered directories.

Adding a new modality = one new `ModalityConfig` entry in `modalities.py`.

## Apertus 1.5

`tokenizers/Apertus_1p5` is the tokenizer of the
[apertus-ai/Apertus-v1.5-8B](https://huggingface.co/apertus-ai/Apertus-v1.5-8B)
release. `omnitok/recipes/apertus_1p5.py` builds it from the Apertus 1 base
([swiss-ai/Apertus-8B-2509](https://huggingface.co/swiss-ai/Apertus-8B-2509)),
and its docstring records every change from Apertus 1. Don't hand-edit the
files: change the recipe or `chat_templates/Apertus_1p5/` and rebuild.

```bash
# Base from the Hub at the pinned revision; add
# --base-tokenizer tokenizers/Apertus_1_base to build offline.
python -m omnitok.cli build-apertus-1p5 --output-path ./Apertus_1p5
```

From Python: `omnitok.build_apertus_1p5("./Apertus_1p5")`.

| IDs             | Tokens                                                                                                                    |
|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| 0–131,071       | text vocabulary                                                                                                           |
| 131,072         | `<\|RESERVED_OMNI_000\|>` (boundary)                                                                                      |
| 131,073–131,079 | vision structure: img_start, img_end, img_token_start, img_end_of_row, img_end_of_frame, img_generation_start, image      |
| 131,080–131,087 | audio structure: audio_start, audio_end, stt_transcribe, stt_continue, tts_continue, audio, stt_translate, audio_annotate |
| 131,088–131,271 | reserved for future modalities                                                                                            |
| 131,272–262,343 | vision content (Emu3.5 codebook, 131,072)                                                                                 |
| 262,344–266,439 | audio content (WavTokenizer codebook, 4,096)                                                                              |

`<image>`, `<audio>`, `<think>` and `</think>` encode to the same ids as
`<|image|>`, `<|audio|>`, `<|inner_prefix|>` and `<|inner_suffix|>`, so data
loaders need no `.replace()`.

## Apertus 2

| Tokenizer            | Contents                                                                                                                                                                                                                                                |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Apertus_2`          | 200,064-token text base, imported unchanged from [`preliminary_mul_200k`](https://github.com/swiss-ai/apertus-tokenizer-development/tree/28ad57a2757f6f72edb0def57ee725b6812f2df3/preliminary_mul_200k); keeps its NFC normalizer and automatic BOS/EOS |
| `Apertus_2_instruct` | the base with ids 40–46 renamed to `<\|in\|>`, `<\|/in\|>`, `<\|out\|>`, `<\|/out\|>`, `<\|hdr\|>`, `<\|wait\|>` (EOS), `<\|pad\|>`; no normalizer or automatic BOS/EOS                                                                                 |

There is no chat template. Encode conversations with
[apertus-common](https://github.com/swiss-ai/apertus-common), which reads the
binding in `apertus_encoding.json` and requires `tokenizers>=0.23.2,<0.24`.
Plain Hugging Face tokenization would also turn control spellings inside
message text, such as a literal `<|out|>`, into control tokens. Deploy the
instruct files by copying them: a transformers 5.x `save_pretrained` adds a
post-processor to `tokenizer.json`, which then fails validation and the
binding.

```bash
# base: a checkout of the pinned upstream directory (Git LFS: run git lfs pull)
python -m omnitok.cli build-apertus-2 base \
  --input-tokenizer /path/to/preliminary_mul_200k --output-path ./Apertus_2
python -m omnitok.cli build-apertus-2 instruct \
  --input-tokenizer tokenizers/Apertus_2 --output-path ./Apertus_2_instruct
```

Both commands check the input against its pinned hashes and need an empty
output directory.

Multimodal support is planned for Apertus 2.5. Its base ships a `<SPECIAL_*>`
reserve pool, so it will use `add_modality_in_place` (see
[docs/omnimodal_config.md](docs/omnimodal_config.md)), and its recipe should
expose the multimodal role tokens like 1.5's `EXTRA_SPECIAL_TOKENS`.

## Authors

- Yixuan Xu (yixuan.xu@ai.ethz.ch)
- Raphael Kreft (rkreft@ai.ethz.ch)
