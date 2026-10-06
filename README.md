# omnitok

A repository and Python library that is the central place for the Apertus
tokenizers: each model's canonical tokenizer files, the recipes that build them
and the checks that verify them. A tokenizer is either imported unchanged from
upstream (e.g. the Apertus 2 text base, from
[apertus-tokenizer-development](https://github.com/swiss-ai/apertus-tokenizer-development))
or derived by a recipe from a tokenizer kept here (e.g. Apertus 1.5 from the
Apertus 1 base).

## Contents

- [Supported versions](#supported-versions)
- [Repository structure](#repository-structure)
- [Validating a deployed model](#validating-a-deployed-model)
- [Recipes and the registry](#recipes-and-the-registry)
  - [Saving](#saving)
- [Apertus 1.5](#apertus-15)
- [Apertus 2](#apertus-2)
- [Authors](#authors)

## Supported versions

This package needs Python ≥ 3.11. Loading the committed files needs only
transformers and tokenizers, not this package; CI tests loading on Python 3.11
to 3.14.

| Use                                                                                       | transformers               | tokenizers       | Declared in                                                   |
|-------------------------------------------------------------------------------------------|----------------------------|------------------|---------------------------------------------------------------|
| Load the committed tokenizers                                                             | ≥ 4.48.2                   | ≥ 0.21.0         | CI (training stack 4.48.2 / 0.21.0)                           |
| Build and save with this package                                                          | ≥ 4.56, ≠ 4.57.2, ≠ 4.57.3 | ≥ 0.22.2         | `pyproject.toml`                                              |
| `Apertus_2_instruct` through [apertus-common](https://github.com/swiss-ai/apertus-common) | ≥ 5.16                     | ≥ 0.23.2, < 0.24 | apertus-common runtime dependencies |

**Slow and broken versions.** Below tokenizers 0.22.2, the Apertus 1.5
tokenizer, with its 135k added tokens, takes 1–3 min to load instead of about
3 s. transformers 4.56 is the first release that accepts tokenizers 0.22.2,
and 5.16 the first that accepts tokenizers 0.23. transformers 4.57.2 fails on
directories that hold a model `config.json`, such as Hub snapshots; 4.57.3
queries the Hub on every load by repo id, even with `HF_HUB_OFFLINE=1`, and
fails without network access.

**How CI checks version compatibility.** The full matrix runs on every pull
request, push to `main`, and Monday at 04:30 UTC, resolving floating versions
afresh. The rows are defined in [.github/workflows/ci.yml](.github/workflows/ci.yml).

- **Build/save (4 rows):** lower bounds, latest transformers 4.x, transformers
  5.x with tokenizers 0.22, and latest releases, on Python 3.11 to 3.14 in
  that order. Every row runs the full suite, including byte-identical
  rebuilds and compatible saves of **both `Apertus_1p5` and
  `Apertus_2_instruct`**.
- **Load (6 rows):** every registered artifact is loaded in those four rows
  plus the training (4.48.2 / 0.21.0) and serving (4.51.1 / 0.21.1) stacks.
  Checks cover token ids, config values, BOS/EOS insertion and chat templates.
- **Registry:** imported files must match their pinned hashes; derived outputs
  and deployment manifests must match the committed files. Unregistered
  artifact directories fail CI.

Identical builds plus loading the canonical files verify compatibility across
the matrix without running every writer/reader pair. This covers tokenizer
artifacts; `apertus-common` runs separate conversation round-trip and harness
checks within its narrower runtime range above.

**Saving and deploying.**

- Save only through the recipes ([Saving](#saving)). Their default
  `compatible` mode is what CI checks above.
- `build-apertus-1p5 --save-mode current_version` writes the installed
  transformers' own format, for a consumer pinned to that version. CI only
  checks it saved and loaded by the same version; other major versions may not
  load it.
- Deploy by copying the files. A plain `save_pretrained` re-save breaks them:
  - on transformers 5.x, the class becomes `TokenizersBackend`, which 4.x
    cannot load; native saves also change the canonical artifact bytes;
  - on 4.x, the Apertus 1.5 config grows to 25 MB and fails validation.
- Repair a re-saved deployment with `validate_model.sh --fix`, not by hand:
  restoring only `tokenizer_class` loads on 4.x but loses the multimodal role
  tokens, e.g. `tokenizer.eoa_token`.

## Repository structure

```
apertus-omni-tokenizer/
├── README.md
├── pyproject.toml
├── validate_model.sh               # check a served model dir against validation/
├── .github/workflows/ci.yml        # version matrix, load-only rows, manifest check
├── omnitok/                        # the library
│   ├── __init__.py                 # public API exports
│   ├── registry.py                 # every tokenizer: origin or parent, pins, recipe
│   ├── recipes/                    # per-model build recipes
│   │   ├── apertus_1p5.py          # Apertus 1 -> 1.5 recipe (build_apertus_1p5)
│   │   └── apertus_2.py            # Apertus 2 base import + instruct recipe
│   ├── builder.py                  # add_modality() / add_modality_in_place()
│   ├── modalities.py               # ModalityConfig, built-in VISION/AUDIO configs
│   ├── instruct.py                 # create_instruct_tokenizer(): chat template + SFT
│   ├── io.py                       # low-level file I/O (rename, alias, save, detect)
│   └── cli.py                      # python -m omnitok.cli <subcommand>
├── tokenizers/                     # committed tokenizers: tokenizer.json,
│   │                               # tokenizer_config.json, special_tokens_map.json
│   ├── Apertus_1_base/             # Apertus 1 base model tokenizer, parent of 1.5
│   ├── Apertus_1/                  # Apertus 1.0 instruct tokenizer
│   ├── Apertus_1p5/                # Apertus 1.5 tokenizer
│   ├── Apertus_2/                  # Apertus 2 text base, imported unchanged
│   └── Apertus_2_instruct/         # + generation_config.json
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
    ├── test_registry.py            # pins, byte-exact rebuilds, re-saves, manifests
    ├── test_save.py                # save_tokenizer_files modes
    ├── test_load.py                # committed tokenizers load; every config value reachable
    ├── test_tokenizers.py          # committed tokenizers load; pinned special-token ids
    ├── test_apertus_recipe.py      # 1.5: wrong base rejected, rebuild from the Apertus_1 mirror
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

`validate_model.sh` checks that a served model directory ships the exact
canonical tokenizer files: it compares their md5s with the manifests under
`validation/` and exits non-zero on any mismatch.

```bash
# Remote (no checkout needed): cd into the model dir, then validate it
cd /path/to/served/model
curl -fsSL https://raw.githubusercontent.com/swiss-ai/apertus-omni-tokenizer/main/validate_model.sh \
  | bash

# Another tokenizer:  | bash -s -- . Apertus_1
# Repair mode:        | bash -s -- --fix

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
✔ generation_config.json (eos_token_id includes: 2 68 72)

OK: Apertus_1p5 tokenizer matches the canonical checksums.
```

Usage: `validate_model.sh [--fix] [MODEL_PATH] [MODEL_NAME]`

- `MODEL_PATH`: the model directory; default: the current directory.
- `MODEL_NAME`: `Apertus_1p5` (default), `Apertus_1`, `Apertus_2` or
  `Apertus_2_instruct`.
- `--fix`: re-downloads each missing or mismatched file, checks its md5, keeps
  the old one as `<file>.bak` (`.bak.1`, … on reruns) and validates again.
- `generation_config.json` (only `Apertus_1p5` and `Apertus_2_instruct`) must
  exist and is checked per field, never fixed: for `Apertus_1p5`,
  `eos_token_id` must include 2, 68 and 72; for `Apertus_2_instruct`, it must be
  exactly `[12, 2]` (either order), with `pad_token_id: 3`.
- Manifests and fixes come from GitHub `main` (override with `BASE_URL` and
  `BRANCH`); a checkout's own copies are the fallback.
  `validation/gen_checksums.sh` regenerates the manifests, and CI keeps them in
  sync.

## Recipes and the registry

`omnitok/registry.py` lists every directory under `tokenizers/` and where its
files come from. To add a tokenizer:

1. Register it as one of:
   - `Imported`: upstream files copied unchanged into `tokenizers/<name>/`,
     with the origin (Hub repo or git repository, pinned revision) and the
     SHA-256 of every file. No recipe.
   - `Derived`: built by a recipe from a registered parent, e.g. `Apertus_1p5`
     from `Apertus_1_base`.
2. For a derived tokenizer, write a module in `omnitok/recipes/` whose build
   function takes `(parent_dir, output_dir)` and ends with
   `save_tokenizer_files` ([Saving](#saving)). Its docstring records what it
   changes from the parent and why (see `omnitok/recipes/apertus_1p5.py`).
   Commit its output.
3. To validate deployments, add the files to `validation/gen_checksums.sh` and
   run it.
4. Run `pytest tests/test_registry.py`: it checks every pin, rebuilds every
   derived tokenizer byte-for-byte, checks the manifests and fails on
   unregistered directories.

Recipes are built from this toolbox:

| Tool                                                                             | Purpose                                                                                          |
|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `add_modality` (`builder.py`)                                                    | append reserved structure slots and content tokens for a modality (the Apertus 1.5 scheme)       |
| `add_modality_in_place` (`builder.py`)                                           | rename a base's pre-baked reserve slots and append content tokens (planned for Apertus 2.5)      |
| `ModalityConfig`, `VISION`, `AUDIO` (`modalities.py`)                            | a modality's structure tokens and content-token format; register new ones in `MODALITY_REGISTRY` |
| `create_instruct_tokenizer` (`instruct.py`)                                      | add a chat template and SFT begin/end sequences                                                  |
| `rename_reserved_tokens`, `add_token_alias`, `mark_tokens_non_special` (`io.py`) | rename tokens, alias literals through the normalizer, mark tokens non-special                    |
| `save_tokenizer_files`, `load_backend_state`, `rewrite_backend_state` (`io.py`)  | write the final files ([Saving](#saving)), read or edit a saved `tokenizer.json`                 |

### Saving

Every recipe ends with `save_tokenizer_files(output, state, config, mode=...)`
(instead of `save_pretrained`, whose output depends on the installed
transformers). It takes the final `tokenizer.json` content (`state`, e.g. from
`load_backend_state`) and config, plus the config keys to keep and any
overrides:

- `compatible` (default): `tokenizer.json` through the tokenizers library, the
  config as sorted JSON with only the kept keys and
  `tokenizer_class: PreTrainedTokenizerFast`, a special-tokens map with only
  `bos`/`eos`/`pad`/`unk`, and the chat template in `chat_template.jinja`.
  Committed derived tokenizers always use this mode.
- `current_version`: the installed transformers' `save_pretrained` (when to use
  it: [Supported versions](#supported-versions)). A recipe may refuse it, as
  Apertus 2 instruct does, keeping its recipe output canonical across versions.

Other files, such as `generation_config.json`, the recipe writes as
plain JSON.

## Apertus 1.5

`tokenizers/Apertus_1p5` is the tokenizer of the
[apertus-ai/Apertus-v1.5-8B](https://huggingface.co/apertus-ai/Apertus-v1.5-8B)
release. `omnitok/recipes/apertus_1p5.py` builds it from the Apertus 1 base
([swiss-ai/Apertus-8B-2509](https://huggingface.co/swiss-ai/Apertus-8B-2509)),
and its docstring records every change from Apertus 1. Don't hand-edit the
files: change the recipe or `chat_templates/Apertus_1p5/` and rebuild.

```bash
# Base from the Hub at the pinned revision; add
# --base-tokenizer tokenizers/Apertus_1_base to build offline, and
# --save-mode current_version for the installed transformers' own format.
python -m omnitok.cli build-apertus-1p5 --output-path ./Apertus_1p5
```

From Python: `omnitok.build_apertus_1p5("./Apertus_1p5")`.

| IDs             | Tokens                                                                                                                    |
|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| 0–131,071       | text vocabulary                                                                                                           |
| 131,072         | `<\|RESERVED_OMNI_000\|>` (boundary marker)                                                                               |
| 131,073–131,079 | vision structure: img_start, img_end, img_token_start, img_end_of_row, img_end_of_frame, img_generation_start, image      |
| 131,080–131,087 | audio structure: audio_start, audio_end, stt_transcribe, stt_continue, tts_continue, audio, stt_translate, audio_annotate |
| 131,088–131,271 | reserved for future modalities                                                                                            |
| 131,272–262,343 | vision content (Emu3.5 codebook, 131,072)                                                                                 |
| 262,344–266,439 | audio content (WavTokenizer codebook, 4,096)                                                                              |

`<image>`, `<audio>`, `<think>` and `</think>` encode to the same ids as
`<|image|>`, `<|audio|>`, `<|inner_prefix|>` and `<|inner_suffix|>`.

## Apertus 2

| Tokenizer            | Contents                                                                                                                          |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `Apertus_2`          | 200,064-token text base, imported unchanged from [`preliminary_mul_200k`][pm200k]; keeps its NFC normalizer and automatic BOS/EOS |
| `Apertus_2_instruct` | compact special-token layout: assigned ids 0–12, reserves 13–123; no normalizer or automatic BOS/EOS                              |

[pm200k]: https://github.com/swiss-ai/apertus-tokenizer-development/tree/28ad57a2757f6f72edb0def57ee725b6812f2df3/preliminary_mul_200k

The instruct recipe removes the legacy role, reasoning, tool, reflection,
and image/audio spellings, replaces the assistant delimiters with
`<|out|>` / `<|/out|>` at their original ids 10/11, and renames the existing
padding token at id 3.
Its complete assigned layout is:

| IDs    | Tokens (in order)                                                             |
|--------|-------------------------------------------------------------------------------|
| 0–3    | `<unk>`, `<s>`, `</s>`, `<\|pad\|>`                                           |
| 4–6    | `<iban-pii>`, `<email-pii>`, `<ip-pii>`                                       |
| 7–12   | `<\|in\|>`, `<\|/in\|>`, `<\|hdr\|>`, `<\|out\|>`, `<\|/out\|>`, `<\|wait\|>` |
| 13–123 | `<SPECIAL_13>` through `<SPECIAL_123>`                                        |

All ids 0–123 remain special. Ordinary vocabulary ids 124–200,063 and BPE
merges are unchanged. Existing checkpoints and tokenized data using the
previous layout need remapping before using this artifact.

BOS/EOS placement is owned entirely by the chat template or conversation
encoder (`apertus-common` here): the tokenizer has no post-processor and sets
`add_bos_token` and `add_eos_token` to false. The role metadata (`<s>` for BOS,
`<|wait|>` for EOS) does not insert tokens. Generation stopping is separate:
`generation_config.json` specifies `eos_token_id: [12, 2]` (`<|wait|>`, `</s>`)
and `pad_token_id: 3`; deployment validation requires those two EOS ids.
The current apertus-common parser uses wait as its terminator; support for
`</s>` as a conversation terminator remains follow-up work.

There is no chat template. Load with HF `AutoTokenizer`, then pass the tokenizer
to `apertus_common.load_encoding(tokenizer)` to encode conversations. The config
names six `extra_special_tokens`: `input_start_token`, `input_end_token`,
`output_start_token`, `output_end_token`, `header_end_token`, and `wait_token`;
padding uses the standard `pad_token`. IDs come from those attributes, with no
sidecar binding file. Provenance stays in this repository's registry and checksums.
The encoder disables special-token matching for content and excludes **all**
backend special IDs, including reserved slots. Plain HF `encode` requires
`split_special_tokens=True` as well as `add_special_tokens=False` for this behavior.
Deploy the files by copying them ([Supported versions](#supported-versions)).

```bash
# base: a checkout of the pinned upstream directory (Git LFS: run git lfs pull)
python -m omnitok.cli build-apertus-2 base \
  --input-tokenizer /path/to/preliminary_mul_200k --output-path ./Apertus_2
python -m omnitok.cli build-apertus-2 instruct \
  --input-tokenizer tokenizers/Apertus_2 --output-path ./Apertus_2_instruct
```

Both commands check the input against its pinned hashes and need an empty or
absent output directory.

Multimodal support is planned for Apertus 2.5. Its base ships a `<SPECIAL_*>`
reserve pool, so it will use `add_modality_in_place` (see
[docs/omnimodal_config.md](docs/omnimodal_config.md)), and its recipe should
expose the multimodal role tokens like 1.5's `EXTRA_SPECIAL_TOKENS`.

## Authors

- Yixuan Xu (yixuan.xu@ai.ethz.ch)
- Raphael Kreft (rkreft@ai.ethz.ch)
