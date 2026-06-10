# omnitok

Extend Apertus / LLaMA-3 text tokenizers with vision and audio modality tokens.

## Usage

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

## Known codebook sizes

| Tokenizer | Modality | Codebook Size |
|-----------|----------|---------------|
| Emu3.5 (IBQ) | Vision | 131,072 |
| Emu3 | Vision | 32,768 |
| WavTokenizer | Audio | 4,096 |

## Structure

```
apertus-omni-tokenizer/
├── README.md
├── pyproject.toml
├── omnitok/
│   ├── __init__.py      # public API exports
│   ├── modalities.py    # ModalityConfig dataclass, built-in VISION/AUDIO configs
│   ├── builder.py       # add_modality() -- the main engine
│   ├── instruct.py      # create_instruct_tokenizer() -- chat template + SFT sequences
│   ├── io.py            # low-level file I/O (rename, alias, detect, save)
│   └── cli.py           # CLI wrapper (python -m omnitok.cli)
└── tests/
│   ├── conftest.py           # shared fixtures
│   ├── test_alias.py         # token alias tests (<image> == <|image|>)
│   ├── test_builder.py       # add_modality tests
│   ├── test_chat_template.py # add chat template test
│   └── test_task_tokens.py   # task token contract tests
└── examples/
│   └── rename_tool_tokens.py # Script used to add new special tokens in tool calling parsing
└── chat_templates/
    ├── Apertus_1     # Chat template used for Apertus 1.0 
    │   └── chat_template.jinja
    └── Apertus_1p5   # Chat template used for Apertus 1.5
        └── chat_template.jinja
```

Adding a new modality = one new `ModalityConfig` entry in `modalities.py`.

## Author

Yixuan Xu (yixuan.xu@ai.ethz.ch)
