# omnimodal_config

`omnimodal_config` in `tokenizer_config.json` is the metadata contract
between built omni-tokenizers and their consumers
(Megatron-LM, serving, data pipelines).
It is derived from the tokenizer vocabulary at build time,
and verified on every write path.
The shipped Apertus 2 tokenizers are text-only and carry none;
the in-place values below come from the multimodal Apertus 2 prototype (#34),
the scheme planned for Apertus 2.5.

## Fields

```json
{
  "base_vocab_size": 200064,
  "omnimodal_config": {
    "omni_special_token_offset": 200064,
    "modalities": [
      {
        "name": "vision",
        "offset": 200064,
        "vocab_size": 131072,
        "start_token": 27,
        "end_token": 28,
        "structure_token_ids": {"<|img_start|>": 27, "...": 0, "<|image|>": 18}
      },
      {"name": "audio", "offset": 331136, "vocab_size": 4096, "...": 0}
    ]
  }
}
```

| Field | Meaning |
| --- | --- |
| `base_vocab_size` | text-only vocab size (top-level key, next to `omnimodal_config`) |
| `omni_special_token_offset` | equal to `base_vocab_size`; kept for existing readers |
| `modalities[].offset` | id of the modality's first content token |
| `modalities[].vocab_size` | number of content tokens (codebook size) |
| `modalities[].start_token` / `end_token` | ids of the span delimiters |
| `modalities[].structure_token_ids` | token name -> id for every structure token, including reused placeholders |

Modalities are sorted by `offset`: the list follows the id order of the
content ranges and the serialized config is deterministic.
Look modalities up by `name` rather than by position.

## Contracts

- **Content ids are contiguous**: `id = offset + index`.
  The build verifies this for every modality and fails on gaps,
  so consumers may rely on the arithmetic.
- **Structure tokens resolve by name**, via `structure_token_ids`
  or the tokenizer itself. Do not assume geometry:
  Apertus 1.5's structure tokens sit in a contiguous block above
  `base_vocab_size`, in-place builds on the Apertus 2 base put them at low
  ids inside the base vocab (reused placeholders at 18/19, renamed pool slots
  at 27-39).
- **In-place recipes may drop the post-processor**:
  with `strip_post_processor=True` the artifact has none,
  so encoding is exact (`add_special_tokens=True` inserts nothing)
  and BOS/EOS belong to the chat template (apertus-program#420).
  Without it, the base's BOS/EOS post-processor is kept.

## Special flags vs. named roles

All omni tokens — structure *and* content — are added tokens
with `special=True`: each is matched as one token
and dropped by `skip_special_tokens`.

Separately, transformers keeps *named roles*:
the base roles `bos_token`, `eos_token`, `pad_token` and `unk_token`,
plus the `additional_special_tokens` list.
Which omni tokens are enrolled in `additional_special_tokens`
depends on the build scheme:

- `add_modality` (append, Apertus 1.5) enrolls every omni token,
  so its raw output lists ~135k of them and `all_special_ids` grows to match.
  transformers 4.x writes them as `additional_special_tokens`
  (in `special_tokens_map.json` and `tokenizer_config.json`);
  5.x writes no `special_tokens_map.json` and puts them under
  `extra_special_tokens` in `tokenizer_config.json`.
  The 1.5 recipe's finalize step keeps only the base roles in
  `special_tokens_map.json` and its eight role tokens as `extra_special_tokens`;
  the committed 1.5 tokenizer reports 12 `all_special_ids`.
- `add_modality_in_place` enrolls none,
  so only the base roles are named.

Consequences:

- Do **not** enumerate omni tokens via `all_special_ids`; it is version-dependent.
  Use `structure_token_ids` and the content ranges.
- To collect the *text* special tokens (e.g. goldfish-loss exemption):
  added tokens with `special=True` and `id < base_vocab_size`,
  read from `added_tokens_decoder`.

## Consumers

Megatron-LM's `populate_omni_metadata_from_tokenizer` reads `base_vocab_size`
and `modalities[].{name, offset, vocab_size}` into training args;
per-modality loss weighting and vocab padding derive from those.
Its goldfish exemption currently assumes the 1.5 geometry
(a contiguous span above `base_vocab_size`), and must switch to
the id-set derivation above for in-place tokenizers.
