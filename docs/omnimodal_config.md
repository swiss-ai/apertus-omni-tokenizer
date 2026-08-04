# omnimodal_config

`omnimodal_config` in `tokenizer_config.json` is the metadata contract
between built omni-tokenizers and their consumers
(Megatron-LM, serving, data pipelines).
It is derived from the tokenizer vocabulary at build time,
and verified on every write path.

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

Modalities are sorted by `offset`.

## Contracts

- **Content ids are contiguous**: `id = offset + index`.
  The build verifies this for every modality and fails on gaps,
  so consumers may rely on the arithmetic.
- **Structure tokens resolve by name**, via `structure_token_ids`
  or the tokenizer itself. Do not assume geometry:
  Apertus 1.5's structure tokens sit in a contiguous block above
  `base_vocab_size`, Apertus 2's sit at low ids inside the base vocab
  (reused placeholders at 18/19, renamed pool slots at 27-39).
- **Apertus 2 artifacts have no post-processor**:
  encoding is exact (`add_special_tokens=True` inserts nothing),
  and BOS/EOS belong to the chat template (apertus-program#420).

## Special flags vs. named roles

All omni tokens — structure *and* content — are added tokens
with `special=True` (atomic, stripped by `skip_special_tokens`).
Enrollment in the `additional_special_tokens` *named role* differs by version.
1.5-era append builds enrolled everything, so fresh rebuilds on current
transformers serialize ~135k entries into `special_tokens_map.json`
and `all_special_ids`; Apertus 2 builds enroll nothing beyond the base roles,
keeping `special_tokens_map.json` role-only.

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
the id-set derivation above for Apertus 2 tokenizers.
