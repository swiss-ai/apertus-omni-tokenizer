# Possible Extensions

Ideas considered but intentionally deferred. Not implemented yet.

## Auto-reuse of pre-existing modality tokens

In the current in-place mode, each structure token declares its allocation strategy
explicitly in its `TokenRename` (`source="reuse"` for tokens that already exist in the
base vocab such as `<|image|>`/`<|audio|>`, `source="pool"`/`"explicit"` for tokens that
must be claimed from the `<SPECIAL_*>` reserve pool). An auto-reuse mode would drop the
per-token `source` label and instead let the builder decide at build time by inspecting
the input tokenizer's vocabulary: for each structure-token target name, if a token with
that name already exists, reuse its id (wire the metadata, consume no pool slot, perform
no rename); otherwise allocate the next free `<SPECIAL_n>` and rename it. This makes a
single `ModalityConfig` adapt across base tokenizers that pre-bake different sets of
special tokens, at the cost of making reuse implicit (a silent name match rather than a
declared, auditable intent) and requiring the expected-existing tokens to be whitelisted
so they do not trip `allow_existing=False` strict-mode collision checks.
