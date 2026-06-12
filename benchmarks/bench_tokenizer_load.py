#!/usr/bin/env python3
"""Benchmark omni tokenizer load time and verify encoding correctness.

Load time is gated by the installed `tokenizers` version, not by the artifact
or by how tokens were added (see PR #1 discussion). The fix for slow loading
with many added tokens (huggingface/tokenizers#1635) shipped in tokenizers
0.22.2 via huggingface/tokenizers#1891 -- one patch release wide:

    tokenizers   Tokenizer.from_file (44MB tokenizer.json, 135k added tokens)
    0.21.0       ~60 s
    0.22.0       ~69 s
    0.22.1       ~70 s   <- what nemo-rl's uv.lock resolves; latest before 2026-01-05
    0.22.2       ~0.6 s  <- the fix
    0.23.1       ~0.5 s

Full AutoTokenizer.from_pretrained on the same artifact: transformers 5.3.0 +
tokenizers 0.22.1 = ~204 s; + tokenizers 0.22.2 = ~3-5 s; token IDs are
byte-identical across versions. NOTE: transformers <= 4.52 hard-pins
tokenizers<0.22 and raises ImportError on newer -- bump both there
(transformers 4.57.1 + tokenizers 0.22.2 verified).

Reproduce the cliff in isolated envs:

    uv venv /tmp/t && uv pip install --python /tmp/t/bin/python tokenizers==0.22.2
    /tmp/t/bin/python benchmarks/bench_tokenizer_load.py <tokenizer_dir> --rust-only
    # then flip the pin to ==0.22.1 and watch it take ~70s

Usage:
    python benchmarks/bench_tokenizer_load.py <tokenizer_dir> [--runs N] [--rust-only]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

RUST_SNIPPET = """
import json, sys, time
from tokenizers import Tokenizer
t0 = time.perf_counter()
Tokenizer.from_file(sys.argv[1])
print(json.dumps({"from_file_s": round(time.perf_counter() - t0, 3)}))
"""

FULL_SNIPPET = """
import json, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
t0 = time.perf_counter()
AutoTokenizer.from_pretrained(sys.argv[1])
print(json.dumps({"from_pretrained_s": round(time.perf_counter() - t0, 3)}))
"""


def fresh_process(snippet: str, arg: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", snippet, arg],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def correctness(tokenizer_dir: str) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    vis0 = tok.convert_tokens_to_ids("<|visual token 0|>")
    stress = "".join(f"<|visual token {i}|>" for i in range(1000))
    ids = tok.encode(stress, add_special_tokens=False)
    contiguous = ids == list(range(vis0, vis0 + 1000))
    sample = "check <|img_start|><|visual token 0|><|img_end|>"
    skipped = tok.decode(tok.encode(sample, add_special_tokens=False),
                         skip_special_tokens=True)
    print(f"  len(tokenizer)            : {len(tok)}")
    print(f"  '<|visual token 0|>' id   : {vis0}")
    print(f"  1000-token stress, one id per token, contiguous: {contiguous}")
    print(f"  decode(skip_special_tokens=True) drops modality tokens: "
          f"{skipped.strip() == 'check'}")
    if not contiguous:
        raise SystemExit("FAIL: content tokens did not encode to single contiguous ids")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tokenizer_dir")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--rust-only", action="store_true",
                        help="only time tokenizers.Tokenizer.from_file")
    args = parser.parse_args()

    import tokenizers
    print(f"python {sys.version.split()[0]} | tokenizers {tokenizers.__version__}",
          end="")
    if not args.rust_only:
        import transformers
        print(f" | transformers {transformers.__version__}", end="")
    print()

    tj = os.path.join(args.tokenizer_dir, "tokenizer.json")
    print(f"tokenizer.json: {os.path.getsize(tj):,} bytes")
    for i in range(args.runs):
        r = fresh_process(RUST_SNIPPET, tj)
        print(f"  Tokenizer.from_file        run {i + 1}: {r['from_file_s']}s")
    if args.rust_only:
        return
    for i in range(args.runs):
        r = fresh_process(FULL_SNIPPET, args.tokenizer_dir)
        print(f"  AutoTokenizer.from_pretrained run {i + 1}: {r['from_pretrained_s']}s")
    print("correctness:")
    correctness(args.tokenizer_dir)


if __name__ == "__main__":
    main()
