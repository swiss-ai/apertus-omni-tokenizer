"""CLI entry points for omnitok."""

from __future__ import annotations

import argparse
import json
import sys


def cmd_add_modality(args: argparse.Namespace) -> None:
    from .builder import add_modality

    extra_config = None
    if args.extra_config:
        extra_config = json.loads(args.extra_config)

    slot_assignments = None
    if args.slot_assignments:
        slot_assignments = json.loads(args.slot_assignments)

    add_modality(
        input_tokenizer_path=args.input_tokenizer,
        output_path=args.output_path,
        modality=args.modality,
        vocab_size=args.vocab_size,
        allocation=args.allocation,
        slot_assignments=slot_assignments,
        allow_existing=args.allow_existing,
        dry_run=args.dry_run,
        num_reserved_tokens=args.num_reserved_tokens,
        extra_config=extra_config,
    )


def cmd_add_instruct(args: argparse.Namespace) -> None:
    from .instruct import create_instruct_tokenizer

    create_instruct_tokenizer(
        base_tokenizer_path=args.base_tokenizer_path,
        instruct_tokenizer_path=args.instruct_tokenizer_path,
        output_path=args.output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="omni-tokenizer",
        description="Create omnimodal tokenizers by adding modality tokens to HuggingFace text tokenizers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── add-modality ──────────────────────────────────────────────────────
    p_add = sub.add_parser(
        "add-modality",
        help="Add a modality (vision, audio, ...) to a tokenizer.",
    )
    p_add.add_argument(
        "--input-tokenizer", required=True,
        help="Path or HF model ID for the input tokenizer.",
    )
    p_add.add_argument(
        "--output-path", required=True,
        help="Where to save the result.",
    )
    p_add.add_argument(
        "--modality", required=True,
        help="Modality name (vision, audio).",
    )
    p_add.add_argument(
        "--vocab-size", type=int, required=True,
        help="Number of content tokens (codebook size).",
    )
    p_add.add_argument(
        "--allocation", choices=["append", "in_place"], default="append",
        help="append (default): add a RESERVED_OMNI block on top. in_place: reuse "
             "pre-baked specials + the <SPECIAL_*> reserve pool in the base vocab.",
    )
    p_add.add_argument(
        "--slot-assignments", type=str, default=None,
        help='in_place only: JSON map of explicit pool-ORDINAL overrides (the N in '
             '<SPECIAL_N>, not a token id), e.g. \'{"<|img_start|>": 40}\' pins it to '
             '<SPECIAL_40>.',
    )
    p_add.add_argument(
        "--allow-existing", action=argparse.BooleanOptionalAction, default=True,
        help="Accept (skip) tokens that already exist. --no-allow-existing raises "
             "on any unexpected pre-existing token (default: allow).",
    )
    p_add.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the change report without writing anything.",
    )
    p_add.add_argument(
        "--num-reserved-tokens", type=int, default=200,
        help="append mode: number of RESERVED_OMNI slots (default: 200).",
    )
    p_add.add_argument(
        "--extra-config", type=str, default=None,
        help='JSON string for modality metadata, e.g. \'{"type": "Emu3.5"}\'.',
    )
    p_add.set_defaults(func=cmd_add_modality)

    # ── add-instruct ──────────────────────────────────────────────────────
    p_inst = sub.add_parser(
        "add-instruct",
        help="Add chat template + SFT sequences to an omni-tokenizer.",
    )
    p_inst.add_argument(
        "--base-tokenizer-path", required=True,
        help="Path to base omni-tokenizer.",
    )
    p_inst.add_argument(
        "--instruct-tokenizer-path", required=True,
        help="Path or HF model ID for chat template source.",
    )
    p_inst.add_argument(
        "--output-path", required=True,
        help="Where to save the result.",
    )
    p_inst.set_defaults(func=cmd_add_instruct)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
