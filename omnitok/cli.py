"""CLI entry points for omnitok."""

from __future__ import annotations

import argparse
import json


def cmd_add_modality(args: argparse.Namespace) -> None:
    from .builder import add_modality

    extra_config = None
    if args.extra_config:
        extra_config = json.loads(args.extra_config)

    add_modality(
        input_tokenizer_path=args.input_tokenizer,
        output_path=args.output_path,
        modality=args.modality,
        vocab_size=args.vocab_size,
        num_reserved_tokens=args.num_reserved_tokens,
        extra_config=extra_config,
    )


def cmd_build_apertus_1p5(args: argparse.Namespace) -> None:
    from .apertus import build_apertus_1p5

    build_apertus_1p5(
        output_path=args.output_path,
        base_tokenizer_path=args.base_tokenizer,
        revision=args.revision,
        chat_template_file=args.chat_template,
        work_dir=args.work_dir,
    )


def cmd_build_apertus_2(args: argparse.Namespace) -> None:
    from .versions import apertus_2

    apertus_2.build(args.input_tokenizer, args.output_path)


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
        "--num-reserved-tokens", type=int, default=200,
        help="Number of RESERVED_OMNI slots (default: 200).",
    )
    p_add.add_argument(
        "--extra-config", type=str, default=None,
        help='JSON string for modality metadata, e.g. \'{"type": "Emu3.5"}\'.',
    )
    p_add.set_defaults(func=cmd_add_modality)

    # ── build-apertus-1p5 ───────────────────────────────────────────────────
    from .apertus import BASE_REPO, BASE_REVISION

    p_v15 = sub.add_parser(
        "build-apertus-1p5",
        help="Build the canonical Apertus 1.5 tokenizer from the Apertus 1 base.",
    )
    p_v15.add_argument("--output-path", required=True,
                       help="Where to write the final tokenizer.")
    p_v15.add_argument("--base-tokenizer", default=BASE_REPO,
                       help=f"Apertus 1 tokenizer (path or HF ID, default: {BASE_REPO}).")
    p_v15.add_argument("--revision", default=BASE_REVISION,
                       help=f"Hub commit for --base-tokenizer (default: {BASE_REVISION[:12]}).")
    p_v15.add_argument("--chat-template", default=None,
                       help="Chat template file (default: the checked-in Apertus 1.5 template).")
    p_v15.add_argument("--work-dir", default=None,
                       help="Keep intermediate stage directories here instead of a tempdir.")
    p_v15.set_defaults(func=cmd_build_apertus_1p5)

    # ── build-apertus-2 ─────────────────────────────────────────────────
    p_v2 = sub.add_parser(
        "build-apertus-2",
        help="Build the Apertus 2 omni tokenizer from the pinned text base.",
    )
    p_v2.add_argument(
        "--input-tokenizer", required=True,
        help="Path to the preliminary_mul_200k text base.",
    )
    p_v2.add_argument(
        "--output-path", required=True,
        help="Where to save the result.",
    )
    p_v2.set_defaults(func=cmd_build_apertus_2)

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
