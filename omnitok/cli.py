"""CLI entry point: build the Apertus 1.5 tokenizer from the Apertus 1 base.

The generic modality/instruct stages live in omnitok.builder / omnitok.instruct
and are driven by the recipe in omnitok.apertus; this CLI exposes the recipe.
"""

from __future__ import annotations

import argparse


def main() -> None:
    from .apertus import BASE_REPO, BASE_REVISION, build_apertus_1p5

    parser = argparse.ArgumentParser(
        prog="omni-tokenizer",
        description=(
            "Build the canonical Apertus 1.5 tokenizer "
            "(tokenizers/Apertus_1p5) from the Apertus 1 base."
        ),
    )
    parser.add_argument(
        "--output-path", required=True,
        help="Where to write the final tokenizer.",
    )
    parser.add_argument(
        "--base-tokenizer", default=BASE_REPO,
        help=f"Apertus 1 tokenizer (path or HF ID, default: {BASE_REPO}).",
    )
    parser.add_argument(
        "--revision", default=BASE_REVISION,
        help="Hub commit for --base-tokenizer; ignored for local paths "
             f"(default: {BASE_REVISION[:12]}).",
    )
    parser.add_argument(
        "--chat-template", default=None,
        help="Apertus 1.5 chat template file "
             "(default: chat_templates/Apertus_1p5/chat_template.jinja).",
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Keep intermediate stage directories here instead of a tempdir.",
    )
    args = parser.parse_args()

    build_apertus_1p5(
        output_path=args.output_path,
        base_tokenizer_path=args.base_tokenizer,
        revision=args.revision,
        chat_template_file=args.chat_template,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    main()
