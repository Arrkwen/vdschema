"""vdswitch CLI — legacy monolith/up → vdschema."""

from __future__ import annotations

import argparse
import sys

from . import converters  # noqa: F401 — register built-in converters
from .converters.registry import parse_task, supported_sources, supported_tasks
from .converters.sources import Source
from .switch import switch


def build_parser() -> argparse.ArgumentParser:
    task_choices = tuple(
        sorted(task.value for task in supported_tasks())
    )
    source_choices = tuple(
        sorted(source.value for source in supported_sources())
    )
    parser = argparse.ArgumentParser(
        prog="vdswitch",
        description="Convert third party annotations to vdschema format.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=task_choices,
        help="Target vdschema TaskType",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=source_choices,
        help="Third party source",
    )
    parser.add_argument(
        "--input-data",
        required=True,
        nargs="+",
        metavar="PATH",
        help="Third party  annotation data file path(s)",
    )
    parser.add_argument(
        "--input-label",
        required=True,
        help="Third party annotation dictionary file path",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for vdschema annotation data and dictionary",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Dataset root; joined with media paths in annotation files to "
            "resolve absolute image/video paths (e.g. for width/height); "
            "use when auto-detection fails"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = switch(
        task=parse_task(args.task),
        source=Source.parse(args.source),
        input_data=args.input_data,
        input_label=args.input_label,
        output=args.output,
        root=args.root,
    )
    print(f"vdswitch: wrote vdschema dataset to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
