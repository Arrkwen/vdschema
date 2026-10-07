"""vdswitch CLI — legacy monolith → vdschema."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TextIO

from vdschema import AnnotationFormatError

from . import converters  # noqa: F401 — register built-in converters
from .converters.registry import parse_task, supported_sources, supported_tasks
from .converters.sources import Source
from .help import format_help_text
from .options import parse_option_pairs
from .switch import DEFAULT_OUTPUT_DIR, switch

_USAGE_INDENT = " " * 16
_CONVERT_EPILOG = "Input formats depend on --task. Run: vdswitch help [--task NAME]"
_ANSI_BOLD = "\033[1m"
_ANSI_RESET = "\033[0m"
_WINDOWS_VT: bool | None = None


def _enable_windows_vt() -> bool:  # pragma: no cover
    global _WINDOWS_VT
    if _WINDOWS_VT is not None:
        return _WINDOWS_VT
    import ctypes  # pragma: no cover — exercised only on Windows consoles

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        _WINDOWS_VT = False
        return False
    kernel32 = windll.kernel32  # pragma: no cover
    enable_vt = 0x0004
    ok = True
    for handle_id in (-11, -12):  # stdout, stderr
        handle = kernel32.GetStdHandle(handle_id)
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ok = False
            break
        if not kernel32.SetConsoleMode(handle, mode.value | enable_vt):
            ok = False
            break
    _WINDOWS_VT = ok
    return ok


def _terminal_bold(text: str, *, stream: TextIO | None = None) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    stream = stream or sys.stdout
    if not stream.isatty():
        return text
    if sys.platform == "win32" and not _enable_windows_vt():
        return text
    return f"{_ANSI_BOLD}{text}{_ANSI_RESET}"


class _HelpArgumentParser(argparse.ArgumentParser):
    """Apply bold epilog when printing help to a terminal."""

    def print_help(self, file: TextIO | None = None) -> None:  # ty: ignore[invalid-method-override]
        if file is None:
            file = sys.stdout
        epilog = self.epilog
        if epilog:
            self.epilog = _terminal_bold(epilog, stream=file)
        try:
            super().print_help(file)
        finally:
            self.epilog = epilog


def _join_choices(choices: tuple[str, ...]) -> str:
    return "{" + ",".join(choices) + "}"


def _convert_usage(
    task_choices: tuple[str, ...], source_choices: tuple[str, ...]
) -> str:
    i = _USAGE_INDENT
    tasks = _join_choices(task_choices)
    sources = _join_choices(source_choices)
    return (
        "vdswitch [-h]\n"
        f"{i}--task {tasks}\n"
        f"{i}--source {sources}\n"
        f"{i}--input PATH [PATH ...]\n"
        f"{i}[--output OUTPUT]\n"
        f"{i}[--option KEY=VALUE ...]"
    )


def _help_subcommand_usage(
    task_choices: tuple[str, ...], source_choices: tuple[str, ...]
) -> str:
    i = _USAGE_INDENT
    tasks = _join_choices(task_choices)
    sources = _join_choices(source_choices)
    return f"vdswitch help [-h]\n{i}[--task {tasks}]\n{i}[--source {sources}]"


def build_convert_parser() -> argparse.ArgumentParser:
    task_choices = tuple(sorted(task.value for task in supported_tasks()))
    source_choices = tuple(sorted(source.value for source in supported_sources()))
    parser = _HelpArgumentParser(
        prog="vdswitch",
        usage=_convert_usage(task_choices, source_choices),
        description="Convert third party annotations to vdschema format.",
        epilog=_CONVERT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=task_choices,
        help="Target vdschema task type",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=source_choices,
        help="Third party source (monolith, coco, yolo, …)",
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        metavar="PATH",
        help="Annotation data path(s) (file, directory, or manifest)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for vdschema annotation data and dictionary",
    )
    parser.add_argument(
        "--option",
        action="append",
        metavar="KEY=VALUE",
        default=None,
        help="Converter option (repeatable); keys depend on --task and --source",
    )
    return parser


def build_help_parser() -> argparse.ArgumentParser:
    task_choices = tuple(sorted(task.value for task in supported_tasks()))
    source_choices = tuple(sorted(source.value for source in supported_sources()))
    parser = argparse.ArgumentParser(
        prog="vdswitch help",
        usage=_help_subcommand_usage(task_choices, source_choices),
        description="Show expected --input and --option formats for a task/source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=task_choices,
        default=None,
        help="vdschema task type (omit to list all tasks)",
    )
    parser.add_argument(
        "--source",
        choices=source_choices,
        default=None,
        help="Optional; see vdswitch help --task …",
    )
    return parser


def run_help(argv: list[str]) -> int:
    args = build_help_parser().parse_args(argv)
    try:
        task = parse_task(args.task) if args.task is not None else None
        source = Source.parse(args.source) if args.source is not None else None
        if task is None and source is not None:
            print(
                "vdswitch help: --source requires --task",
                file=sys.stderr,
            )
            return 2
        text = format_help_text(task=task, source=source)
    except ValueError as exc:
        print(f"vdswitch help: error: {exc}", file=sys.stderr)
        return 1
    print(text)
    return 0


def run_convert(argv: list[str]) -> int:
    args = build_convert_parser().parse_args(argv)
    try:
        options = parse_option_pairs(args.option)
        out = switch(
            task=parse_task(args.task),
            source=Source.parse(args.source),
            input=args.input,
            output=args.output,
            options=options,
        )
    except (AnnotationFormatError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"vdswitch: error: {exc}", file=sys.stderr)
        return 1
    print(f"vdswitch: wrote vdschema dataset to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "help":
        return run_help(argv[1:])
    return run_convert(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
