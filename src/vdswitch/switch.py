"""vdswitch conversion entry point."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from vdschema import TaskType

from .converters.registry import (
    get_converter_class,
    parse_task,
    supported_sources,
    supported_tasks,
)
from .converters.sources import Source
from .options import resolve_converter_inputs


def _coerce_input_paths(
    input: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(input, (str, Path)):
        return [Path(input).expanduser().resolve()]
    paths = [Path(item).expanduser().resolve() for item in input]
    if not paths:
        raise ValueError("input must contain at least one path")
    return paths


DEFAULT_OUTPUT_DIR = "output"


def switch(
    *,
    task: TaskType,
    source: Source,
    input: str | Path | Sequence[str | Path],
    output: str | Path = DEFAULT_OUTPUT_DIR,
    options: Mapping[str, str] | None = None,
) -> Path:
    """Convert third party annotations to vdschema.

    Pass converter-specific settings via ``options`` (``--option KEY=VALUE`` on
    the CLI). Allowed keys depend on ``task`` and ``source``; see
    ``vdswitch help --task … --source …``.

    Use ``--option prefix=PATH`` (all sources) to prepend each JSONL row's
    ``filename`` field.

    Each ``input`` file is converted in order into the same ``output``
    directory; when multiple files are passed, later conversions overwrite
    same-named outputs.
    """
    task_type = parse_task(task)
    converter_cls = get_converter_class(task_type, source)
    output_dir = Path(output).expanduser().resolve()
    for input_path in _coerce_input_paths(input):
        resolved = resolve_converter_inputs(
            converter_cls,
            input=input_path,
            options=options,
        )
        converter_cls(
            input=input_path,
            category=resolved.category,
            output_dir=output_dir,
            root=resolved.root,
            convert_options=resolved.convert_options,
            prefix=resolved.prefix,
        ).run()
    return output_dir


__all__ = ["switch", "parse_task", "supported_sources", "supported_tasks"]
