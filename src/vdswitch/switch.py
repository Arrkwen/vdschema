"""vdswitch conversion entry point."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from vdschema import TaskType

from .converters.registry import (
    get_converter_class,
    parse_task,
    supported_sources,
    supported_tasks,
)
from .converters.sources import Source


def _coerce_input_data_paths(
    input_data: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(input_data, (str, Path)):
        return [Path(input_data).expanduser().resolve()]
    paths = [Path(item).expanduser().resolve() for item in input_data]
    if not paths:
        raise ValueError("input_data must contain at least one path")
    return paths


DEFAULT_OUTPUT_DIR = "output"


def switch(
    *,
    task: TaskType,
    source: Source,
    input_data: str | Path | Sequence[str | Path],
    input_label: str | Path | None = None,
    input_root: str | Path | None = None,
    output: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Convert third party annotations to vdschema.

    ``input_root`` is the dataset root directory. Media paths recorded in
    annotation files are joined with ``input_root`` to resolve absolute image
    or video paths (for example to read width and height). Pass it when
    auto-detection fails.

    Each ``input_data`` file is converted in order into the same ``output``
    directory; when multiple files are passed, later conversions overwrite
    same-named outputs.
    """
    task_type = parse_task(task)
    converter_cls = get_converter_class(task_type, source)
    output_dir = Path(output).expanduser().resolve()
    for data_path in _coerce_input_data_paths(input_data):
        converter_cls(
            input_data=data_path,
            input_label=input_label,
            output_dir=output_dir,
            input_root=input_root,
        ).run()
    return output_dir


__all__ = ["switch", "parse_task", "supported_sources", "supported_tasks"]
