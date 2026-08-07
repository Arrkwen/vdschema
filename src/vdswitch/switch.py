"""vdswitch conversion entry point."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from vdschema import TaskType

from .converters.registry import get_converter_class, parse_task, supported_sources, supported_tasks
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


def switch(
    *,
    task: TaskType,
    source: Source,
    input_data: str | Path | Sequence[str | Path],
    input_label: str | Path,
    output: str | Path,
    root: str | Path | None = None,
) -> Path:
    """Convert third party annotations to vdschema.

    ``root`` is the dataset root directory. Media paths recorded in annotation
    files are joined with ``root`` to resolve absolute image or video paths
    (for example to read width and height). Pass it when auto-detection fails.
    """
    task_type = parse_task(task)
    converter_cls = get_converter_class(task_type, source)
    output_dir = Path(output).expanduser().resolve()
    for data_path in _coerce_input_data_paths(input_data):
        converter_cls(
            input_data=data_path,
            input_label=input_label,
            output_dir=output_dir,
            root=root,
        ).run()
    return output_dir


__all__ = ["switch", "parse_task", "supported_sources", "supported_tasks"]
