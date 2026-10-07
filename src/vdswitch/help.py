"""Text help for ``vdswitch help``."""

from __future__ import annotations

from vdschema import TaskType

from .converters.base import BaseConverter
from .converters.registry import get_converter_class, supported_sources, supported_tasks
from .converters.sources import Source
from .options.spec import (
    example_option_command_lines,
    format_option_spec_help_lines,
    partition_converter_options,
)


def help_hint(task: TaskType) -> str:
    return f"Run: vdswitch help --task {task.value}"


def _default_source_for_task(task: TaskType) -> Source:
    sources = supported_sources(task)
    if not sources:
        raise ValueError(f"no converter registered for task={task.value!r}")
    if Source.MONOLITH in sources:
        return Source.MONOLITH
    return sorted(sources, key=lambda item: item.value)[0]


def _resolve_source(task: TaskType, source: Source | None) -> Source:
    allowed = supported_sources(task)
    if source is not None:
        if source not in allowed:
            names = ", ".join(
                item.value for item in sorted(allowed, key=lambda s: s.value)
            )
            raise ValueError(
                f"unsupported source={source.value!r} for task={task.value!r}, "
                f"allowed: {names}"
            )
        return source
    return _default_source_for_task(task)


def _example_command(task: TaskType, source: Source) -> str:
    cls = get_converter_class(task, source)
    input_path = cls.example_input
    category_path = cls.example_category

    lines = [
        "vdswitch \\",
        f"  --task {task.value} \\",
        f"  --source {source.value} \\",
        f"  --input {input_path} \\",
    ]
    lines.extend(
        example_option_command_lines(
            cls,
            input_path=input_path,
            category_path=category_path,
        )
    )
    lines.append("  --output output")
    return "\n".join(lines)


def _converter_class_for_task(task: TaskType) -> type[BaseConverter]:
    return get_converter_class(task, _default_source_for_task(task))


def _param_block(
    label: str,
    description: str,
    *,
    example: str | None = None,
) -> list[str]:
    lines = [f"  {label}", f"      {description.strip()}"]
    if example:
        lines.append(f"      e.g. {example}")
    return lines


def _required_and_optional_sections(cls: type[BaseConverter]) -> list[str]:
    sections: list[str] = ["", "Required"]
    sections.extend(
        _param_block(
            "--input PATH",
            cls.input_help or "(see converter docstring)",
            example=cls.input_sample or None,
        )
    )
    sections.extend(
        _param_block(
            "--output DIR",
            "Directory for vdschema task JSONL and meta files.",
        )
    )

    required_opts, optional_opts = partition_converter_options(cls)
    if required_opts:
        sections.extend(["", "Required --option (repeatable)"])
        for spec in required_opts:
            sections.extend(format_option_spec_help_lines(spec))

    optional_lines: list[str] = []
    for spec in optional_opts:
        optional_lines.extend(format_option_spec_help_lines(spec))

    if optional_lines:
        sections.extend(["", "Optional --option (repeatable)", *optional_lines])

    return sections


def format_task_summary_lines() -> list[str]:
    lines: list[str] = []
    for task in sorted(supported_tasks(), key=lambda item: item.value):
        cls = _converter_class_for_task(task)
        sources = ", ".join(
            item.value
            for item in sorted(supported_sources(task), key=lambda s: s.value)
        )
        first = cls.input_help.split("\n", maxsplit=1)[0].strip()
        if first:
            lines.append(f"  {task.value} ({sources}): {first}")
        else:
            lines.append(f"  {task.value} ({sources})")
    return lines


def format_help_text(*, task: TaskType | None, source: Source | None) -> str:
    if task is None:
        lines = [
            "vdswitch — input formats by task",
            "",
            "Run with --task for full --input and --option specs:",
            "",
            *format_task_summary_lines(),
            "",
            "Example: vdswitch help --task classification",
            "         vdswitch help --task detection --source coco",
        ]
        return "\n".join(lines)

    allowed = sorted(supported_sources(task), key=lambda item: item.value)
    effective = _resolve_source(task, source)
    cls = get_converter_class(task, effective)
    if source is not None:
        source_line = source.value
    else:
        source_line = ", ".join(item.value for item in allowed)
        if len(allowed) > 1:
            source_line += f" (below: {effective.value}; pass --source to switch)"

    sections = [
        f"Task: {task.value}",
        f"Source: {source_line}",
        cls.source_note,
    ]
    sections.extend(_required_and_optional_sections(cls))
    if cls.typical_layout:
        sections.extend(["", "Typical paths under dataset root:", cls.typical_layout])
    sections.extend(
        [
            "",
            "Example convert command (adjust paths):",
            _example_command(task, effective),
            "",
            help_hint(task),
        ]
    )
    return "\n".join(sections)
