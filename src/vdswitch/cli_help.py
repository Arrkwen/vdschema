"""Text help for ``vdswitch help``."""

from __future__ import annotations

from vdschema import TaskType

from .converters.registry import get_converter_class, supported_sources, supported_tasks
from .converters.sources import Source
from .messages import help_hint


def _default_source_for_task(task: TaskType) -> Source:
    sources = supported_sources(task)
    if not sources:
        raise ValueError(f"no converter registered for task={task.value!r}")
    if Source.UP in sources:
        return Source.UP
    return sorted(sources, key=lambda item: item.value)[0]


def _resolve_source(task: TaskType, source: Source | None) -> Source:
    allowed = supported_sources(task)
    if source is not None:
        if source not in allowed:
            names = ", ".join(item.value for item in sorted(allowed, key=lambda s: s.value))
            raise ValueError(
                f"unsupported source={source.value!r} for task={task.value!r}, "
                f"allowed: {names}"
            )
        return source
    return _default_source_for_task(task)


def _example_command(task: TaskType, source: Source) -> str:
    if source is Source.COCO:
        data_by_task = {
            TaskType.DETECTION: "annotations/instances_train2017.json",
            TaskType.KEYPOINT: "annotations/person_keypoints_train2017.json",
            TaskType.SEGMENTATION: "annotations/instances_train2017.json",
        }
        data = data_by_task.get(task, "annotations/instances.json")
        return (
            "vdswitch \\\n"
            f"  --task {task.value} \\\n"
            "  --source coco \\\n"
            f"  --input-data /path/to/{data} \\\n"
            f"  --input-label /path/to/{data} \\\n"
            "  --input-root /path/to/dataset \\\n"
            "  --output output"
        )

    if source is Source.YOLO and task is TaskType.DETECTION:
        return (
            "vdswitch \\\n"
            "  --task detection \\\n"
            "  --source yolo \\\n"
            "  --input-data /path/to/train.txt \\\n"
            "  --input-label /path/to/classes.txt \\\n"
            "  --input-root /path/to/dataset \\\n"
            "  --output output"
        )

    if task is TaskType.ACTION:
        return (
            "vdswitch \\\n"
            "  --task action \\\n"
            "  --source up \\\n"
            "  --input-data /path/to/meta/video_train.txt \\\n"
            "  --input-label /path/to/meta/label_dict.json \\\n"
            "  --input-root /path/to/dataset \\\n"
            "  --output output"
        )

    label = _label_placeholder(task)
    data = "meta/train_baseline.jsonl"
    if task is TaskType.SEQUENCE:
        data = "meta/train_baseline.jsonl"
        label = "meta/vocab.txt"
    return (
        "vdswitch \\\n"
        f"  --task {task.value} \\\n"
        "  --source up \\\n"
        f"  --input-data /path/to/{data} \\\n"
        f"  --input-label /path/to/{label} \\\n"
        "  --input-root /path/to/dataset \\\n"
        "  --output output"
    )


def _converter_class_for_task(task: TaskType) -> type:
    return get_converter_class(task, _default_source_for_task(task))


def format_task_summary_lines() -> list[str]:
    lines: list[str] = []
    for task in sorted(supported_tasks(), key=lambda item: item.value):
        cls = _converter_class_for_task(task)
        sources = ", ".join(
            item.value for item in sorted(supported_sources(task), key=lambda s: s.value)
        )
        first = cls.input_data_help.split("\n", maxsplit=1)[0].strip()
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
            "Run with --task for full --input-data / --input-label specs:",
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
        "",
        "--input-data",
        cls.input_data_help or "(see converter docstring)",
        "",
        "Sample line / file excerpt:",
        cls.input_data_sample or "(none)",
        "",
        "--input-label",
        cls.input_label_help or "(see converter docstring)",
        "",
        "Sample content:",
        cls.input_label_sample or "(none)",
    ]
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


def _label_placeholder(task: TaskType) -> str:
    if task is TaskType.SEQUENCE:
        return "meta/vocab.txt"
    if task is TaskType.ACTION:
        return "meta/label_dict.json"
    return "meta/label_dict.json"
