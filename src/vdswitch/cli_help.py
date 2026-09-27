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
    if Source.MONOLITH in sources:
        return Source.MONOLITH
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


def _example_paths(task: TaskType, source: Source) -> tuple[str, str | None]:
    if source is Source.COCO:
        data_by_task = {
            TaskType.DETECTION: "/path/to/annotations/instances_train2017.json",
            TaskType.KEYPOINT: "/path/to/annotations/person_keypoints_train2017.json",
            TaskType.SEGMENTATION: "/path/to/annotations/instances_train2017.json",
        }
        data = data_by_task.get(task, "/path/to/annotations/instances.json")
        return data, None
    if source is Source.YOLO and task is TaskType.DETECTION:
        return "/path/to/train.txt", "/path/to/classes.txt"
    if source is Source.IMAGENET and task is TaskType.CLASSIFICATION:
        return "/path/to/train", None
    if source is Source.OCR and task is TaskType.SEQUENCE:
        return "/path/to/anno.txt", "/path/to/vocab.txt"
    if source is Source.LABELBEE:
        return "/path/to/labelbee/json", None
    if source is Source.LABELME:
        return "/path/to/labelme/json", None
    if source is Source.VOC:
        return "/path/to/VOC2007/ImageSets/Main/train.txt", None
    if task is TaskType.ACTION:
        return "/path/to/meta/video_train.txt", "/path/to/meta/label_dict.json"
    if task is TaskType.SEQUENCE:
        return "/path/to/meta/train_baseline.jsonl", "/path/to/meta/vocab.txt"
    return "/path/to/meta/train_baseline.jsonl", "/path/to/meta/label_dict.json"


def _example_command(task: TaskType, source: Source) -> str:
    cls = get_converter_class(task, source)
    data_path, label_path = _example_paths(task, source)
    if label_path is None and not cls.input_label_same_as_data:
        label_path = f"/path/to/{_label_placeholder(task)}"

    lines = [
        "vdswitch \\",
        f"  --task {task.value} \\",
        f"  --source {source.value} \\",
        f"  --input-data {data_path} \\",
    ]
    if label_path is not None and not cls.input_label_same_as_data:
        lines.append(f"  --input-label {label_path} \\")
    if not cls.input_root_optional:
        lines.append("  --input-root /path/to/dataset \\")
    lines.append("  --output output")
    return "\n".join(lines)


def _converter_class_for_task(task: TaskType) -> type:
    return get_converter_class(task, _default_source_for_task(task))


def _optional_flag_notes(cls: type) -> list[str]:
    notes: list[str] = []
    if cls.input_label_same_as_data:
        notes.append("--input-label: optional (defaults to each --input-data path)")
    if cls.input_root_optional:
        notes.append(
            "--input-root: optional when paths in data are absolute or "
            "inferable from layout"
        )
    return notes


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
    ]
    optional_notes = _optional_flag_notes(cls)
    if optional_notes:
        sections.extend(["", "Optional flags:", *[f"  {note}" for note in optional_notes]])
    sections.extend(
        [
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
    )
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
