"""Detection converters."""

from __future__ import annotations

import json
from pathlib import Path

from vdschema import AnnotationWriter, Name, TaskType

from ..messages import help_hint
from ..utils.jsonl import load_jsonl_object
from .base import BaseConverter
from .registry import register_converter
from .sources import Source


def _load_det_label(label_path: Path) -> dict[int, Name]:
    with label_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"invalid detection label file: {label_path} "
            f'(expected JSON {{"1":"name1","2":"name2", ...}}). '
            f"{help_hint(TaskType.DETECTION)}"
        )
    return {int(key): Name(str(name)) for key, name in raw.items()}


@register_converter(task=TaskType.DETECTION, source=Source.UP)
class MonolithUpDetectionConverter(BaseConverter):
    """UP baseline jsonl + label_dict.json → vdschema detection."""

    input_data_help = (
        "JSONL file: one JSON object per line with filename and instances "
        "(bbox, label or category_id, optional is_ignored)."
    )
    input_label_help = (
        "JSON file: string keys are category ids, values are class names "
        '(e.g. {"1":"person","2":"car"}).'
    )
    input_data_sample = (
        '{"filename":"img.jpg","instances":[{"id":0,"label":1,'
        '"bbox":[10,10,100,100]}]}'
    )
    input_label_sample = '{"1":"person","2":"car"}'
    typical_layout = (
        "  meta/train_baseline.jsonl  — annotation JSONL\n"
        "  meta/label_dict.json       — id → class name map\n"
        "  images/…                   — media (use --input-root)"
    )

    def _convert(self) -> None:
        writer = AnnotationWriter(
            TaskType.DETECTION,
            label=_load_det_label(self.input_label),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )

        with self.input_data.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = load_jsonl_object(line, path=self.input_data, lineno=lineno)
                instances = []
                for idx, inst in enumerate(record.get("instances") or []):
                    item = {
                        "id": int(inst.get("id", idx)),
                        "category_id": int(inst.get("label", inst.get("category_id"))),
                        "bbox": inst["bbox"],
                    }
                    if inst.get("is_ignored") or inst.get("is_ignore"):
                        item["is_ignored"] = True
                    instances.append(item)
                width, height = self.image_size_resolver.resolve(record)
                writer.append(
                    filename=record["filename"],
                    width=width,
                    height=height,
                    instances=instances,
                )

        writer.save()
