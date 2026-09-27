"""Classification converters."""

from __future__ import annotations

import json
from pathlib import Path

from vdschema import AnnotationWriter, Name, TaskType

from ..messages import help_hint
from ..utils.jsonl import load_jsonl_object
from .base import BaseConverter
from .registry import register_converter
from .sources import Source


def _load_cls_label(label_path: Path) -> dict[str, dict[int, Name]]:
    with label_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"invalid classification label file: {label_path} "
            f'(expected JSON {{"attr": ["class_a", "class_b", ...]}}). '
            f"{help_hint(TaskType.CLASSIFICATION)}"
        )
    heads: dict[str, dict[int, Name]] = {}
    for attr, names in raw.items():
        if not isinstance(names, list) or not names:
            raise ValueError(
                f"label[{attr!r}] must be a non-empty list of class names in {label_path}. "
                f"{help_hint(TaskType.CLASSIFICATION)}"
            )
        heads[str(attr)] = {idx + 1: Name(str(name)) for idx, name in enumerate(names)}
    return heads


@register_converter(task=TaskType.CLASSIFICATION, source=Source.UP)
class MonolithUpClassificationConverter(BaseConverter):
    """UP baseline jsonl + label_dict.json → vdschema classification."""

    input_data_help = (
        "JSONL file: one JSON object per line with filename and attribute.\n"
        "attribute maps each head name to one-hot dict {class_name: 0|1, ...}."
    )
    input_label_help = (
        "JSON file: each key is a classification head (attribute) name; "
        "each value is an ordered list of class names (not numeric id maps)."
    )
    input_data_sample = (
        '{"filename":"img.jpg","attribute":{"gender":{"male":0,"female":1}}}'
    )
    input_label_sample = '{"gender":["male","female"]}'
    typical_layout = (
        "  meta/train_baseline.jsonl  — annotation JSONL\n"
        "  meta/label_dict.json       — head → class name lists\n"
        "  images/…                   — media (use --input-root)"
    )

    def _convert(self) -> None:
        label_heads = _load_cls_label(self.input_label)
        with self.input_label.open(encoding="utf-8") as f:
            raw = json.load(f)
        ordered_heads = {str(k): list(v) for k, v in raw.items()}

        writer = AnnotationWriter(
            TaskType.CLASSIFICATION,
            label=label_heads,
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
                attribute = record.get("attribute") or {}
                categories = []
                for attr_name, one_hot in attribute.items():
                    if not isinstance(one_hot, dict):
                        continue
                    names = ordered_heads.get(str(attr_name))
                    if not names:
                        continue
                    category_ids = [
                        idx + 1
                        for idx, name in enumerate(names)
                        if int(one_hot.get(name, 0)) == 1
                    ]
                    if not category_ids:
                        continue
                    categories.append(
                        {"category_attr": str(attr_name), "category_ids": category_ids}
                    )
                width, height = self.image_size_resolver.resolve(record)
                writer.append(
                    filename=record["filename"],
                    width=width,
                    height=height,
                    categories=categories,
                )

        writer.save()
