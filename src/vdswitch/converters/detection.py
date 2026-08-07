"""Detection converters."""

from __future__ import annotations

import json
from pathlib import Path

from vdschema import AnnotationWriter, Name, TaskType

from .base import BaseConverter
from .registry import register_converter_for_sources
from .sources import Source


def _load_det_label(label_path: Path) -> dict[int, Name]:
    with label_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"invalid det label_dict: {label_path}")
    return {int(key): Name(str(name)) for key, name in raw.items()}


@register_converter_for_sources(
    task=TaskType.DETECTION,
    sources=(Source.MONOLITH, Source.UP),
)
class MonolithUpDetectionConverter(BaseConverter):
    """monolith/up baseline jsonl + label_dict.json → vdschema detection."""

    def _convert(self) -> None:
        writer = AnnotationWriter(
            TaskType.DETECTION,
            label=_load_det_label(self.input_label),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )

        with self.input_data.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
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
