"""Sequence converters."""

from __future__ import annotations

from vdschema import AnnotationWriter, TaskType

from ..utils.jsonl import load_jsonl_object
from .base import BaseConverter
from .registry import register_converter_for_sources
from .sources import Source


@register_converter_for_sources(
    task=TaskType.SEQUENCE,
    sources=(Source.MONOLITH, Source.UP),
)
class MonolithUpSequenceConverter(BaseConverter):
    """monolith/up baseline jsonl + vocab.txt → vdschema sequence."""

    def _convert(self) -> None:
        writer = AnnotationWriter(
            TaskType.SEQUENCE,
            label=self.input_label,
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
                width, height = self.image_size_resolver.resolve(record)
                writer.append(
                    filename=record["filename"],
                    width=width,
                    height=height,
                    sequences=list(record.get("sequences") or []),
                )

        writer.save()
