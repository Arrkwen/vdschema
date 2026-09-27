"""Sequence converters."""

from __future__ import annotations

from vdschema import AnnotationWriter, TaskType

from ..utils.jsonl import load_jsonl_object
from .base import BaseConverter
from .registry import register_converter
from .sources import Source


@register_converter(task=TaskType.SEQUENCE, source=Source.MONOLITH)
class MonolithUpSequenceConverter(BaseConverter):
    """Monolith baseline jsonl + vocab.txt → vdschema sequence."""

    input_data_help = (
        "JSONL file: one JSON object per line with filename and sequences "
        "(list of token strings from the vocab file)."
    )
    input_label_help = (
        "Plain-text vocab file: one token per line (not JSON). "
        "Copied to vdschema meta as the sequence vocabulary."
    )
    input_data_sample = '{"filename":"img.jpg","sequences":["B","1","0"]}'
    input_label_sample = "0\n1\nB\n"
    typical_layout = (
        "  meta/train_baseline.jsonl  — annotation JSONL\n"
        "  meta/vocab.txt             — one token per line\n"
        "  images/…                   — media (use --input-root)"
    )

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
