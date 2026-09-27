"""OCR line manifest → vdschema sequence."""

from __future__ import annotations

from PIL import Image

from vdschema import AnnotationWriter, TaskType

from ..utils.ocr_dataset import iter_ocr_lines
from .base import BaseConverter
from .registry import register_converter
from .sources import Source


@register_converter(task=TaskType.SEQUENCE, source=Source.OCR)
class OcrSequenceConverter(BaseConverter):
    """OCR manifest (path + label string) + vocab.txt → vdschema sequence."""

    source_note = "Classic OCR list file: one sample per line (path TAB label text)."

    input_data_help = (
        "Text manifest: each line is ``<image_path>\\t<label>`` (or space-separated). "
        "Label string is split into character tokens for the sequence."
    )
    input_label_help = (
        "vocab.txt: one token per line (character or subword in the label alphabet)."
    )
    input_data_sample = "images/001.jpg\thello"
    input_label_sample = "h\ne\nl\no\n"
    typical_layout = (
        "  anno.txt                 — image path + transcription per line\n"
        "  vocab.txt                — allowed tokens\n"
        "  images/…                 — media (use --input-root)"
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        same_dir = self.output_dir == self.input_data.parent
        stem = self.input_data.stem
        self.output_data_filename = (
            f"{stem}_vdschema.jsonl" if same_dir else f"{stem}.jsonl"
        )

    def _convert(self) -> None:
        writer = AnnotationWriter(
            TaskType.SEQUENCE,
            label=self.input_label,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for image_path, list_relative, tokens in iter_ocr_lines(
            self.input_data, root=self.input_root
        ):
            if not image_path.is_file():
                continue
            with Image.open(image_path) as img:
                width, height = img.size
            writer.append(
                filename=list_relative,
                width=width,
                height=height,
                sequences=tokens,
            )
        writer.save()
