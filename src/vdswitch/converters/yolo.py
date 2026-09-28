"""YOLO txt labels → vdschema detection."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vdschema import AnnotationWriter, TaskType

from ..utils.yolo_dataset import (
    iter_image_list,
    load_yolo_class_names,
    parse_yolo_label_file,
    yolo_label_path_for_image,
)
from .base import BaseConverter
from .registry import register_converter
from .sources import Source


@register_converter(task=TaskType.DETECTION, source=Source.YOLO)
class YoloDetectionConverter(BaseConverter):
    """YOLO train list + classes.txt → vdschema detection."""

    source_note = "Ultralytics / Darknet YOLO layout (normalized cxcywh label txts)."

    input_data_help = (
        "Text file listing one image path per line (e.g. train.txt). "
        "Paths are relative to --input-root when set."
    )
    input_label_help = (
        "classes.txt: one class name per line; line index is YOLO class_id (0-based)."
    )
    input_data_sample = "images/train/sample.jpg"
    input_label_sample = "person\ncar\n"
    typical_layout = (
        "  train.txt              — image paths, one per line\n"
        "  classes.txt            — class names (0-based ids)\n"
        "  images/train/*.jpg     — images\n"
        "  labels/train/*.txt     — YOLO boxes (paired by path)"
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        same_dir = self.output_dir == self.input_data.parent
        stem = self.input_data.stem
        self.output_data_filename = (
            f"{stem}_vdschema.jsonl" if same_dir else f"{stem}.jsonl"
        )
        self.output_meta_filename = "label_dict.json"

    def _media_relative(self, image_path: Path, list_relative: str) -> str:
        if self.input_root is not None:
            try:
                return image_path.resolve().relative_to(self.input_root).as_posix()
            except ValueError:
                pass
        return list_relative

    def _convert(self) -> None:
        label = load_yolo_class_names(self.input_label)
        writer = AnnotationWriter(
            TaskType.DETECTION,
            label=label,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for image_path, list_relative in iter_image_list(
            self.input_data, root=self.input_root
        ):
            if not image_path.is_file():
                continue
            with Image.open(image_path) as img:
                width, height = img.size
            label_path = yolo_label_path_for_image(image_path)
            instances = parse_yolo_label_file(label_path, width=width, height=height)
            writer.append(
                filename=self._media_relative(image_path, list_relative),
                width=width,
                height=height,
                instances=instances,
            )
        writer.save()
