"""ImageNet folder layout → vdschema classification."""

from __future__ import annotations

from PIL import Image

from vdschema import AnnotationWriter, TaskType

from ..utils.imagenet_dataset import (
    DEFAULT_HEAD,
    build_imagenet_head,
    sorted_class_dirs,
)
from .base import BaseConverter
from .registry import register_converter
from .sources import Source

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@register_converter(task=TaskType.CLASSIFICATION, source=Source.IMAGENET)
class ImagenetClassificationConverter(BaseConverter):
    """ImageNet train/<class_name>/* → vdschema classification."""

    source_note = "ImageNet-style directory tree: one subfolder per class."

    input_data_help = (
        "Directory whose immediate subfolders are class names "
        "(e.g. train/n01440764/*.JPEG)."
    )
    input_label_help = (
        "Same directory as --input-data (class names inferred from folder names)."
    )
    input_data_sample = "train/n01440764/sample.JPEG"
    input_label_sample = "(same directory as --input-data)"
    typical_layout = (
        "  train/<class_name>/*.jpg   — one folder per category\n"
        "  --input-data and --input-label both point at train/"
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        same_dir = self.output_dir == self.input_data
        stem = self.input_data.name
        self.output_data_filename = (
            f"{stem}_vdschema.jsonl" if same_dir else f"{stem}.jsonl"
        )
        self.output_meta_filename = "label_dict.json"

    def _ensure_inputs(self) -> None:
        if not self.input_data.is_dir():
            raise FileNotFoundError(f"input data not found: {self.input_data}")
        if not self.input_label.is_dir() and not self.input_label.is_file():
            raise FileNotFoundError(f"input label not found: {self.input_label}")

    def _convert(self) -> None:
        heads, name_to_id = build_imagenet_head(self.input_data)
        dataset_root = self.input_root or self.input_data.parent
        writer = AnnotationWriter(
            TaskType.CLASSIFICATION,
            label=heads,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for class_dir in sorted_class_dirs(self.input_data):
            category_id = name_to_id[class_dir.name]
            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                with Image.open(image_path) as img:
                    width, height = img.size
                filename = image_path.relative_to(dataset_root).as_posix()
                writer.append(
                    filename=filename,
                    width=width,
                    height=height,
                    categories=[
                        {
                            "category_attr": DEFAULT_HEAD,
                            "category_ids": [category_id],
                        }
                    ],
                )
        writer.save()
