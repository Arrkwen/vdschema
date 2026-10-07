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

    input_is_dir = True
    example_input = "/path/to/train"

    source_note = "ImageNet-style directory tree: one subfolder per class."

    input_help = (
        "Directory whose immediate subfolders are class names "
        "(e.g. train/n01440764/*.JPEG)."
    )
    input_sample = "train/n01440764/sample.JPEG"
    typical_layout = (
        "  train/<class_name>/*.jpg   — one folder per category\n"
        "  --input points at train/ (dataset root defaults to its parent)"
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        same_dir = self.output_dir == self.input
        stem = self.input.name
        self.output_data_filename = (
            f"{stem}_vdschema.jsonl" if same_dir else f"{stem}.jsonl"
        )
        self.output_meta_filename = "label_dict.json"

    def _convert(self) -> None:
        heads, name_to_id = build_imagenet_head(self.input)
        dataset_root = self.root or self.input.parent
        writer = AnnotationWriter(
            TaskType.CLASSIFICATION,
            label=heads,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for class_dir in sorted_class_dirs(self.input):
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
                    filename=self.prefix_media_filename(filename),
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
