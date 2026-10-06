"""Shared ``converter_options`` tuples."""

from __future__ import annotations

from .spec import ConverterOptionSpec

MONOLITH_PATH_OPTIONS: tuple[ConverterOptionSpec, ...] = (
    ConverterOptionSpec(
        "category",
        "Label or vocab file (JSON or plain text).",
        required=True,
    ),
    ConverterOptionSpec(
        "root",
        "Dataset root for images/ or video/ paths referenced in --input.",
        required=True,
        example="/path/to/dataset",
    ),
)

YOLO_PATH_OPTIONS: tuple[ConverterOptionSpec, ...] = (
    ConverterOptionSpec(
        "category",
        "classes.txt — one class name per line (0-based class id).",
        required=True,
        example="/path/to/classes.txt",
    ),
    ConverterOptionSpec(
        "root",
        "Root for image paths listed in train.txt.",
        required=True,
        example="/path/to/dataset",
    ),
)

OCR_PATH_OPTIONS: tuple[ConverterOptionSpec, ...] = (
    ConverterOptionSpec(
        "category",
        "Plain-text vocabulary (one token per line).",
        required=True,
        example="/path/to/vocab.txt",
    ),
    ConverterOptionSpec(
        "root",
        "Dataset root for image paths in the manifest.",
        required=True,
        example="/path/to/dataset",
    ),
)

COCO_CATEGORY_OPTIONS: tuple[ConverterOptionSpec, ...] = (
    ConverterOptionSpec(
        "category_id_contiguous",
        "0|1 — keep source ids (0, default) or contiguous remap by sorted id (1).",
        example="0",
    ),
    ConverterOptionSpec(
        "category_id_start",
        "0|1 — first remapped id when contiguous=1 (default 1).",
        example="1",
    ),
)

OPTIONAL_ROOT: tuple[ConverterOptionSpec, ...] = (
    ConverterOptionSpec(
        "root",
        "Dataset root when media paths in annotations are relative.",
        example="/path/to/dataset",
    ),
)

VOC_OPTIONS: tuple[ConverterOptionSpec, ...] = (
    ConverterOptionSpec(
        "root",
        "PASCAL VOC year folder when layout cannot be inferred.",
        example="/path/to/VOC2007",
    ),
    ConverterOptionSpec(
        "category",
        "Optional classes.txt (defaults to categories from --input layout).",
    ),
)
