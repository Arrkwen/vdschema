"""Per-source ``converter_options`` (source-specific keys + shared globals)."""

from __future__ import annotations

from .spec import ConverterOptionSpec

_GLOBAL: tuple[ConverterOptionSpec, ...] = (
    ConverterOptionSpec(
        "prefix",
        "Prepended to each JSONL row filename (e.g. split/v1/1.jpg).",
        example="split/v1",
    ),
)


def _options(*specs: ConverterOptionSpec) -> tuple[ConverterOptionSpec, ...]:
    declared = {spec.key for spec in specs}
    extra = [item for item in _GLOBAL if item.key not in declared]
    return (*specs, *extra)


COCO_OPTIONS: tuple[ConverterOptionSpec, ...] = _options(
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

MONOLITH_OPTIONS: tuple[ConverterOptionSpec, ...] = _options(
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

YOLO_OPTIONS: tuple[ConverterOptionSpec, ...] = _options(
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

OCR_OPTIONS: tuple[ConverterOptionSpec, ...] = _options(
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

LABELME_OPTIONS: tuple[ConverterOptionSpec, ...] = _options(
    ConverterOptionSpec(
        "root",
        "Dataset root when `imagePath` in JSON is relative to --input.",
        example="/path/to/dataset",
    ),
)

LABELBEE_OPTIONS: tuple[ConverterOptionSpec, ...] = _options(
    ConverterOptionSpec(
        "root",
        "Dataset root for image files paired with LabelBee JSON by stem.",
        example="/path/to/dataset",
    ),
)

VOC_OPTIONS: tuple[ConverterOptionSpec, ...] = _options(
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

GLOBAL_ONLY_OPTIONS: tuple[ConverterOptionSpec, ...] = _options()

IMAGENET_OPTIONS = GLOBAL_ONLY_OPTIONS
