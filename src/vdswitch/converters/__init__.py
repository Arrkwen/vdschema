"""Built-in converters; import submodules to populate the registry."""

from . import coco, imagenet, labelbee, monolith, ocr, yolo

__all__ = [
    "coco",
    "yolo",
    "imagenet",
    "labelbee",
    "monolith",
    "ocr",
    "MonolithUpActionConverter",
    "MonolithUpClassificationConverter",
    "MonolithUpDetectionConverter",
    "MonolithUpSequenceConverter",
    "CocoDetectionConverter",
    "CocoKeypointConverter",
    "CocoSegmentationConverter",
    "YoloDetectionConverter",
    "ImagenetClassificationConverter",
    "LabelBeeClassificationConverter",
    "LabelBeeDetectionConverter",
    "LabelBeeKeypointConverter",
    "LabelBeeSegmentationConverter",
    "OcrSequenceConverter",
]

from .coco import (
    CocoDetectionConverter,
    CocoKeypointConverter,
    CocoSegmentationConverter,
)
from .imagenet import ImagenetClassificationConverter
from .labelbee import (
    LabelBeeClassificationConverter,
    LabelBeeDetectionConverter,
    LabelBeeKeypointConverter,
    LabelBeeSegmentationConverter,
)
from .monolith import (
    MonolithUpActionConverter,
    MonolithUpClassificationConverter,
    MonolithUpDetectionConverter,
    MonolithUpSequenceConverter,
)
from .ocr import OcrSequenceConverter
from .yolo import YoloDetectionConverter
