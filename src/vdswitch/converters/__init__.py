"""Built-in converters; import submodules to populate the registry."""

from . import action, classification, coco, detection, imagenet, labelbee, ocr, sequence, yolo

__all__ = [
    "action",
    "classification",
    "detection",
    "sequence",
    "coco",
    "yolo",
    "imagenet",
    "labelbee",
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

from .action import MonolithUpActionConverter
from .classification import MonolithUpClassificationConverter
from .coco import (
    CocoDetectionConverter,
    CocoKeypointConverter,
    CocoSegmentationConverter,
)
from .detection import MonolithUpDetectionConverter
from .imagenet import ImagenetClassificationConverter
from .labelbee import (
    LabelBeeClassificationConverter,
    LabelBeeDetectionConverter,
    LabelBeeKeypointConverter,
    LabelBeeSegmentationConverter,
)
from .ocr import OcrSequenceConverter
from .sequence import MonolithUpSequenceConverter
from .yolo import YoloDetectionConverter
