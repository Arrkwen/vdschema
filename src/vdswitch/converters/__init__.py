"""Built-in converters; import submodules to populate the registry."""

from . import action, classification, coco, detection, imagenet, ocr, sequence, yolo

__all__ = [
    "action",
    "classification",
    "detection",
    "sequence",
    "coco",
    "yolo",
    "imagenet",
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
from .ocr import OcrSequenceConverter
from .sequence import MonolithUpSequenceConverter
from .yolo import YoloDetectionConverter
