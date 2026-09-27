"""Built-in converters; import submodules to populate the registry."""

from . import action, classification, coco, detection, sequence, yolo

__all__ = [
    "action",
    "classification",
    "detection",
    "sequence",
    "coco",
    "yolo",
    "MonolithUpActionConverter",
    "MonolithUpClassificationConverter",
    "MonolithUpDetectionConverter",
    "MonolithUpSequenceConverter",
    "CocoDetectionConverter",
    "CocoKeypointConverter",
    "CocoSegmentationConverter",
    "YoloDetectionConverter",
]

from .action import MonolithUpActionConverter
from .classification import MonolithUpClassificationConverter
from .coco import (
    CocoDetectionConverter,
    CocoKeypointConverter,
    CocoSegmentationConverter,
)
from .detection import MonolithUpDetectionConverter
from .sequence import MonolithUpSequenceConverter
from .yolo import YoloDetectionConverter
