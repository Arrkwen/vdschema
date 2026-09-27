"""Built-in converters; import submodules to populate the registry."""

from . import coco, imagenet, labelbee, labelme, monolith, ocr, voc, yolo

__all__ = [
    "coco",
    "yolo",
    "imagenet",
    "labelbee",
    "labelme",
    "monolith",
    "ocr",
    "voc",
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
    "LabelMeDetectionConverter",
    "LabelMeKeypointConverter",
    "LabelMeSegmentationConverter",
    "VocDetectionConverter",
    "VocSegmentationConverter",
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
from .labelme import (
    LabelMeDetectionConverter,
    LabelMeKeypointConverter,
    LabelMeSegmentationConverter,
)
from .monolith import (
    MonolithUpActionConverter,
    MonolithUpClassificationConverter,
    MonolithUpDetectionConverter,
    MonolithUpSequenceConverter,
)
from .ocr import OcrSequenceConverter
from .voc import VocDetectionConverter, VocSegmentationConverter
from .yolo import YoloDetectionConverter
