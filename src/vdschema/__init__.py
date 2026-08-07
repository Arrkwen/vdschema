"""Unified annotation schema and JSONL IO."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vdschema")
except PackageNotFoundError:
    __version__ = "unknown"
from .annotation_dict import Name
from .annotation_io import AnnotationReader, AnnotationWriter
from .annotation_format import (
    ActionAnnotation,
    AnnotationFormatError,
    BaseAnnotation,
    Bbox,
    ClassificationAnnotation,
    ConversationAnnotation,
    ConversationRole,
    ConversationTurn,
    DetectionAnnotation,
    Keypoint,
    KeypointAnnotation,
    RelationshipAnnotation,
    SegmentationAnnotation,
    SegmentationRLE,
    SequenceAnnotation,
    TaskType,
    VlmAnnotation,
)
from vdswitch import switch
from vdswitch.converters.sources import Source

__all__ = [
    "__version__",
    "AnnotationWriter",
    "AnnotationReader",
    "TaskType",
    "AnnotationFormatError",
    "Name",
    "BaseAnnotation",
    "DetectionAnnotation",
    "KeypointAnnotation",
    "SegmentationAnnotation",
    "ClassificationAnnotation",
    "RelationshipAnnotation",
    "VlmAnnotation",
    "ConversationAnnotation",
    "SequenceAnnotation",
    "ActionAnnotation",
    "Bbox",
    "Keypoint",
    "SegmentationRLE",
    "ConversationRole",
    "ConversationTurn",
    "Source",
    "switch",
]
