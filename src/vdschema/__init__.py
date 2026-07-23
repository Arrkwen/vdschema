"""Unified annotation schema and JSONL IO."""

from ._version import __version__
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

__all__ = [
    "__version__",
    "AnnotationWriter",
    "AnnotationReader",
    "TaskType",
    "AnnotationFormatError",
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
]
