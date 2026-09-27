"""Unified annotation schema and JSONL IO."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("vdschema")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
from .annotation_dict import Name
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
from .annotation_io import AnnotationReader, AnnotationWriter

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


def __getattr__(name: str) -> Any:
    if name == "switch":
        from vdswitch import switch as switch_fn

        return switch_fn
    if name == "Source":
        from vdswitch.converters.sources import Source as source_enum

        return source_enum
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
