"""
Internal data structures aligned with annotation_data.

Application code usually does not need to instantiate these classes directly.
annotation_format converts plain bbox lists, dicts, mask arrays, and other
basic inputs into these structures.
"""

from __future__ import annotations

from ._annotation_components import (
    ActionEvent,
    AnnotationFormatError,
    Bbox,
    ClassificationHead,
    ContentImage,
    ContentPart,
    ContentText,
    ConversationRole,
    ConversationTurn,
    Instance,
    Keypoint,
    Relationship,
    SegmentationRLE,
    TrackItem,
)
from ._annotation_records import (
    ActionAnnotation,
    BaseAnnotation,
    ClassificationAnnotation,
    ConversationAnnotation,
    DetectionAnnotation,
    KeypointAnnotation,
    RelationshipAnnotation,
    SegmentationAnnotation,
    SequenceAnnotation,
    TaskType,
    VlmAnnotation,
)

__all__ = [
    "ActionAnnotation",
    "ActionEvent",
    "AnnotationFormatError",
    "BaseAnnotation",
    "Bbox",
    "ClassificationAnnotation",
    "ClassificationHead",
    "ContentImage",
    "ContentPart",
    "ContentText",
    "ConversationAnnotation",
    "ConversationRole",
    "ConversationTurn",
    "DetectionAnnotation",
    "Instance",
    "Keypoint",
    "KeypointAnnotation",
    "Relationship",
    "RelationshipAnnotation",
    "SegmentationAnnotation",
    "SegmentationRLE",
    "SequenceAnnotation",
    "TaskType",
    "TrackItem",
    "VlmAnnotation",
]
