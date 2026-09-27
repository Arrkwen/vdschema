"""Task-specific annotation record types and TaskType registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ._annotation_components import (
    ActionEvent,
    AnnotationFormatError,
    ClassificationHead,
    ConversationTurn,
    Instance,
    Relationship,
    _action_event,
    _classification_head,
    _conversation_turn,
    _ensure_positive_int,
    _instance,
    _relationship,
)


@dataclass(kw_only=True)
class BaseAnnotation:
    """Common root fields shared by all annotation records."""

    filename: str
    width: int
    height: int

    def validate(self) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BaseAnnotation:
        raise NotImplementedError

    def validate_base(self) -> None:
        if not self.filename:
            raise AnnotationFormatError("filename must not be empty")
        _ensure_positive_int(self.width, "width")
        _ensure_positive_int(self.height, "height")

    def base_dict(self) -> dict[str, Any]:
        self.validate_base()
        return {
            "filename": self.filename,
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def _base_kwargs(cls, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "filename": raw["filename"],
            "width": int(raw["width"]),
            "height": int(raw["height"]),
        }


@dataclass(kw_only=True)
class DetectionAnnotation(BaseAnnotation):
    instances: list[Any] = field(default_factory=list)
    description: str | None = None

    def __post_init__(self) -> None:
        self.instances = [_instance(item) for item in self.instances]

    def validate(self) -> None:
        self.validate_base()
        for instance in self.instances:
            instance.validate()
            if (
                instance.bbox is None
                and instance.polygon is None
                and instance.polyline is None
                and instance.point is None
            ):
                raise AnnotationFormatError(
                    "detection instances require bbox, polygon, polyline, or point"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out = {
            **self.base_dict(),
            "instances": [item.to_dict() for item in self.instances],
        }
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DetectionAnnotation:
        return cls(
            **cls._base_kwargs(raw),
            instances=[Instance.from_dict(item) for item in raw["instances"]],
            description=raw.get("description"),
        )


@dataclass(kw_only=True)
class KeypointAnnotation(DetectionAnnotation):
    def validate(self) -> None:
        self.validate_base()
        for instance in self.instances:
            instance.validate()
            if not instance.keypoints:
                raise AnnotationFormatError(
                    "keypoint annotations require instance.keypoints"
                )


@dataclass(kw_only=True)
class SegmentationAnnotation(DetectionAnnotation):
    def validate(self) -> None:
        self.validate_base()
        for instance in self.instances:
            instance.validate()
            if instance.rle_mask is None and instance.polygon is None:
                raise AnnotationFormatError(
                    "segmentation instances require rle_mask or polygon"
                )


@dataclass(kw_only=True)
class ClassificationAnnotation(BaseAnnotation):
    categories: list[Any] = field(default_factory=list)
    description: str | None = None

    def __post_init__(self) -> None:
        self.categories = [_classification_head(item) for item in self.categories]

    def validate(self) -> None:
        self.validate_base()
        for category in self.categories:
            category.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out = {
            **self.base_dict(),
            "categories": [item.to_dict() for item in self.categories],
        }
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ClassificationAnnotation:
        return cls(
            **cls._base_kwargs(raw),
            categories=[
                ClassificationHead.from_dict(item) for item in raw["categories"]
            ],
            description=raw.get("description"),
        )


@dataclass(kw_only=True)
class RelationshipAnnotation(BaseAnnotation):
    instances: list[Any] = field(default_factory=list)
    relationships: list[Any] = field(default_factory=list)
    description: str | None = None

    def __post_init__(self) -> None:
        self.instances = [_instance(item) for item in self.instances]
        self.relationships = [_relationship(item) for item in self.relationships]

    def validate(self) -> None:
        self.validate_base()
        if not self.instances and self.relationships:
            raise AnnotationFormatError(
                "instances is empty when relationships is not empty"
            )
        if not self.relationships and self.instances:
            raise AnnotationFormatError(
                "relationships is empty when instances is not empty"
            )
        instance_ids = {instance.id for instance in self.instances}
        for instance in self.instances:
            instance.validate()
            if instance.bbox is None and instance.polygon is None:
                raise AnnotationFormatError(
                    "relationship instances require bbox or polygon"
                )
        for relationship in self.relationships:
            relationship.validate()
            if relationship.subject_id not in instance_ids:
                raise AnnotationFormatError(
                    "relationship.subject_id must reference instances[].id"
                )
            if relationship.object_id not in instance_ids:
                raise AnnotationFormatError(
                    "relationship.object_id must reference instances[].id"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out = {
            **self.base_dict(),
            "instances": [item.to_dict() for item in self.instances],
            "relationships": [item.to_dict() for item in self.relationships],
        }
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RelationshipAnnotation:
        return cls(
            **cls._base_kwargs(raw),
            instances=[Instance.from_dict(item) for item in raw["instances"]],
            relationships=[
                Relationship.from_dict(item) for item in raw["relationships"]
            ],
            description=raw.get("description"),
        )


@dataclass(kw_only=True)
class VlmAnnotation(BaseAnnotation):
    description: str

    def validate(self) -> None:
        self.validate_base()
        if not self.description:
            raise AnnotationFormatError("description must not be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self.base_dict(), "description": self.description}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> VlmAnnotation:
        return cls(**cls._base_kwargs(raw), description=raw["description"])


@dataclass(kw_only=True)
class ConversationAnnotation(BaseAnnotation):
    conversations: list[Any] = field(default_factory=list)
    description: str | None = None

    def __post_init__(self) -> None:
        self.conversations = [_conversation_turn(item) for item in self.conversations]

    def validate(self) -> None:
        self.validate_base()
        if not self.conversations:
            raise AnnotationFormatError("conversations must not be empty")
        for turn in self.conversations:
            turn.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out = {
            **self.base_dict(),
            "conversations": [item.to_dict() for item in self.conversations],
        }
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ConversationAnnotation:
        return cls(
            **cls._base_kwargs(raw),
            conversations=[
                ConversationTurn.from_dict(item) for item in raw["conversations"]
            ],
            description=raw.get("description"),
        )


@dataclass(kw_only=True)
class SequenceAnnotation(BaseAnnotation):
    """Ordered token sequence associated with an image."""

    sequences: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sequences = [str(item) for item in self.sequences]

    def validate(self) -> None:
        self.validate_base()
        if any(not item for item in self.sequences):
            raise AnnotationFormatError("sequences must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self.base_dict(), "sequences": list(self.sequences)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SequenceAnnotation:
        return cls(**cls._base_kwargs(raw), sequences=raw["sequences"])


@dataclass(kw_only=True)
class ActionAnnotation(BaseAnnotation):
    actions: list[Any] = field(default_factory=list)
    description: str | None = None

    def __post_init__(self) -> None:
        self.actions = [_action_event(item) for item in self.actions]

    def validate(self) -> None:
        self.validate_base()
        if not self.actions:
            raise AnnotationFormatError("actions must not be empty")
        for action in self.actions:
            action.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out = {
            **self.base_dict(),
            "actions": [item.to_dict() for item in self.actions],
        }
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ActionAnnotation:
        return cls(
            **cls._base_kwargs(raw),
            actions=[ActionEvent.from_dict(item) for item in raw["actions"]],
            description=raw.get("description"),
        )


class TaskType(str, Enum):
    """Supported annotation task kinds."""

    DETECTION = "detection"
    KEYPOINT = "keypoint"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    RELATIONSHIP = "relationship"
    VLM = "vlm"
    CONVERSATION = "conversation"
    SEQUENCE = "sequence"
    ACTION = "action"

    @property
    def annotation_class(self) -> type[BaseAnnotation]:
        try:
            return _TASK_ANNOTATION[self]
        except KeyError as exc:
            raise AnnotationFormatError(
                f"unsupported task type: {self.value!r}"
            ) from exc

    @property
    def has_label_dict(self) -> bool:
        return self not in _TASKS_WITHOUT_LABEL_DICT


_TASKS_WITHOUT_LABEL_DICT = frozenset({TaskType.VLM, TaskType.CONVERSATION})

_TASK_ANNOTATION: dict[TaskType, type[BaseAnnotation]] = {
    TaskType.DETECTION: DetectionAnnotation,
    TaskType.KEYPOINT: KeypointAnnotation,
    TaskType.SEGMENTATION: SegmentationAnnotation,
    TaskType.CLASSIFICATION: ClassificationAnnotation,
    TaskType.RELATIONSHIP: RelationshipAnnotation,
    TaskType.VLM: VlmAnnotation,
    TaskType.CONVERSATION: ConversationAnnotation,
    TaskType.SEQUENCE: SequenceAnnotation,
    TaskType.ACTION: ActionAnnotation,
}
