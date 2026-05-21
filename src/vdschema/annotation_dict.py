"""Per-task label vocabularies aligned with annotation_dict.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .annotation_format import (
    ActionAnnotation,
    AnnotationFormatError,
    BaseAnnotation,
    ClassificationAnnotation,
    RelationshipAnnotation,
    TaskType,
)

SCHEMA_BASE_URL = (
    "https://github.com/Arrkwen/vdschema/blob/main/src/vdschema/schema"
)
ANNOTATION_SCHEMA_ID = f"{SCHEMA_BASE_URL}/annotation_schema.json"
ANNOTATION_DATA_FILENAME = "annotation_meta.jsonl"
ANNOTATION_DICT_FILENAME = "annotation_dict.json"


def meta_path_for(
    task_dir: str | Path,
    task_meta_filename: str = ANNOTATION_DATA_FILENAME,
) -> Path:
    """Default JSONL path under a task directory."""
    return Path(task_dir) / task_meta_filename


def dict_path_for(
    task_dir: str | Path,
    task_dict_name: str = ANNOTATION_DICT_FILENAME,
) -> Path:
    """Default label dictionary path under a task directory."""
    return Path(task_dir) / task_dict_name


@dataclass(frozen=True)
class CategoryMap:
    """Bidirectional mapping: ``id -> name``."""

    id_to_name: dict[int, str]
    name_to_id: dict[str, int]

    @classmethod
    def from_dict(cls, mapping: dict[int, str]) -> CategoryMap:
        id_to_name: dict[int, str] = {}
        name_to_id: dict[str, int] = {}
        for raw_id, name in mapping.items():
            label_id = int(raw_id)
            label_name = str(name)
            if label_id in id_to_name:
                raise AnnotationFormatError(f"duplicate category id={label_id}")
            if label_name in name_to_id:
                raise AnnotationFormatError(f"duplicate category name={label_name!r}")
            id_to_name[label_id] = label_name
            name_to_id[label_name] = label_id
        return cls(id_to_name=id_to_name, name_to_id=name_to_id)

    def to_entries(self, *, id_key: str, name_key: str) -> list[dict[str, Any]]:
        return [
            {id_key: i, name_key: self.id_to_name[i]}
            for i in sorted(self.id_to_name)
        ]

    def to_category_entries(self) -> list[dict[str, Any]]:
        return self.to_entries(id_key="category_id", name_key="category_name")

    def to_relationship_entries(self) -> list[dict[str, Any]]:
        return self.to_entries(id_key="relationship_id", name_key="relationship_name")

    def to_action_entries(self) -> list[dict[str, Any]]:
        return self.to_entries(id_key="action_id", name_key="action_name")


def _entries_to_dict(
    entries: list[dict[str, Any]],
    *,
    id_key: str,
    name_key: str,
) -> dict[int, str]:
    return {int(entry[id_key]): str(entry[name_key]) for entry in entries}


class TaskLabelDict:
    """Base class for task-scoped label dictionaries."""

    def to_task_dict(self) -> dict[str, Any]:
        """Return labels in the same plain-dict shape as Writer ``task_dict``."""
        raise NotImplementedError

    def to_data(self) -> dict[str, Any]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_data(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        return


class DetectionLabelDict(TaskLabelDict):
    """Vocabulary for detection / keypoint / segmentation tasks."""

    def __init__(
        self,
        detection: dict[int, str],
        *,
        annotation_schema_ref: str = ANNOTATION_SCHEMA_ID,
    ) -> None:
        self.annotation_schema_ref = annotation_schema_ref
        self.detection = CategoryMap.from_dict(detection)

    def to_task_dict(self) -> dict[int, str]:
        return dict(self.detection.id_to_name)

    def to_data(self) -> dict[str, Any]:
        return {
            "annotation_schema_ref": self.annotation_schema_ref,
            "detection": self.detection.to_category_entries(),
        }

    @classmethod
    def load(cls, path: str | Path) -> DetectionLabelDict:
        raw = _read_json_object(path)
        if not raw.get("detection"):
            raise AnnotationFormatError(f"{path}: missing detection vocabulary")
        return cls(
            _entries_to_dict(
                raw["detection"],
                id_key="category_id",
                name_key="category_name",
            ),
            annotation_schema_ref=raw.get(
                "annotation_schema_ref", ANNOTATION_SCHEMA_ID
            ),
        )

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        for instance in annotation.instances:  # type: ignore[attr-defined]
            if instance.category_id not in self.detection.id_to_name:
                raise AnnotationFormatError(
                    f"unknown category_id={instance.category_id}"
                )


class ClassificationLabelDict(TaskLabelDict):
    """Vocabulary for image-level classification: ``{category_type: {id: name}}``."""

    def __init__(
        self,
        heads: dict[str, dict[int, str]],
        *,
        annotation_schema_ref: str = ANNOTATION_SCHEMA_ID,
    ) -> None:
        self.annotation_schema_ref = annotation_schema_ref
        self.heads = {
            category_type: CategoryMap.from_dict(category_map)
            for category_type, category_map in heads.items()
        }

    def to_task_dict(self) -> dict[str, dict[int, str]]:
        return {
            category_type: dict(vocab.id_to_name)
            for category_type, vocab in self.heads.items()
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "annotation_schema_ref": self.annotation_schema_ref,
            "classification": [
                {
                    "category_type": category_type,
                    "category_map": vocab.to_category_entries(),
                }
                for category_type, vocab in sorted(self.heads.items())
            ],
        }

    @classmethod
    def load(cls, path: str | Path) -> ClassificationLabelDict:
        raw = _read_json_object(path)
        if not raw.get("classification"):
            raise AnnotationFormatError(f"{path}: missing classification vocabulary")
        heads = {
            str(head["category_type"]): _entries_to_dict(
                head["category_map"],
                id_key="category_id",
                name_key="category_name",
            )
            for head in raw["classification"]
        }
        return cls(
            heads,
            annotation_schema_ref=raw.get(
                "annotation_schema_ref", ANNOTATION_SCHEMA_ID
            ),
        )

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, ClassificationAnnotation):
            return
        for head in annotation.categories:
            vocab = self.heads.get(head.category_type)
            if vocab is None:
                raise AnnotationFormatError(
                    f"unknown category_type={head.category_type!r}"
                )
            for category_id in head.category_ids:
                if category_id not in vocab.id_to_name:
                    raise AnnotationFormatError(
                        f"unknown category_id={category_id} "
                        f"for category_type={head.category_type!r}"
                    )


class RelationshipLabelDict(TaskLabelDict):
    """Vocabulary for relationship tasks (instance + relation labels)."""

    def __init__(
        self,
        detection: dict[int, str],
        relationship: dict[int, str],
        *,
        annotation_schema_ref: str = ANNOTATION_SCHEMA_ID,
    ) -> None:
        self.annotation_schema_ref = annotation_schema_ref
        self.detection = CategoryMap.from_dict(detection)
        self.relationship = CategoryMap.from_dict(relationship)

    def to_task_dict(self) -> dict[str, dict[int, str]]:
        return {
            "detection": dict(self.detection.id_to_name),
            "relationship": dict(self.relationship.id_to_name),
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "annotation_schema_ref": self.annotation_schema_ref,
            "detection": self.detection.to_category_entries(),
            "relationship": self.relationship.to_relationship_entries(),
        }

    @classmethod
    def load(cls, path: str | Path) -> RelationshipLabelDict:
        raw = _read_json_object(path)
        if not raw.get("detection") or not raw.get("relationship"):
            raise AnnotationFormatError(
                f"{path}: relationship task requires detection and relationship vocabulary"
            )
        return cls(
            _entries_to_dict(
                raw["detection"],
                id_key="category_id",
                name_key="category_name",
            ),
            _entries_to_dict(
                raw["relationship"],
                id_key="relationship_id",
                name_key="relationship_name",
            ),
            annotation_schema_ref=raw.get(
                "annotation_schema_ref", ANNOTATION_SCHEMA_ID
            ),
        )

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, RelationshipAnnotation):
            return
        for instance in annotation.instances:
            if instance.category_id not in self.detection.id_to_name:
                raise AnnotationFormatError(
                    f"unknown category_id={instance.category_id}"
                )
        for rel in annotation.relationships:
            if rel.relation_type not in self.relationship.name_to_id:
                raise AnnotationFormatError(
                    f"unknown relation_type={rel.relation_type!r}"
                )


class ActionLabelDict(TaskLabelDict):
    """Vocabulary for video action tasks."""

    def __init__(
        self,
        action: dict[int, str],
        *,
        annotation_schema_ref: str = ANNOTATION_SCHEMA_ID,
    ) -> None:
        self.annotation_schema_ref = annotation_schema_ref
        self.action = CategoryMap.from_dict(action)

    def to_task_dict(self) -> dict[int, str]:
        return dict(self.action.id_to_name)

    def to_data(self) -> dict[str, Any]:
        return {
            "annotation_schema_ref": self.annotation_schema_ref,
            "action": self.action.to_action_entries(),
        }

    @classmethod
    def load(cls, path: str | Path) -> ActionLabelDict:
        raw = _read_json_object(path)
        if not raw.get("action"):
            raise AnnotationFormatError(f"{path}: missing action vocabulary")
        return cls(
            _entries_to_dict(
                raw["action"],
                id_key="action_id",
                name_key="action_name",
            ),
            annotation_schema_ref=raw.get(
                "annotation_schema_ref", ANNOTATION_SCHEMA_ID
            ),
        )

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, ActionAnnotation):
            return
        for event in annotation.actions:
            if event.action_id not in self.action.id_to_name:
                raise AnnotationFormatError(f"unknown action_id={event.action_id}")


@dataclass(frozen=True)
class NoLabelDict(TaskLabelDict):
    """Placeholder for tasks without label vocabulary (VLM, conversation)."""

    def to_task_dict(self) -> dict[str, Any]:
        return {}

    def to_data(self) -> dict[str, Any]:
        return {}

    def save(self, path: str | Path) -> None:
        return


_LABEL_DICT_BY_TASK: dict[TaskType, type[TaskLabelDict]] = {
    TaskType.DETECTION: DetectionLabelDict,
    TaskType.KEYPOINT: DetectionLabelDict,
    TaskType.SEGMENTATION: DetectionLabelDict,
    TaskType.CLASSIFICATION: ClassificationLabelDict,
    TaskType.RELATIONSHIP: RelationshipLabelDict,
    TaskType.ACTION: ActionLabelDict,
    TaskType.VLM: NoLabelDict,
    TaskType.CONVERSATION: NoLabelDict,
}


def build_label_dict(
    task_type: TaskType,
    task_dict: dict[str, Any] | dict[int, str] | None,
) -> TaskLabelDict:
    """Build a task-specific label dictionary from a plain ``task_dict``."""
    if not task_type.has_label_dict:
        return NoLabelDict()
    if task_dict is None:
        raise AnnotationFormatError(f"{task_type.value} requires task_dict")

    if task_type is TaskType.RELATIONSHIP:
        if not isinstance(task_dict, dict):
            raise AnnotationFormatError("relationship task_dict must be a mapping")
        try:
            return RelationshipLabelDict(
                detection=task_dict["detection"],
                relationship=task_dict["relationship"],
            )
        except KeyError as exc:
            raise AnnotationFormatError(
                "relationship task_dict requires 'detection' and 'relationship'"
            ) from exc

    label_cls = _LABEL_DICT_BY_TASK[task_type]
    return label_cls(task_dict)  # type: ignore[arg-type,call-arg]


def load_label_dict(
    task_type: TaskType,
    task_dir: str | Path,
    *,
    task_dict_name: str = ANNOTATION_DICT_FILENAME,
) -> TaskLabelDict | None:
    """Load ``task_dict_name`` under ``task_dir``; return ``None`` when not applicable."""
    if not task_type.has_label_dict:
        return None
    dict_path = dict_path_for(task_dir, task_dict_name)
    if not dict_path.is_file():
        raise AnnotationFormatError(f"missing label dictionary: {dict_path}")
    return _LABEL_DICT_BY_TASK[task_type].load(dict_path)  # type: ignore[attr-defined]


def _read_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AnnotationFormatError(f"{path}: label dictionary must be a JSON object")
    return data
