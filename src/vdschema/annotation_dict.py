"""Per-task label vocabularies aligned with annotation_meta.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

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
ANNOTATION_SCHEMA_ID = f"{SCHEMA_BASE_URL}/annotation_data.json"
ANNOTATION_DATA_FILENAME = "annotation_data.jsonl"
ANNOTATION_META_FILENAME = "annotation_meta.json"


def data_path_for(
    task_dir: str | Path,
    task_data_filename: str = ANNOTATION_DATA_FILENAME,
) -> Path:
    """Default annotation JSONL path under a task directory."""
    return Path(task_dir) / task_data_filename


def meta_path_for(
    task_dir: str | Path,
    task_meta_filename: str = ANNOTATION_META_FILENAME,
) -> Path:
    """Default label-meta (vocabulary) path under a task directory."""
    return Path(task_dir) / task_meta_filename


@dataclass(frozen=True)
class Name:
    """A canonical label name with optional aliases and prompts."""

    name: str
    alias: tuple[str, ...] = field(default_factory=tuple)
    prompt: tuple[str, ...] = field(default_factory=tuple)

    def __init__(
        self,
        name: str,
        *,
        alias: Sequence[str] | None = None,
        prompt: Sequence[str] | None = None,
    ) -> None:
        object.__setattr__(self, "name", str(name).strip())
        object.__setattr__(
            self,
            "alias",
            tuple(str(item).strip() for item in (alias or ())),
        )
        object.__setattr__(
            self,
            "prompt",
            tuple(str(item).strip() for item in (prompt or ())),
        )
        if not self.name:
            raise AnnotationFormatError("name must not be empty")
        if any(not item for item in self.alias):
            raise AnnotationFormatError("alias must contain non-empty strings")
        if any(not item for item in self.prompt):
            raise AnnotationFormatError("prompt must contain non-empty strings")


def _coerce_category(value: Name | str | Any) -> Name:
    if isinstance(value, Name):
        return value
    if isinstance(value, str):
        return Name(value)
    raise AnnotationFormatError(
        "label values must be Name(...) or str, "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class CategoryMap:
    """Bidirectional mapping: ``id -> Name``."""

    id_to_category: dict[int, Name]
    name_to_id: dict[str, int]

    @property
    def id_to_name(self) -> dict[int, str]:
        return {i: cat.name for i, cat in self.id_to_category.items()}

    @classmethod
    def from_dict(cls, mapping: Mapping[int, Name | str]) -> CategoryMap:
        id_to_category: dict[int, Name] = {}
        name_to_id: dict[str, int] = {}
        for raw_id, value in mapping.items():
            label_id = int(raw_id)
            category = _coerce_category(value)
            if label_id in id_to_category:
                raise AnnotationFormatError(f"duplicate category id={label_id}")
            _register_name(name_to_id, category.name, label_id)
            for alias in category.alias:
                _register_name(name_to_id, alias, label_id)
            id_to_category[label_id] = category
        return cls(id_to_category=id_to_category, name_to_id=name_to_id)

    def to_label(self) -> dict[int, Name]:
        return dict(self.id_to_category)

    def to_entries(
        self,
        *,
        id_key: str,
        name_key: str,
        alias_key: str,
        prompt_key: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for label_id in sorted(self.id_to_category):
            category = self.id_to_category[label_id]
            entry: dict[str, Any] = {
                id_key: label_id,
                name_key: category.name,
            }
            if category.alias:
                entry[alias_key] = list(category.alias)
            if category.prompt:
                entry[prompt_key] = list(category.prompt)
            entries.append(entry)
        return entries

    def to_category_entries(self) -> list[dict[str, Any]]:
        return self.to_entries(
            id_key="category_id",
            name_key="category_name",
            alias_key="category_alias",
            prompt_key="category_prompt",
        )

    def to_relationship_entries(self) -> list[dict[str, Any]]:
        return self.to_entries(
            id_key="relationship_id",
            name_key="relationship_name",
            alias_key="relationship_alias",
            prompt_key="relationship_prompt",
        )


def _register_name(name_to_id: dict[str, int], name: str, label_id: int) -> None:
    if name in name_to_id:
        raise AnnotationFormatError(f"duplicate category name/alias={name!r}")
    name_to_id[name] = label_id


def _entries_to_label(
    entries: list[dict[str, Any]],
    *,
    id_key: str,
    name_key: str,
    alias_key: str,
    prompt_key: str,
) -> dict[int, Name]:
    return {
        int(entry[id_key]): Name(
            entry[name_key],
            alias=entry.get(alias_key) or (),
            prompt=entry.get(prompt_key) or (),
        )
        for entry in entries
    }


class TaskLabelDict:
    """Base class for task-scoped label dictionaries."""

    def to_label(self) -> dict[str, Any]:
        """Return labels in the same shape as Writer ``label``."""
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
        detection: Mapping[int, Name | str],
        *,
        annotation_schema_ref: str = ANNOTATION_SCHEMA_ID,
    ) -> None:
        self.annotation_schema_ref = annotation_schema_ref
        self.detection = CategoryMap.from_dict(detection)

    def to_label(self) -> dict[int, Name]:
        return self.detection.to_label()

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
            _entries_to_label(
                raw["detection"],
                id_key="category_id",
                name_key="category_name",
                alias_key="category_alias",
                prompt_key="category_prompt",
            ),
            annotation_schema_ref=raw.get(
                "annotation_schema_ref", ANNOTATION_SCHEMA_ID
            ),
        )

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        for instance in annotation.instances:  # type: ignore[attr-defined]
            if instance.category_id not in self.detection.id_to_category:
                raise AnnotationFormatError(
                    f"unknown category_id={instance.category_id}"
                )


class ClassificationLabelDict(TaskLabelDict):
    """Vocabulary for image-level classification: ``{category_attr: {id: Name}}``."""

    def __init__(
        self,
        heads: Mapping[str, Mapping[int, Name | str]],
        *,
        annotation_schema_ref: str = ANNOTATION_SCHEMA_ID,
    ) -> None:
        self.annotation_schema_ref = annotation_schema_ref
        self.heads = {
            category_attr: CategoryMap.from_dict(category_map)
            for category_attr, category_map in heads.items()
        }

    def to_label(self) -> dict[str, dict[int, Name]]:
        return {
            category_attr: vocab.to_label()
            for category_attr, vocab in self.heads.items()
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "annotation_schema_ref": self.annotation_schema_ref,
            "classification": [
                {
                    "category_attr": category_attr,
                    "category_label": vocab.to_category_entries(),
                }
                for category_attr, vocab in sorted(self.heads.items())
            ],
        }

    @classmethod
    def load(cls, path: str | Path) -> ClassificationLabelDict:
        raw = _read_json_object(path)
        if not raw.get("classification"):
            raise AnnotationFormatError(f"{path}: missing classification vocabulary")
        heads = {
            str(head["category_attr"]): _entries_to_label(
                head["category_label"],
                id_key="category_id",
                name_key="category_name",
                alias_key="category_alias",
                prompt_key="category_prompt",
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
            vocab = self.heads.get(head.category_attr)
            if vocab is None:
                raise AnnotationFormatError(
                    f"unknown category_attr={head.category_attr!r}"
                )
            for category_id in head.category_ids:
                if category_id not in vocab.id_to_category:
                    raise AnnotationFormatError(
                        f"unknown category_id={category_id} "
                        f"for category_attr={head.category_attr!r}"
                    )


class RelationshipLabelDict(TaskLabelDict):
    """Vocabulary for relationship tasks (instance + relation labels)."""

    def __init__(
        self,
        detection: Mapping[int, Name | str],
        relationship: Mapping[int, Name | str],
        *,
        annotation_schema_ref: str = ANNOTATION_SCHEMA_ID,
    ) -> None:
        self.annotation_schema_ref = annotation_schema_ref
        self.detection = CategoryMap.from_dict(detection)
        self.relationship = CategoryMap.from_dict(relationship)

    def to_label(self) -> dict[str, dict[int, Name]]:
        return {
            "detection": self.detection.to_label(),
            "relationship": self.relationship.to_label(),
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
            _entries_to_label(
                raw["detection"],
                id_key="category_id",
                name_key="category_name",
                alias_key="category_alias",
                prompt_key="category_prompt",
            ),
            _entries_to_label(
                raw["relationship"],
                id_key="relationship_id",
                name_key="relationship_name",
                alias_key="relationship_alias",
                prompt_key="relationship_prompt",
            ),
            annotation_schema_ref=raw.get(
                "annotation_schema_ref", ANNOTATION_SCHEMA_ID
            ),
        )

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, RelationshipAnnotation):
            return
        for instance in annotation.instances:
            if instance.category_id not in self.detection.id_to_category:
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
        action: Mapping[int, Name | str],
        *,
        annotation_schema_ref: str = ANNOTATION_SCHEMA_ID,
    ) -> None:
        self.annotation_schema_ref = annotation_schema_ref
        self.action = CategoryMap.from_dict(action)

    def to_label(self) -> dict[int, Name]:
        return self.action.to_label()

    def to_data(self) -> dict[str, Any]:
        return {
            "annotation_schema_ref": self.annotation_schema_ref,
            "action": self.action.to_category_entries(),
        }

    @classmethod
    def load(cls, path: str | Path) -> ActionLabelDict:
        raw = _read_json_object(path)
        if not raw.get("action"):
            raise AnnotationFormatError(f"{path}: missing action vocabulary")
        return cls(
            _entries_to_label(
                raw["action"],
                id_key="category_id",
                name_key="category_name",
                alias_key="category_alias",
                prompt_key="category_prompt",
            ),
            annotation_schema_ref=raw.get(
                "annotation_schema_ref", ANNOTATION_SCHEMA_ID
            ),
        )

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, ActionAnnotation):
            return
        for event in annotation.actions:
            if event.category_id not in self.action.id_to_category:
                raise AnnotationFormatError(f"unknown category_id={event.category_id}")


@dataclass(frozen=True)
class NoLabelDict(TaskLabelDict):
    """Placeholder for tasks without label vocabulary (VLM, conversation, sequence)."""

    def to_label(self) -> dict[str, Any]:
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
    TaskType.SEQUENCE: NoLabelDict,
}


def build_label_dict(
    task_type: TaskType,
    label: dict[str, Any] | Mapping[int, Name | str] | None,
) -> TaskLabelDict:
    """Build a task-specific label dictionary from Writer ``label``."""
    if not task_type.has_label_dict:
        return NoLabelDict()
    if label is None:
        raise AnnotationFormatError(f"{task_type.value} requires label")

    if task_type is TaskType.RELATIONSHIP:
        if not isinstance(label, dict):
            raise AnnotationFormatError("relationship label must be a mapping")
        try:
            return RelationshipLabelDict(
                detection=label["detection"],
                relationship=label["relationship"],
            )
        except KeyError as exc:
            raise AnnotationFormatError(
                "relationship label requires 'detection' and 'relationship'"
            ) from exc

    label_cls = _LABEL_DICT_BY_TASK[task_type]
    return label_cls(label)  # type: ignore[arg-type,call-arg]


def load_label_dict(
    task_type: TaskType,
    task_dir: str | Path,
    *,
    task_meta_filename: str = ANNOTATION_META_FILENAME,
) -> TaskLabelDict | None:
    """Load ``task_meta_filename`` under ``task_dir``; return ``None`` when not applicable."""
    if not task_type.has_label_dict:
        return None
    meta_path = meta_path_for(task_dir, task_meta_filename)
    if not meta_path.is_file():
        raise AnnotationFormatError(f"missing label dictionary: {meta_path}")
    return _LABEL_DICT_BY_TASK[task_type].load(meta_path)  # type: ignore[attr-defined]


def _read_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AnnotationFormatError(f"{path}: label dictionary must be a JSON object")
    return data
