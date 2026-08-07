"""Per-task label vocabularies aligned with annotation_meta.json."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from .annotation_format import (
    ActionAnnotation,
    AnnotationFormatError,
    BaseAnnotation,
    ClassificationAnnotation,
    DetectionAnnotation,
    RelationshipAnnotation,
    SequenceAnnotation,
    TaskType,
)

SCHEMA_BASE_URL = (
    "https://github.com/Arrkwen/vdschema/blob/main/src/vdschema/schema"
)
ANNOTATION_SCHEMA_ID = f"{SCHEMA_BASE_URL}/annotation_data.json"
ANNOTATION_DATA_FILENAME = "annotation_data.jsonl"
ANNOTATION_META_FILENAME = "annotation_meta.json"
ANNOTATION_VOCAB_FILENAME = "annotation_vocab.txt"

_CATEGORY_ENTRY_KEYS = {
    "id_key": "category_id",
    "name_key": "category_name",
    "alias_key": "category_alias",
    "prompt_key": "category_prompt",
}
_RELATIONSHIP_ENTRY_KEYS = {
    "id_key": "relationship_id",
    "name_key": "relationship_name",
    "alias_key": "relationship_alias",
    "prompt_key": "relationship_prompt",
}


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


def vocab_path_for(
    task_dir: str | Path,
    task_vocab_filename: str = ANNOTATION_VOCAB_FILENAME,
) -> Path:
    """Default sequence vocabulary path under a task directory."""
    return Path(task_dir) / task_vocab_filename


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


def _coerce_name(value: Name | str | Any) -> Name:
    if isinstance(value, Name):
        return value
    if isinstance(value, str):
        return Name(value)
    raise AnnotationFormatError(
        "label values must be Name(...) or str, "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class LabelMap:
    """Bidirectional mapping: ``id -> Name``."""

    id_to_entry: dict[int, Name]
    name_to_id: dict[str, int]

    @classmethod
    def from_id_map(cls, mapping: Mapping[int, Name | str]) -> LabelMap:
        id_to_entry: dict[int, Name] = {}
        name_to_id: dict[str, int] = {}
        for raw_id, value in mapping.items():
            label_id = int(raw_id)
            entry = _coerce_name(value)
            if label_id in id_to_entry:
                raise AnnotationFormatError(f"duplicate category id={label_id}")
            _register_name(name_to_id, entry.name, label_id)
            for alias in entry.alias:
                _register_name(name_to_id, alias, label_id)
            id_to_entry[label_id] = entry
        return cls(id_to_entry=id_to_entry, name_to_id=name_to_id)

    def to_label(self) -> dict[int, Name]:
        return dict(self.id_to_entry)

    def to_entries(
        self,
        *,
        id_key: str,
        name_key: str,
        alias_key: str,
        prompt_key: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for label_id in sorted(self.id_to_entry):
            item = self.id_to_entry[label_id]
            entry: dict[str, Any] = {
                id_key: label_id,
                name_key: item.name,
            }
            if item.alias:
                entry[alias_key] = list(item.alias)
            if item.prompt:
                entry[prompt_key] = list(item.prompt)
            entries.append(entry)
        return entries

    def to_category_entries(self) -> list[dict[str, Any]]:
        return self.to_entries(**_CATEGORY_ENTRY_KEYS)

    def to_relationship_entries(self) -> list[dict[str, Any]]:
        return self.to_entries(**_RELATIONSHIP_ENTRY_KEYS)


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


def _category_entries_to_label(entries: list[dict[str, Any]]) -> dict[int, Name]:
    return _entries_to_label(entries, **_CATEGORY_ENTRY_KEYS)


def _relationship_entries_to_label(entries: list[dict[str, Any]]) -> dict[int, Name]:
    return _entries_to_label(entries, **_RELATIONSHIP_ENTRY_KEYS)


def _schema_ref(raw: dict[str, Any]) -> str:
    return raw.get("annotation_schema_ref", ANNOTATION_SCHEMA_ID)


def _require_key(raw: dict[str, Any], path: str | Path, key: str) -> Any:
    value = raw.get(key)
    if not value:
        raise AnnotationFormatError(f"{path}: missing {key} vocabulary")
    return value


def _validate_category_ids(
    category_ids: Sequence[int],
    vocab: LabelMap,
    *,
    context: str | None = None,
) -> None:
    for category_id in category_ids:
        if category_id not in vocab.id_to_entry:
            suffix = f" {context}" if context else ""
            raise AnnotationFormatError(f"unknown category_id={category_id}{suffix}")


def _validate_instances(instances: Sequence[Any], vocab: LabelMap) -> None:
    _validate_category_ids(
        [instance.category_id for instance in instances],
        vocab,
    )


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


class SingleVocabLabelDict(TaskLabelDict):
    """Shared base for tasks with one flat ``id -> Name`` vocabulary."""

    _vocab_key: ClassVar[str]

    def __init__(
        self,
        mapping: Mapping[int, Name | str],
        *,
        annotation_schema_ref: str = ANNOTATION_SCHEMA_ID,
    ) -> None:
        self.annotation_schema_ref = annotation_schema_ref
        self.vocab = LabelMap.from_id_map(mapping)

    def to_label(self) -> dict[int, Name]:
        return self.vocab.to_label()

    def to_data(self) -> dict[str, Any]:
        return {
            "annotation_schema_ref": self.annotation_schema_ref,
            self._vocab_key: self.vocab.to_category_entries(),
        }

    @classmethod
    def load(cls, path: str | Path) -> SingleVocabLabelDict:
        raw = _read_json_object(path)
        entries = _require_key(raw, path, cls._vocab_key)
        return cls(
            _category_entries_to_label(entries),
            annotation_schema_ref=_schema_ref(raw),
        )


class DetectionLabelDict(SingleVocabLabelDict):
    """Vocabulary for detection / keypoint / segmentation tasks."""

    _vocab_key = "detection"

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, DetectionAnnotation):
            return
        _validate_instances(annotation.instances, self.vocab)


class ActionLabelDict(SingleVocabLabelDict):
    """Vocabulary for video action tasks."""

    _vocab_key = "action"

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, ActionAnnotation):
            return
        _validate_category_ids(
            [event.category_id for event in annotation.actions],
            self.vocab,
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
            category_attr: LabelMap.from_id_map(category_label)
            for category_attr, category_label in heads.items()
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
        heads_raw = _require_key(raw, path, "classification")
        heads = {
            str(head["category_attr"]): _category_entries_to_label(
                head["category_label"]
            )
            for head in heads_raw
        }
        return cls(heads, annotation_schema_ref=_schema_ref(raw))

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, ClassificationAnnotation):
            return
        for head in annotation.categories:
            vocab = self.heads.get(head.category_attr)
            if vocab is None:
                raise AnnotationFormatError(
                    f"unknown category_attr={head.category_attr!r}"
                )
            _validate_category_ids(
                head.category_ids,
                vocab,
                context=f"for category_attr={head.category_attr!r}",
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
        self.detection = LabelMap.from_id_map(detection)
        self.relationship = LabelMap.from_id_map(relationship)

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
            _category_entries_to_label(raw["detection"]),
            _relationship_entries_to_label(raw["relationship"]),
            annotation_schema_ref=_schema_ref(raw),
        )

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, RelationshipAnnotation):
            return
        _validate_instances(annotation.instances, self.detection)
        for rel in annotation.relationships:
            if rel.relation_type not in self.relationship.name_to_id:
                raise AnnotationFormatError(
                    f"unknown relation_type={rel.relation_type!r}"
                )


def _load_vocab_tokens(path: str | Path) -> list[str]:
    """Load one token per line from a vocabulary file."""
    tokens: list[str] = []
    seen: set[str] = set()
    vocab_path = Path(path)
    with vocab_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            content = line.rstrip("\n\r")
            if not content:
                continue
            if content.lstrip().startswith("#"):
                continue
            if content != content.strip():
                raise AnnotationFormatError(
                    f"{vocab_path}:{lineno} vocab line must not have leading or trailing whitespace"
                )
            if any(ch.isspace() for ch in content):
                raise AnnotationFormatError(
                    f"{vocab_path}:{lineno} vocab line must contain exactly one token"
                )
            token = content
            if token in seen:
                raise AnnotationFormatError(
                    f"{vocab_path}:{lineno} duplicate vocab token={token!r}"
                )
            seen.add(token)
            tokens.append(token)
    if not tokens:
        raise AnnotationFormatError(f"{vocab_path}: vocabulary file is empty")
    return tokens


class SequenceLabelDict(TaskLabelDict):
    """Token vocabulary for sequence tasks (external file only)."""

    def __init__(self, *, vocab_path: str | Path) -> None:
        path = Path(vocab_path)
        if not path.is_file():
            raise AnnotationFormatError(f"sequence vocab file not found: {path}")
        self.vocab_path = path
        self._tokens = _load_vocab_tokens(path)
        self._vocab_set = set(self._tokens)

    @property
    def vocab_set(self) -> set[str]:
        return self._vocab_set

    @classmethod
    def from_writer_label(
        cls,
        label: str | Path | Mapping[str, Any],
    ) -> SequenceLabelDict:
        if isinstance(label, (str, Path)):
            source = Path(label)
        elif isinstance(label, dict):
            if "vocab" not in label:
                raise AnnotationFormatError("sequence label requires a vocabulary file path")
            source = Path(label["vocab"])
        else:
            raise AnnotationFormatError("sequence label must be a vocabulary file path")
        if not source.is_file():
            raise AnnotationFormatError(f"sequence vocab file not found: {source}")
        return cls(vocab_path=source)

    def to_label(self) -> dict[str, Any]:
        return {"vocab": ANNOTATION_VOCAB_FILENAME}

    def to_data(self) -> dict[str, Any]:
        raise AnnotationFormatError("sequence vocabulary is stored in annotation_vocab.txt")

    def save(self, task_dir: str | Path) -> None:
        target_dir = Path(task_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.vocab_path, target_dir / ANNOTATION_VOCAB_FILENAME)

    @classmethod
    def load(cls, task_dir: str | Path) -> SequenceLabelDict:
        vocab_path = vocab_path_for(task_dir)
        if not vocab_path.is_file():
            raise AnnotationFormatError(f"missing vocabulary file: {vocab_path}")
        return cls(vocab_path=vocab_path)

    def validate_annotation(self, annotation: BaseAnnotation) -> None:
        if not isinstance(annotation, SequenceAnnotation):
            return
        for token in annotation.sequences:
            if token not in self.vocab_set:
                raise AnnotationFormatError(
                    f"unknown sequence token={token!r}; token must appear in vocabulary file"
                )


@dataclass(frozen=True)
class NoLabelDict(TaskLabelDict):
    """Placeholder for tasks without label vocabulary (VLM, conversation)."""

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
}


def build_label_dict(
    task_type: TaskType,
    label: dict[str, Any] | Mapping[int, Name | str] | str | Path | None,
) -> TaskLabelDict:
    """Build a task-specific label dictionary from Writer ``label``."""
    if not task_type.has_label_dict:
        return NoLabelDict()
    if label is None:
        raise AnnotationFormatError(f"{task_type.value} requires label")

    if task_type is TaskType.SEQUENCE:
        return SequenceLabelDict.from_writer_label(label)  # type: ignore[arg-type]

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
    """Load label vocabulary under ``task_dir``; return ``None`` when not applicable."""
    if not task_type.has_label_dict:
        return None
    if task_type is TaskType.SEQUENCE:
        vocab_path = vocab_path_for(task_dir)
        if not vocab_path.is_file():
            raise AnnotationFormatError(f"missing vocabulary file: {vocab_path}")
        return SequenceLabelDict.load(task_dir)
    meta_path = meta_path_for(task_dir, task_meta_filename)
    if not meta_path.is_file():
        raise AnnotationFormatError(f"missing label dictionary: {meta_path}")
    return _LABEL_DICT_BY_TASK[task_type].load(meta_path)  # type: ignore[attr-defined]


def _read_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AnnotationFormatError(f"{path}: label dictionary must be a JSON object")
    return data
