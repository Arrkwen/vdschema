"""
JSONL reader/writer for annotation records.

Paths are bound at construction time via ``task_dir``. Vocabulary files are
written or read automatically according to ``task_type``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .annotation_dict import (
    ANNOTATION_DATA_FILENAME,
    ANNOTATION_META_FILENAME,
    NoLabelDict,
    SequenceLabelDict,
    TaskLabelDict,
    build_label_dict,
    data_path_for,
    load_label_dict,
    meta_path_for,
    vocab_path_for,
)
from .annotation_format import AnnotationFormatError, BaseAnnotation, TaskType


DEFAULT_OUTPUT_DIR = "output"


def resolve_task_dir(
    task_type: TaskType,
    task_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Resolve output directory: default ``output`` → ``output/{task_type}``."""
    path = Path(task_dir)
    if path == Path(DEFAULT_OUTPUT_DIR):
        return path / task_type.value
    return path


class AnnotationWriter:
    """Write annotation records and optional label dictionary under ``task_dir``."""

    def __init__(
        self,
        task_type: TaskType,
        label: dict[str, Any] | dict[int, Any] | str | Path | None = None,
        task_dir: str | Path = DEFAULT_OUTPUT_DIR,
        *,
        task_data_filename: str = ANNOTATION_DATA_FILENAME,
        task_meta_filename: str = ANNOTATION_META_FILENAME,
    ) -> None:
        self.task_type = task_type
        self.task_dir = resolve_task_dir(task_type, task_dir)
        self.task_data_filename = task_data_filename
        self.task_meta_filename = task_meta_filename
        self.annotation_cls = task_type.annotation_class
        self.label_dict = build_label_dict(task_type, label)
        self.records: list[BaseAnnotation] = []

    def save_dir(self) -> Path:
        """Return the resolved output directory for this writer."""
        return self.task_dir

    @property
    def data_path(self) -> Path:
        """Path to the annotation JSONL file."""
        return data_path_for(self.task_dir, self.task_data_filename)

    @property
    def meta_path(self) -> Path:
        """Path to the label-meta (vocabulary) file."""
        return meta_path_for(self.task_dir, self.task_meta_filename)

    @property
    def vocab_path(self) -> Path:
        """Path to the sequence vocabulary file."""
        return vocab_path_for(self.task_dir)

    def append_annotation(self, annotation: BaseAnnotation) -> BaseAnnotation:
        if not isinstance(annotation, self.annotation_cls):
            raise AnnotationFormatError(
                f"Expected {self.annotation_cls.__name__}, got "
                f"{type(annotation).__name__}"
            )
        annotation.validate()
        self.label_dict.validate_annotation(annotation)
        self.records.append(annotation)
        return annotation

    def append(
        self,
        annotation: BaseAnnotation | None = None,
        **kwargs: Any,
    ) -> BaseAnnotation:
        if annotation is None:
            annotation = self.annotation_cls(**kwargs)
        return self.append_annotation(annotation)

    def save(self) -> None:
        """Write JSONL and label meta (when applicable) under ``task_dir``."""
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with self.data_path.open("w", encoding="utf-8") as f:
            for annotation in self.records:
                f.write(
                    json.dumps(
                        annotation.to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        if isinstance(self.label_dict, SequenceLabelDict):
            self.label_dict.save(self.task_dir)
        elif not isinstance(self.label_dict, NoLabelDict):
            self.label_dict.save(self.meta_path)


class AnnotationReader:
    """Read annotation JSONL and optional label dictionary from ``task_dir``."""

    def __init__(
        self,
        task_type: TaskType,
        task_dir: str | Path,
        *,
        task_data_filename: str = ANNOTATION_DATA_FILENAME,
        task_meta_filename: str = ANNOTATION_META_FILENAME,
    ) -> None:
        self.task_type = task_type
        self.task_dir = resolve_task_dir(task_type, task_dir)
        self.task_data_filename = task_data_filename
        self.task_meta_filename = task_meta_filename
        self.annotation_cls = task_type.annotation_class

    @property
    def data_path(self) -> Path:
        """Path to the annotation JSONL file."""
        return data_path_for(self.task_dir, self.task_data_filename)

    @property
    def meta_path(self) -> Path:
        """Path to the label-meta (vocabulary) file."""
        return meta_path_for(self.task_dir, self.task_meta_filename)

    @property
    def vocab_path(self) -> Path:
        """Path to the sequence vocabulary file."""
        return vocab_path_for(self.task_dir)

    def iter_raw(self) -> Iterator[dict[str, Any]]:
        path = self.data_path
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AnnotationFormatError(
                        f"{path}:{lineno} failed to parse JSON"
                    ) from exc

    def _resolve_label_dict(self) -> TaskLabelDict:
        loaded = load_label_dict(
            self.task_type,
            self.task_dir,
            task_meta_filename=self.task_meta_filename,
        )
        return loaded if loaded is not None else NoLabelDict()

    def _parse_annotations(
        self,
        label_dict: TaskLabelDict,
    ) -> list[BaseAnnotation]:
        annotations: list[BaseAnnotation] = []
        for record in self.iter_raw():
            annotation = self.annotation_cls.from_dict(record)
            annotation.validate()
            label_dict.validate_annotation(annotation)
            annotations.append(annotation)
        return annotations

    def iter_annotations(self) -> Iterator[BaseAnnotation]:
        yield from self._parse_annotations(self._resolve_label_dict())

    def load(self) -> tuple[list[BaseAnnotation], dict[str, Any] | None]:
        """Return parsed annotation ``data`` and ``label`` (``None`` if not used)."""
        loaded = load_label_dict(
            self.task_type,
            self.task_dir,
            task_meta_filename=self.task_meta_filename,
        )
        label_dict = loaded if loaded is not None else NoLabelDict()
        label = None if loaded is None else loaded.to_label()
        data = self._parse_annotations(label_dict)
        return data, label

    def validate(self) -> bool:
        """Return ``True`` if all records and label meta are valid, else ``False``."""
        try:
            self.load()
        except AnnotationFormatError:
            return False
        return True
