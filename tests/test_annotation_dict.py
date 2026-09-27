"""Label dictionary loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdschema import AnnotationFormatError, AnnotationWriter, Name, TaskType
from vdschema.annotation_dict import (
    build_label_dict,
    load_label_dict,
    meta_path_for,
)


def test_build_label_dict_detection() -> None:
    ld = build_label_dict(TaskType.DETECTION, {1: Name("cat")})
    assert ld.to_label()[1].name == "cat"


def test_load_label_dict_missing_meta(tmp_path: Path) -> None:
    task_dir = tmp_path / "det"
    task_dir.mkdir()
    with pytest.raises(AnnotationFormatError, match="missing label dictionary"):
        load_label_dict(TaskType.DETECTION, task_dir)


def test_writer_invalid_category_on_validate(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.DETECTION,
        label={1: "cat"},
        task_dir=tmp_path / "det",
    )
    with pytest.raises(AnnotationFormatError, match="unknown category_id"):
        writer.append(
            filename="a.jpg",
            width=10,
            height=10,
            instances=[{"id": 0, "category_id": 99, "bbox": [0, 0, 1, 1]}],
        )


def test_relationship_label_dict(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.RELATIONSHIP,
        label={"detection": {1: "a"}, "relationship": {0: "near"}},
        task_dir=tmp_path / "rel",
    )
    writer.append(
        filename="a.jpg",
        width=10,
        height=10,
        instances=[{"id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        relationships=[{"subject_id": 0, "object_id": 0, "relation_type": "near"}],
    )
    writer.save()
    raw = json.loads(meta_path_for(writer.save_dir()).read_text(encoding="utf-8"))
    assert "relationship" in raw


def test_sequence_label_dict_errors(tmp_path: Path) -> None:
    from vdschema.annotation_dict import SequenceLabelDict

    with pytest.raises(AnnotationFormatError, match="vocabulary file path"):
        SequenceLabelDict.from_writer_label({"bad": "x"})

    with pytest.raises(AnnotationFormatError, match="not found"):
        SequenceLabelDict.from_writer_label(str(tmp_path / "missing.txt"))
