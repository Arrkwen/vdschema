"""Negative samples and missing task keys: empty lists and omitted fields."""

from __future__ import annotations

import pytest

from vdschema import (
    ActionAnnotation,
    AnnotationFormatError,
    ClassificationAnnotation,
    ConversationAnnotation,
    DetectionAnnotation,
    RelationshipAnnotation,
    SequenceAnnotation,
    TaskType,
    VlmAnnotation,
)
from vdschema.annotation_io import AnnotationReader, AnnotationWriter

_BASE = {"filename": "images/neg.jpg", "width": 640, "height": 480}


@pytest.mark.parametrize(
    "cls",
    [
        DetectionAnnotation,
        ClassificationAnnotation,
        RelationshipAnnotation,
        SequenceAnnotation,
        ConversationAnnotation,
        ActionAnnotation,
    ],
)
def test_from_dict_missing_task_key_defaults_empty(cls):
    ann = cls.from_dict(dict(_BASE))
    ann.validate()
    assert ann.to_dict()["filename"] == _BASE["filename"]


def test_vlm_empty_description():
    ann = VlmAnnotation.from_dict({**_BASE, "description": ""})
    ann.validate()
    assert "description" not in ann.to_dict()


def test_vlm_missing_description():
    ann = VlmAnnotation.from_dict(dict(_BASE))
    ann.validate()
    assert ann.description == ""


def test_relationship_instances_without_relationships():
    ann = RelationshipAnnotation.from_dict(
        {
            **_BASE,
            "instances": [{"id": 0, "category_id": 1, "bbox": [1, 2, 3, 4]}],
            "relationships": [],
        }
    )
    ann.validate()


def test_relationship_relationships_without_instances_rejected():
    ann = RelationshipAnnotation.from_dict(
        {
            **_BASE,
            "instances": [],
            "relationships": [
                {"subject_id": 0, "object_id": 1, "relation_type": "near"}
            ],
        }
    )
    with pytest.raises(AnnotationFormatError, match="instances is empty"):
        ann.validate()


def test_detection_writer_round_trip_base_only():
    writer = AnnotationWriter(
        TaskType.DETECTION,
        label={1: "person"},
    )
    writer.append(filename=_BASE["filename"], width=640, height=480)
    writer.save()
    reader = AnnotationReader(TaskType.DETECTION, writer.save_dir())
    data, _ = reader.load()
    assert len(data) == 1
    assert data[0].instances == []
