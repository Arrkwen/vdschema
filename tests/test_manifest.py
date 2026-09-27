"""Manifest task: media index with filename, width, height in annotation_data.jsonl."""

from __future__ import annotations

from pathlib import Path

from vdschema import (
    DetectionAnnotation,
    ManifestAnnotation,
    TaskType,
    VlmAnnotation,
)
from vdschema.annotation_dict import ANNOTATION_DATA_FILENAME
from vdschema.annotation_io import AnnotationReader, AnnotationWriter


def test_manifest_writer_round_trip(tmp_path: Path) -> None:
    writer = AnnotationWriter(TaskType.MANIFEST, task_dir=tmp_path / "manifest")
    writer.append(filename="images/a.jpg", width=640, height=480)
    writer.append(filename="videos/b.mp4", width=1920, height=1080)
    writer.save()

    assert writer.data_path.name == ANNOTATION_DATA_FILENAME
    assert not writer.meta_path.is_file()

    reader = AnnotationReader(TaskType.MANIFEST, writer.save_dir())
    data, label = reader.load()
    assert label is None
    assert len(data) == 2
    assert isinstance(data[0], ManifestAnnotation)
    assert data[0].to_dict() == {
        "filename": "images/a.jpg",
        "width": 640,
        "height": 480,
    }
    assert data[1].filename == "videos/b.mp4"


def test_read_manifest_dir_as_detection_or_vlm(tmp_path: Path) -> None:
    writer = AnnotationWriter(TaskType.MANIFEST, task_dir=tmp_path / "manifest")
    writer.append(filename="images/a.jpg", width=10, height=10)
    writer.save()
    manifest_dir = writer.save_dir()

    data_det, _ = AnnotationReader(TaskType.DETECTION, manifest_dir).load()
    assert isinstance(data_det[0], DetectionAnnotation)
    assert data_det[0].instances == []

    data_vlm, _ = AnnotationReader(TaskType.VLM, manifest_dir).load()
    assert isinstance(data_vlm[0], VlmAnnotation)
    assert data_vlm[0].description == ""
