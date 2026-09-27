"""Task-level instance geometry rules (bbox / polygon / rle_mask / keypoints)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdschema import (
    AnnotationFormatError,
    AnnotationReader,
    AnnotationWriter,
    TaskType,
)
from vdschema.annotation_format import Instance


def test_detection_requires_bbox_or_polygon(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.DETECTION,
        label={1: "x"},
        task_dir=tmp_path / "det",
    )
    with pytest.raises(AnnotationFormatError, match="bbox, polygon, or polyline"):
        writer.append(
            filename="a.jpg",
            width=10,
            height=10,
            instances=[{"id": 0, "category_id": 1}],
        )


def test_segmentation_requires_rle_or_polygon(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.SEGMENTATION,
        label={1: "x"},
        task_dir=tmp_path / "seg",
    )
    with pytest.raises(AnnotationFormatError, match="rle_mask or polygon"):
        writer.append(
            filename="a.jpg",
            width=10,
            height=10,
            instances=[{"id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        )


def test_keypoint_requires_keypoints(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.KEYPOINT,
        label={1: "person"},
        task_dir=tmp_path / "kp",
    )
    with pytest.raises(AnnotationFormatError, match="keypoints"):
        writer.append(
            filename="a.jpg",
            width=10,
            height=10,
            instances=[{"id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        )


def test_relationship_polygon_only(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.RELATIONSHIP,
        label={"detection": {1: "a"}, "relationship": {0: "near"}},
        task_dir=tmp_path / "rel",
    )
    writer.append(
        filename="a.jpg",
        width=10,
        height=10,
        instances=[
            {"id": 0, "category_id": 1, "polygon": [0, 0, 5, 0, 5, 5, 0, 5]},
            {"id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]},
        ],
        relationships=[{"subject_id": 0, "object_id": 1, "relation_type": "near"}],
    )
    writer.save()
    data, _ = AnnotationReader(TaskType.RELATIONSHIP, writer.save_dir()).load()
    assert data[0].instances[0].polygon is not None


def test_relationship_requires_bbox_or_polygon(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.RELATIONSHIP,
        label={"detection": {1: "a"}, "relationship": {0: "near"}},
        task_dir=tmp_path / "rel",
    )
    with pytest.raises(AnnotationFormatError, match="bbox or polygon"):
        writer.append(
            filename="a.jpg",
            width=10,
            height=10,
            instances=[{"id": 0, "category_id": 1}],
            relationships=[{"subject_id": 0, "object_id": 0, "relation_type": "near"}],
        )


def test_detection_bbox_and_polygon_together(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.DETECTION,
        label={1: "x"},
        task_dir=tmp_path / "det",
    )
    writer.append(
        filename="a.jpg",
        width=10,
        height=10,
        instances=[
            {
                "id": 0,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "polygon": [0, 0, 10, 0, 10, 10],
            }
        ],
    )
    writer.save()
    data, _ = AnnotationReader(TaskType.DETECTION, writer.save_dir()).load()
    assert data[0].instances[0].bbox is not None
    assert data[0].instances[0].polygon is not None


def test_polygon_invalid_coordinates() -> None:
    with pytest.raises(AnnotationFormatError, match="polygon"):
        Instance.from_dict({"id": 0, "category_id": 1, "polygon": [0, 0, 1]})


def test_detection_polyline_only(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.DETECTION,
        label={1: "lane"},
        task_dir=tmp_path / "det_line",
    )
    writer.append(
        filename="road.jpg",
        width=640,
        height=480,
        instances=[
            {
                "id": 0,
                "category_id": 1,
                "polyline": [100, 400, 200, 350, 320, 300],
            }
        ],
    )
    writer.save()
    data, _ = AnnotationReader(TaskType.DETECTION, writer.save_dir()).load()
    assert data[0].instances[0].polyline == [100.0, 400.0, 200.0, 350.0, 320.0, 300.0]


def test_polyline_invalid_coordinates() -> None:
    with pytest.raises(AnnotationFormatError, match="polyline"):
        Instance.from_dict({"id": 0, "category_id": 1, "polyline": [0, 0, 1]})


def test_jsonl_legacy_segmentation_key(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.DETECTION,
        label={1: "x"},
        task_dir=tmp_path / "det",
    )
    writer.append(
        filename="a.jpg",
        width=10,
        height=10,
        instances=[{"id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]}],
    )
    writer.save()
    line = (
        '{"filename":"b.jpg","width":10,"height":10,'
        '"instances":[{"id":0,"category_id":1,"bbox":[0,0,1,1],'
        '"segmentation":{"size":[10,10],"counts":"6320004"}}]}'
    )
    writer.data_path.write_text(line + "\n", encoding="utf-8")
    data, _ = AnnotationReader(TaskType.DETECTION, writer.save_dir()).load()
    assert data[0].instances[0].rle_mask is not None


def test_action_track_legacy_segmentation_in_jsonl(tmp_path: Path) -> None:
    writer = AnnotationWriter(
        TaskType.ACTION,
        label={1: "walk"},
        task_dir=tmp_path / "act",
    )
    writer.append(
        filename="v.mp4",
        width=100,
        height=100,
        actions=[
            {
                "category_id": 1,
                "track_id": 0,
                "start_idx": 0,
                "end_idx": 0,
                "tracks": [
                    {
                        "frame_idx": 0,
                        "bbox": [0, 0, 10, 10],
                        "segmentation": {"size": [100, 100], "counts": "6320004"},
                    }
                ],
            }
        ],
    )
    writer.save()
    rows = [
        __import__("json").loads(line)
        for line in writer.data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "rle_mask" in rows[0]["actions"][0]["tracks"][0]
