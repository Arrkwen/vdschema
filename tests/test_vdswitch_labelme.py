"""Tests for LabelMe source converters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vdschema import AnnotationReader, Source, TaskType, switch
from vdswitch.cli_help import format_help_text
from vdswitch.converters.labelme import LabelMeDetectionConverter
from vdswitch.converters.registry import get_converter_class, supported_sources
from vdswitch.utils.labelme_dataset import (
    DETECTION_SHAPE_TYPES,
    circle_to_bbox_xyxy,
    collect_label_ids,
    detection_instances_from_document,
    document_size,
    iter_labelme_json_paths,
    keypoint_instances_from_document,
    load_labelme_document,
    points_to_flat,
    rectangle_to_bbox_xyxy,
    resolve_media_filename,
    segmentation_instances_from_document,
)


@pytest.fixture
def labelme_detection_dir(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    images = root / "images"
    json_dir = root / "labelme"
    images.mkdir(parents=True)
    json_dir.mkdir()
    Image.new("RGB", (640, 480), color="white").save(images / "sample.jpg")
    doc = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": "cat",
                "points": [[10, 20], [100, 200]],
                "shape_type": "rectangle",
                "flags": {},
            },
            {
                "label": "lane",
                "points": [[0, 400], [100, 300], [200, 250]],
                "shape_type": "linestrip",
                "flags": {},
            },
            {
                "label": "pin",
                "points": [[50, 60]],
                "shape_type": "point",
                "flags": {},
            },
            {
                "label": "ball",
                "points": [[300, 200], [320, 200]],
                "shape_type": "circle",
                "flags": {},
            },
        ],
        "imagePath": "images/sample.jpg",
        "imageHeight": 480,
        "imageWidth": 640,
    }
    (json_dir / "sample.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


def test_labelme_geometry_helpers() -> None:
    assert rectangle_to_bbox_xyxy([[10, 20], [100, 200]]) == [10, 20, 100, 200]
    assert points_to_flat([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    bbox = circle_to_bbox_xyxy([[10, 10], [20, 10]])
    assert bbox[0] == pytest.approx(0)
    assert bbox[2] == pytest.approx(20)


def test_labelme_detection_switch(labelme_detection_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.LABELME,
        input_data=labelme_detection_dir / "labelme",
        output=out,
        input_root=labelme_detection_dir,
    )
    data, label = AnnotationReader(
        TaskType.DETECTION,
        out,
        task_data_filename="labelme.jsonl",
    ).load()
    assert len(data) == 1
    assert data[0].filename == "images/sample.jpg"
    assert len(data[0].instances) == 4
    assert data[0].instances[0].bbox is not None
    assert data[0].instances[1].polyline is not None
    assert data[0].instances[2].point == [50.0, 60.0]
    assert label[data[0].instances[0].category_id].name == "cat"


def test_labelme_segmentation(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    doc = {
        "imagePath": "a.jpg",
        "imageWidth": 10,
        "imageHeight": 10,
        "shapes": [
            {
                "label": "region",
                "shape_type": "polygon",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        ],
    }
    (json_dir / "a.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "out"
    switch(
        task=TaskType.SEGMENTATION,
        source=Source.LABELME,
        input_data=json_dir,
        output=out,
    )
    data, _ = AnnotationReader(
        TaskType.SEGMENTATION, out, task_data_filename="json.jsonl"
    ).load()
    assert data[0].instances[0].polygon is not None


def test_labelme_keypoint(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    doc = {
        "imagePath": "p.jpg",
        "imageWidth": 64,
        "imageHeight": 64,
        "shapes": [
            {"label": "mark", "shape_type": "point", "points": [[12, 34]]},
        ],
    }
    (json_dir / "p.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "out"
    switch(
        task=TaskType.KEYPOINT,
        source=Source.LABELME,
        input_data=json_dir,
        output=out,
    )
    data, _ = AnnotationReader(
        TaskType.KEYPOINT, out, task_data_filename="json.jsonl"
    ).load()
    assert data[0].instances[0].keypoints is not None


def test_labelme_load_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_labelme_document(bad)

    no_shapes = tmp_path / "noshapes.json"
    no_shapes.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="shapes"):
        load_labelme_document(no_shapes)


def test_labelme_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no LabelMe JSON"):
        switch(
            task=TaskType.DETECTION,
            source=Source.LABELME,
            input_data=empty,
            output=tmp_path / "out",
        )


def test_labelme_invalid_input_file(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match=".json"):
        LabelMeDetectionConverter(
            input_data=path,
            output_dir=tmp_path / "out",
        ).run()


def test_labelme_registry_and_help() -> None:
    assert Source.LABELME in supported_sources(TaskType.DETECTION)
    assert get_converter_class(TaskType.SEGMENTATION, Source.LABELME)
    text = format_help_text(task=TaskType.DETECTION, source=Source.LABELME)
    assert "labelme" in text.lower()


def test_labelme_document_size_and_resolve(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="imageWidth"):
        document_size({})

    json_path = tmp_path / "a.json"
    json_path.write_text("{}", encoding="utf-8")
    name = resolve_media_filename(
        json_path,
        {"imagePath": "pics/a.jpg"},
        input_data=tmp_path,
        input_root=tmp_path,
    )
    assert name == "pics/a.jpg"


def test_labelme_detection_instances_skip_invalid() -> None:
    doc = {
        "shapes": [
            {
                "label": "bad_line",
                "shape_type": "linestrip",
                "points": [[0, 0]],
            },
            {
                "label": "ok",
                "shape_type": "point",
                "points": [[1, 2]],
            },
        ]
    }
    label_to_id = {"bad_line": 1, "ok": 2}
    instances = detection_instances_from_document(doc, label_to_id=label_to_id)
    assert len(instances) == 1
    assert instances[0]["point"] == [1.0, 2.0]


def test_labelme_collect_labels(tmp_path: Path) -> None:
    path = tmp_path / "one.json"
    path.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "a",
                        "shape_type": "rectangle",
                        "points": [[0, 0], [1, 1]],
                    }
                ],
                "imageWidth": 1,
                "imageHeight": 1,
            }
        ),
        encoding="utf-8",
    )
    ids = collect_label_ids([path], DETECTION_SHAPE_TYPES)
    assert ids["a"] == 1
    assert iter_labelme_json_paths(path) == [path]


def test_labelme_resolve_media_from_disk(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 10)).save(images / "pic.jpg")
    json_dir = tmp_path / "ann"
    json_dir.mkdir()
    json_path = json_dir / "pic.json"
    json_path.write_text(
        json.dumps({"shapes": [], "imageWidth": 10, "imageHeight": 10}),
        encoding="utf-8",
    )
    name = resolve_media_filename(
        json_path,
        load_labelme_document(json_path),
        input_data=json_dir,
        input_root=tmp_path,
    )
    assert name == "images/pic.jpg"


def test_labelme_points_and_circle_errors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        points_to_flat([])
    with pytest.raises(ValueError, match="two points"):
        circle_to_bbox_xyxy([[0, 0]])


def test_labelme_detection_polygon_and_description() -> None:
    doc = {
        "shapes": [
            {
                "label": "obj",
                "shape_type": "polygon",
                "points": [[0, 0], [1, 0], [1, 1]],
                "description": "note",
            }
        ]
    }
    instances = detection_instances_from_document(doc, label_to_id={"obj": 1})
    assert instances[0]["polygon"]
    assert instances[0]["text"] == "note"


def test_labelme_resolve_absolute_and_fallback(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (2, 2)).save(outside)
    name = resolve_media_filename(
        root / "x.json",
        {"imagePath": str(outside.resolve())},
        input_data=root,
        input_root=root,
    )
    assert name == "outside.jpg"

    solo = tmp_path / "solo.json"
    solo.write_text("{}", encoding="utf-8")
    assert (
        resolve_media_filename(
            solo,
            {"shapes": []},
            input_data=solo,
            input_root=None,
        )
        == "solo.jpg"
    )


def test_labelme_segmentation_and_keypoint_helpers() -> None:
    label_to_id = {"r": 1, "p": 2}
    seg = segmentation_instances_from_document(
        {
            "shapes": [
                {"label": "r", "shape_type": "polygon", "points": [[0, 0], [1, 0]]},
                {
                    "label": "r",
                    "shape_type": "polygon",
                    "points": [[0, 0], [2, 0], [2, 2], [0, 2]],
                    "description": "x",
                },
            ]
        },
        label_to_id=label_to_id,
    )
    assert len(seg) == 1
    assert seg[0]["text"] == "x"

    kp = keypoint_instances_from_document(
        {
            "shapes": [
                {"label": "p", "shape_type": "point", "points": "bad"},
                {
                    "label": "p",
                    "shape_type": "point",
                    "points": [[3, 4]],
                    "description": "m",
                },
            ]
        },
        label_to_id=label_to_id,
    )
    assert len(kp) == 1
    assert kp[0]["text"] == "m"


def test_labelme_collect_empty_labels(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps({"shapes": [], "imageWidth": 1, "imageHeight": 1}),
        encoding="utf-8",
    )
    assert collect_label_ids([path], DETECTION_SHAPE_TYPES)[""] == 1


def test_labelme_skip_unknown_shape() -> None:
    doc = {"shapes": [{"label": "x", "shape_type": "unknown", "points": []}]}
    assert detection_instances_from_document(doc, label_to_id={"x": 1}) == []


def test_labelme_detection_invalid_shape_points() -> None:
    doc = {
        "shapes": [
            {"label": "a", "shape_type": "rectangle", "points": "invalid"},
        ]
    }
    assert detection_instances_from_document(doc, label_to_id={"a": 1}) == []


def test_labelme_single_json_file(tmp_path: Path) -> None:
    json_path = tmp_path / "one.json"
    json_path.write_text(
        json.dumps(
            {
                "imagePath": "i.jpg",
                "imageWidth": 10,
                "imageHeight": 10,
                "shapes": [
                    {
                        "label": "o",
                        "shape_type": "point",
                        "points": [[1, 1]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.LABELME,
        input_data=json_path,
        output=out,
    )
    data, _ = AnnotationReader(
        TaskType.DETECTION, out, task_data_filename="one.jsonl"
    ).load()
    assert len(data) == 1
