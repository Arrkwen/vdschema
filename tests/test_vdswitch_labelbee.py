"""Tests for LabelBee source converters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vdschema import AnnotationReader, Source, TaskType, switch
from vdswitch.cli_help import format_help_text
from vdswitch.converters.labelbee import LabelBeeDetectionConverter
from vdswitch.converters.registry import get_converter_class, supported_sources
from vdswitch.utils.labelbee_dataset import (
    load_labelbee_document,
    point_list_to_flat,
    rect_to_bbox_xyxy,
    resolve_media_filename,
)


@pytest.fixture
def labelbee_detection_dir(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    images = root / "images"
    json_dir = root / "labelbee"
    images.mkdir(parents=True)
    json_dir.mkdir()
    Image.new("RGB", (640, 480), color="white").save(images / "sample.jpg")
    doc = {
        "width": 640,
        "height": 480,
        "valid": True,
        "rotate": 0,
        "step_1": {
            "toolName": "rectTool",
            "result": [
                {
                    "x": 10,
                    "y": 20,
                    "width": 90,
                    "height": 180,
                    "attribute": "person",
                    "valid": True,
                    "id": "Rp1x6bZs",
                    "sourceID": "",
                    "textAttribute": "",
                    "order": 1,
                }
            ],
        },
        "step_2": {
            "toolName": "lineTool",
            "result": [
                {
                    "pointList": [
                        {"x": 100, "y": 400, "id": "a"},
                        {"x": 200, "y": 350, "id": "b"},
                        {"x": 320, "y": 300, "id": "c"},
                    ],
                    "id": "line1",
                    "valid": True,
                    "order": 1,
                    "attribute": "lane_line",
                    "textAttribute": "",
                }
            ],
        },
    }
    (json_dir / "sample.json").write_text(
        json.dumps(doc, ensure_ascii=False), encoding="utf-8"
    )
    return root


def test_labelbee_detection_switch(
    labelbee_detection_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.LABELBEE,
        input_data=labelbee_detection_dir / "labelbee",
        output=out,
        input_root=labelbee_detection_dir,
    )
    data, label = AnnotationReader(
        TaskType.DETECTION,
        out,
        task_data_filename="labelbee.jsonl",
        task_meta_filename="annotation_meta.json",
    ).load()
    assert len(data) == 1
    assert data[0].filename == "images/sample.jpg"
    assert len(data[0].instances) == 2
    bbox_inst = data[0].instances[0]
    line_inst = data[0].instances[1]
    assert bbox_inst.bbox is not None
    assert bbox_inst.bbox.to_list() == pytest.approx([10, 20, 100, 200])
    assert line_inst.polyline == pytest.approx([100, 400, 200, 350, 320, 300])
    assert label[bbox_inst.category_id].name == "person"
    assert label[line_inst.category_id].name == "lane_line"


def test_labelbee_segmentation_polygon(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    json_dir = root / "json"
    json_dir.mkdir(parents=True)
    doc = {
        "width": 100,
        "height": 100,
        "step_1": {
            "toolName": "polygonTool",
            "result": [
                {
                    "pointList": [
                        {"x": 0, "y": 0},
                        {"x": 10, "y": 0},
                        {"x": 10, "y": 10},
                        {"x": 0, "y": 10},
                    ],
                    "attribute": "region",
                    "valid": True,
                    "id": "p1",
                    "sourceID": "",
                    "order": 1,
                    "textAttribute": "",
                }
            ],
        },
    }
    (json_dir / "a.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "out"
    switch(
        task=TaskType.SEGMENTATION,
        source=Source.LABELBEE,
        input_data=json_dir,
        output=out,
        input_root=root,
    )
    data, _ = AnnotationReader(
        TaskType.SEGMENTATION,
        out,
        task_data_filename="json.jsonl",
    ).load()
    assert data[0].instances[0].polygon == pytest.approx(
        [0, 0, 10, 0, 10, 10, 0, 10]
    )


def test_labelbee_keypoint(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    json_dir = root / "json"
    json_dir.mkdir(parents=True)
    doc = {
        "width": 64,
        "height": 64,
        "step_1": {
            "toolName": "pointTool",
            "result": [
                {
                    "x": 12,
                    "y": 34,
                    "valid": True,
                    "id": "pt1",
                    "sourceID": "",
                    "order": 1,
                    "attribute": "mark",
                    "textAttribute": "",
                }
            ],
        },
    }
    (json_dir / "pt.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "out"
    switch(
        task=TaskType.KEYPOINT,
        source=Source.LABELBEE,
        input_data=json_dir,
        output=out,
    )
    data, _ = AnnotationReader(
        TaskType.KEYPOINT, out, task_data_filename="json.jsonl"
    ).load()
    kp = data[0].instances[0].keypoints
    assert kp is not None
    assert kp[0].to_list() == pytest.approx([12, 34, 2])


def test_labelbee_classification_tag(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    json_dir = root / "json"
    json_dir.mkdir(parents=True)
    doc = {
        "width": 64,
        "height": 64,
        "step_1": {
            "toolName": "tagTool",
            "result": [
                {
                    "id": "tag1",
                    "sourceID": "",
                    "result": {"weather": "sunny;cloudy", "scene": "street"},
                }
            ],
        },
    }
    (json_dir / "tag.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "out"
    switch(
        task=TaskType.CLASSIFICATION,
        source=Source.LABELBEE,
        input_data=json_dir,
        output=out,
    )
    data, label = AnnotationReader(
        TaskType.CLASSIFICATION, out, task_data_filename="json.jsonl"
    ).load()
    assert label["weather"][1].name == "cloudy"
    cats = {c.category_attr: c.category_ids for c in data[0].categories}
    assert len(cats["weather"]) == 2


def test_labelbee_supported_tasks() -> None:
    assert Source.LABELBEE in supported_sources(TaskType.DETECTION)
    assert get_converter_class(TaskType.DETECTION, Source.LABELBEE)


def test_labelbee_help_text() -> None:
    text = format_help_text(task=TaskType.DETECTION, source=Source.LABELBEE)
    assert "labelbee" in text.lower()
    assert "step_N" in text or "step_" in text


def test_labelbee_dataset_helpers(tmp_path: Path) -> None:
    assert rect_to_bbox_xyxy({"x": 1, "y": 2, "width": 3, "height": 4}) == [
        1,
        2,
        4,
        6,
    ]
    assert point_list_to_flat([{"x": 0, "y": 1}, {"x": 2, "y": 3}]) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="pointList"):
        point_list_to_flat([{"x": 0, "y": 1}])

    bad = tmp_path / "bad.json"
    bad.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_labelbee_document(bad)


def test_labelbee_resolve_filename_from_document(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "x.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text("{}", encoding="utf-8")
    doc = {"file_name": "custom/dir/pic.png", "width": 1, "height": 1}
    assert (
        resolve_media_filename(
            json_path, doc, input_data=tmp_path / "nested", input_root=tmp_path
        )
        == "custom/dir/pic.png"
    )


def test_labelbee_detection_polygon_tool(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    doc = {
        "width": 50,
        "height": 50,
        "step_1": {
            "toolName": "polygonTool",
            "result": [
                {
                    "pointList": [
                        {"x": 0, "y": 0},
                        {"x": 5, "y": 0},
                        {"x": 5, "y": 5},
                    ],
                    "attribute": "box",
                    "valid": True,
                    "id": "p",
                    "sourceID": "",
                    "order": 1,
                    "textAttribute": "note",
                }
            ],
        },
    }
    (json_dir / "poly.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.LABELBEE,
        input_data=json_dir / "poly.json",
        output=out,
    )
    data, _ = AnnotationReader(
        TaskType.DETECTION, out, task_data_filename="poly.jsonl"
    ).load()
    assert data[0].instances[0].polygon is not None
    assert data[0].instances[0].text == "note"


def test_labelbee_empty_json_dir_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no LabelBee JSON"):
        switch(
            task=TaskType.DETECTION,
            source=Source.LABELBEE,
            input_data=empty,
            output=tmp_path / "out",
        )


def test_labelbee_invalid_input_data(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match=".json"):
        LabelBeeDetectionConverter(
            input_data=path,
            output_dir=tmp_path / "out",
        ).run()


def test_labelbee_classification_missing_tags(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    (json_dir / "a.json").write_text(
        json.dumps({"width": 1, "height": 1, "step_1": {"toolName": "rectTool"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no tagTool"):
        switch(
            task=TaskType.CLASSIFICATION,
            source=Source.LABELBEE,
            input_data=json_dir,
            output=tmp_path / "out",
        )
