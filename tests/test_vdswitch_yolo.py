"""Tests for YOLO source detection converter."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdschema import AnnotationReader, Source, TaskType, switch
from vdswitch.converters.registry import get_converter_class
from vdswitch.utils.yolo_dataset import (
    load_yolo_class_names,
    parse_yolo_label_file,
    yolo_label_path_for_image,
)


def test_yolo_label_path_mapping() -> None:
    image = Path("/data/images/train/a.jpg")
    assert yolo_label_path_for_image(image) == Path(
        "/data/labels/train/a.txt"
    )


def test_parse_yolo_label_file_xyxy(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    instances = parse_yolo_label_file(label, width=200, height=100)
    assert len(instances) == 1
    assert instances[0]["category_id"] == 0
    assert instances[0]["bbox"] == pytest.approx([60.0, 30.0, 140.0, 70.0])


def test_load_yolo_class_names(tmp_path: Path) -> None:
    path = tmp_path / "classes.txt"
    path.write_text("person\ncar\n", encoding="utf-8")
    names = load_yolo_class_names(path)
    assert names[0].name == "person"
    assert names[1].name == "car"


def test_vdswitch_yolo_detection(yolo_legacy_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.YOLO,
        input_data=yolo_legacy_dir / "train.txt",
        input_label=yolo_legacy_dir / "classes.txt",
        output=out,
        input_root=yolo_legacy_dir,
    )
    data, label = AnnotationReader(
        TaskType.DETECTION,
        out,
        task_data_filename="train.jsonl",
        task_meta_filename="label_dict.json",
    ).load()
    assert label
    assert len(data) == 1
    assert data[0].filename == "images/train/sample.jpg"
    assert len(data[0].instances) == 1
    assert data[0].instances[0].category_id == 0


def test_yolo_unsupported_for_classification() -> None:
    with pytest.raises(ValueError, match="unsupported source='yolo'"):
        get_converter_class(TaskType.CLASSIFICATION, Source.YOLO)
