"""Tests for COCO source converters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdschema import AnnotationReader, Source, TaskType, switch
from vdswitch.cli import main
from vdswitch.converters.registry import get_converter_class
from vdswitch.help import _example_command, format_help_text
from vdswitch.utils.coco_dataset import (
    coco_bbox_xyxy,
    coco_segmentation_rle,
    load_category_map,
)


def test_coco_bbox_xyxy() -> None:
    assert coco_bbox_xyxy({"bbox": [10.0, 20.0, 30.0, 40.0]}) == [
        10.0,
        20.0,
        40.0,
        60.0,
    ]


def test_load_category_map_from_categories_only(tmp_path: Path) -> None:
    path = tmp_path / "categories.json"
    path.write_text(
        json.dumps({"categories": [{"id": 2, "name": "dog"}]}),
        encoding="utf-8",
    )
    label = load_category_map(path, task=TaskType.DETECTION)
    assert label[2].name == "dog"


def test_coco_segmentation_polygon_roundtrip() -> None:
    rle = coco_segmentation_rle(
        [[10, 10, 60, 10, 60, 50, 10, 50]],
        height=100,
        width=200,
    )
    assert rle is not None
    mask = rle.to_mask()
    assert mask.shape == (100, 200)


def test_vdswitch_coco_detection_without_input_label(
    coco_instances_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "vdschema_coco_det"
    switch(
        task=TaskType.DETECTION,
        source=Source.COCO,
        input=coco_instances_path,
        output=out,
    )
    assert (out / "instances.jsonl").is_file()


def test_vdswitch_coco_detection(coco_instances_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "vdschema_coco_det"
    switch(
        task=TaskType.DETECTION,
        source=Source.COCO,
        input=coco_instances_path,
        output=out,
    )
    data, label = AnnotationReader(
        TaskType.DETECTION,
        out,
        task_data_filename="instances.jsonl",
        task_meta_filename="label_dict.json",
    ).load()
    assert label
    assert len(data) == 1
    assert data[0].width == 200
    assert data[0].height == 100
    assert len(data[0].instances) == 2
    assert data[0].instances[0].bbox.to_list() == [10.0, 10.0, 60.0, 50.0]
    assert data[0].instances[1].is_ignored is True


def test_vdswitch_coco_detection_categories_in_json(
    coco_instances_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.COCO,
        input=coco_instances_path,
        output=out,
    )
    assert (out / "instances.jsonl").is_file()
    assert (out / "label_dict.json").is_file()


def test_vdswitch_coco_keypoint(
    coco_keypoint_instances_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "vdschema_coco_kp"
    switch(
        task=TaskType.KEYPOINT,
        source=Source.COCO,
        input=coco_keypoint_instances_path,
        output=out,
    )
    data, _ = AnnotationReader(
        TaskType.KEYPOINT,
        out,
        task_data_filename="instances_keypoints.jsonl",
        task_meta_filename="label_dict.json",
    ).load()
    assert len(data) == 1
    inst = data[0].instances[0]
    assert inst.keypoints is not None
    assert len(inst.keypoints) == 2
    assert inst.keypoints[0].to_list() == [20.0, 20.0, 2]


def test_vdswitch_coco_segmentation(coco_instances_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "vdschema_coco_seg"
    switch(
        task=TaskType.SEGMENTATION,
        source=Source.COCO,
        input=coco_instances_path,
        output=out,
    )
    data, _ = AnnotationReader(
        TaskType.SEGMENTATION,
        out,
        task_data_filename="instances.jsonl",
        task_meta_filename="label_dict.json",
    ).load()
    assert len(data) == 1
    assert len(data[0].instances) == 1
    assert data[0].instances[0].rle_mask is not None


@pytest.mark.parametrize(
    "task",
    [
        TaskType.CLASSIFICATION,
        TaskType.ACTION,
        TaskType.SEQUENCE,
    ],
)
def test_coco_unsupported_tasks(task: TaskType) -> None:
    with pytest.raises(ValueError, match="unsupported source='coco'"):
        get_converter_class(task, Source.COCO)


def test_vdswitch_cli_coco_detection(coco_instances_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli_coco"
    assert (
        main(
            [
                "--task",
                "detection",
                "--source",
                "coco",
                "--input",
                str(coco_instances_path),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert (out / "instances.jsonl").is_file()


def test_vdswitch_help_coco_detection() -> None:
    text = format_help_text(task=TaskType.DETECTION, source=Source.COCO)
    example = _example_command(TaskType.DETECTION, Source.COCO)
    assert "--source coco" in text
    assert "instances_train2017.json" in example
    assert "--input-label" not in example
    assert "--input-root" not in example
    assert "category_id_contiguous" in text
    assert "category=" not in text
    assert "root=" not in text
    assert "--option category_id_contiguous=0" in example
    assert "--option category_id_start=1" in example
    assert "prefix=VALUE" in text
    assert "--option prefix=split/v1" in example
    assert main(["help", "--task", "detection", "--source", "coco"]) == 0


def test_vdswitch_help_detection_defaults_to_up() -> None:
    text = format_help_text(task=TaskType.DETECTION, source=None)
    assert "train_baseline.jsonl" in text
    assert "--source monolith" in text
