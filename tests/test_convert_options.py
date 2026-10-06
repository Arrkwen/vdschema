"""Tests for vdswitch --option parsing and category id densify."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdschema import AnnotationReader, Name, Source, TaskType, switch
from vdswitch.cli import main
from vdswitch.converters.registry import get_converter_class
from vdswitch.options import (
    ConvertOptions,
    densify_label_map,
    parse_option_pairs,
    prepare_category_label,
    resolve_converter_inputs,
)


def test_parse_option_pairs_unknown_key_allowed_at_parse() -> None:
    assert parse_option_pairs(["foo=1"]) == {"foo": "1"}


def test_parse_option_pairs_bad_format() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_option_pairs(["noseparator"])


def test_parse_option_pairs_empty_value() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_option_pairs(["category="])


def test_resolve_converter_inputs_monolith() -> None:
    cls = get_converter_class(TaskType.DETECTION, Source.MONOLITH)
    resolved = resolve_converter_inputs(
        cls,
        input=Path("/data/train.jsonl"),
        options={
            "category": "/data/label.json",
            "root": "/data/root",
        },
    )
    assert resolved.category == Path("/data/label.json").resolve()
    assert resolved.root == Path("/data/root").resolve()


def test_resolve_converter_inputs_rejects_unknown_key() -> None:
    cls = get_converter_class(TaskType.DETECTION, Source.COCO)
    with pytest.raises(ValueError, match="unknown option"):
        resolve_converter_inputs(
            cls,
            input=Path("/data/instances.json"),
            options={"category": "/x"},
        )


def test_densify_label_map_start_one() -> None:
    label = {1: Name("a"), 5: Name("b"), 90: Name("c")}
    new_label, id_map = densify_label_map(label, start=1)
    assert id_map == {1: 1, 5: 2, 90: 3}
    assert set(new_label.keys()) == {1, 2, 3}
    assert new_label[2].name == "b"


def test_densify_label_map_start_zero() -> None:
    label = {10: Name("x"), 20: Name("y")}
    new_label, id_map = densify_label_map(label, start=0)
    assert id_map == {10: 0, 20: 1}
    assert new_label[0].name == "x"


def test_prepare_category_label_noop() -> None:
    label = {5: Name("cat")}
    out, id_map = prepare_category_label(label, ConvertOptions())
    assert out is label
    assert id_map is None


def test_vdswitch_coco_contiguous_category_ids(tmp_path: Path) -> None:
    root = tmp_path / "coco"
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True)
    payload = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 5, "name": "dog"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 1, 1],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 5,
                "bbox": [1, 1, 2, 2],
            },
        ],
    }
    coco_json = ann_dir / "instances.json"
    coco_json.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.COCO,
        input=coco_json,
        output=out,
        options={
            "category_id_contiguous": "1",
            "category_id_start": "1",
        },
    )
    data, label = AnnotationReader(
        TaskType.DETECTION,
        out,
        task_data_filename="instances.jsonl",
        task_meta_filename="label_dict.json",
    ).load()
    assert set(label.keys()) == {1, 2}
    assert label[1].name == "cat"
    assert label[2].name == "dog"
    cids = sorted(inst.category_id for inst in data[0].instances)
    assert cids == [1, 2]


def test_cli_unknown_option(capsys, tmp_path: Path) -> None:
    code = main(
        [
            "--task",
            "detection",
            "--source",
            "coco",
            "--input",
            str(tmp_path / "missing.json"),
            "--option",
            "bad=1",
        ]
    )
    assert code == 1
    assert "unknown option" in capsys.readouterr().err
