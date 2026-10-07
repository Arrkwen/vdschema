"""Tests for vdswitch --option parsing and category id densify."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdschema import AnnotationReader, Name, Source, TaskType, switch
from vdswitch.cli import main
from vdswitch.converters.registry import get_converter_class
from vdswitch.options import (
    ConverterOptionSpec,
    ConvertOptions,
    densify_label_map,
    example_option_command_lines,
    map_category_id,
    parse_option_pairs,
    prepare_category_label,
    resolve_converter_inputs,
)
from vdswitch.options.presets import GLOBAL_ONLY_OPTIONS


def test_parse_option_pairs_unknown_key_allowed_at_parse() -> None:
    assert parse_option_pairs(["foo=1"]) == {"foo": "1"}


def test_parse_option_pairs_bad_format() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_option_pairs(["noseparator"])


def test_parse_option_pairs_empty_key() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_option_pairs(["=value"])


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


def test_prepare_category_label_densifies() -> None:
    label = {3: Name("a")}
    out, id_map = prepare_category_label(
        label, ConvertOptions(category_id_contiguous=True, category_id_start=0)
    )
    assert id_map == {3: 0}
    assert out[0].name == "a"


def test_convert_options_from_mapping_bool_and_start() -> None:
    opts = ConvertOptions.from_mapping(
        {
            "category_id_contiguous": "false",
            "category_id_start": "0",
        }
    )
    assert opts.category_id_contiguous is False
    assert opts.category_id_start == 0


def test_convert_options_from_mapping_true_aliases() -> None:
    assert ConvertOptions.from_mapping(
        {"category_id_contiguous": "yes"}
    ).category_id_contiguous
    assert ConvertOptions.from_mapping(
        {"category_id_contiguous": "on"}
    ).category_id_contiguous


def test_convert_options_invalid_bool() -> None:
    with pytest.raises(ValueError, match="category_id_contiguous"):
        ConvertOptions.from_mapping({"category_id_contiguous": "maybe"})


def test_convert_options_invalid_start() -> None:
    with pytest.raises(ValueError, match="category_id_start"):
        ConvertOptions.from_mapping({"category_id_start": "2"})


def test_densify_label_map_invalid_start() -> None:
    with pytest.raises(ValueError, match="category_id_start"):
        densify_label_map({1: Name("a")}, start=2)


def test_densify_label_map_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        densify_label_map({}, start=1)


def test_map_category_id_passthrough_and_unknown() -> None:
    assert map_category_id(7, None) == 7
    assert map_category_id(1, {1: 10}) == 10
    with pytest.raises(ValueError, match="category_id=99"):
        map_category_id(99, {1: 1})


def test_resolve_converter_inputs_missing_required_category() -> None:
    cls = get_converter_class(TaskType.DETECTION, Source.MONOLITH)
    with pytest.raises(ValueError, match="missing required --option category"):
        resolve_converter_inputs(
            cls,
            input=Path("/data/train.jsonl"),
            options={"root": "/data/root"},
        )


def test_resolve_converter_inputs_missing_required_root() -> None:
    cls = get_converter_class(TaskType.DETECTION, Source.MONOLITH)
    with pytest.raises(ValueError, match="missing required --option root"):
        resolve_converter_inputs(
            cls,
            input=Path("/data/train.jsonl"),
            options={"category": "/data/label.json"},
        )


def test_resolve_converter_inputs_optional_category_defaults_to_input() -> None:
    cls = get_converter_class(TaskType.DETECTION, Source.VOC)
    resolved = resolve_converter_inputs(
        cls,
        input=Path("/data/VOC2007"),
        options={"root": "/data/VOC2007"},
    )
    assert resolved.category == Path("/data/VOC2007").resolve()


def test_resolve_converter_inputs_prefix_option() -> None:
    cls = get_converter_class(TaskType.DETECTION, Source.COCO)
    resolved = resolve_converter_inputs(
        cls,
        input=Path("/data/instances.json"),
        options={"prefix": "split/v1"},
    )
    assert resolved.prefix == "split/v1"


def test_resolve_converter_inputs_empty_prefix_rejected() -> None:
    cls = get_converter_class(TaskType.CLASSIFICATION, Source.IMAGENET)
    with pytest.raises(ValueError, match="prefix"):
        resolve_converter_inputs(
            cls,
            input=Path("/data/train"),
            options={"prefix": "  "},
        )


def test_example_option_command_lines_skips_and_fills() -> None:
    class _Skips:
        converter_options = (
            ConverterOptionSpec("category", "help", required=False),
            ConverterOptionSpec("root", "help", required=False),
            ConverterOptionSpec("custom", "help", required=True),
            ConverterOptionSpec("optional_empty", "help", required=False),
        ) + GLOBAL_ONLY_OPTIONS

    skip_lines = example_option_command_lines(
        _Skips,
        input_path="/in/data",
        category_path=None,
    )
    assert "  --option root=/path/to/dataset \\" in skip_lines
    assert "  --option prefix=split/v1 \\" in skip_lines
    assert not any("custom=" in line for line in skip_lines)

    class _RootRequired:
        converter_options = (
            ConverterOptionSpec("root", "help", required=True),
        ) + GLOBAL_ONLY_OPTIONS

    root_lines = example_option_command_lines(
        _RootRequired,
        input_path="/in/data",
        category_path=None,
    )
    assert "  --option root=/path/to/dataset \\" in root_lines
    assert "  --option prefix=split/v1 \\" in root_lines

    coco = get_converter_class(TaskType.DETECTION, Source.COCO)
    coco_lines = example_option_command_lines(
        coco,
        input_path="/ann/instances.json",
        category_path=None,
    )
    assert any("category_id_contiguous=0" in line for line in coco_lines)
    assert any("category_id_start=" in line for line in coco_lines)
    assert any("prefix=split/v1" in line for line in coco_lines)
    assert not any("category=" in line for line in coco_lines)


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
