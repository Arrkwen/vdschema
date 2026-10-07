"""Help text and dataset utility coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdschema import TaskType
from vdswitch.converters.registry import supported_sources, supported_tasks
from vdswitch.converters.sources import Source
from vdswitch.help import format_help_text
from vdswitch.utils.coco_dataset import (
    coco_keypoints_flat,
    coco_segmentation_rle,
    load_category_map,
    load_coco_json,
)
from vdswitch.utils.jsonl import load_jsonl_object
from vdswitch.utils.ocr_dataset import iter_ocr_lines, tokenize_label
from vdswitch.utils.yolo_dataset import (
    iter_image_list,
    load_yolo_class_names,
    parse_yolo_label_file,
    resolve_image_path,
    yolo_label_path_for_image,
)


def test_format_help_all_tasks() -> None:
    text = format_help_text(task=None, source=None)
    assert "detection" in text
    assert "classification" in text


def test_format_help_prefix_documented_for_every_converter() -> None:
    for task in supported_tasks():
        for source in supported_sources(task):
            text = format_help_text(task=task, source=source)
            assert "prefix=VALUE" in text
            assert "Required\n" in text
            assert "Optional --option" in text
            assert "--option prefix=" in text


def test_format_help_yolo_detection() -> None:
    text = format_help_text(task=TaskType.DETECTION, source=Source.YOLO)
    assert "train.txt" in text
    assert "prefix=VALUE" in text
    assert "--option prefix=" in text


def test_format_help_imagenet() -> None:
    text = format_help_text(task=TaskType.CLASSIFICATION, source=Source.IMAGENET)
    assert "/path/to/train" in text


def test_format_help_ocr_sequence() -> None:
    text = format_help_text(task=TaskType.SEQUENCE, source=Source.OCR)
    assert "anno.txt" in text


def test_format_help_action_monolith() -> None:
    text = format_help_text(task=TaskType.ACTION, source=Source.MONOLITH)
    assert "video_train.txt" in text


def test_format_help_bad_source_for_task() -> None:
    with pytest.raises(ValueError, match="unsupported source"):
        format_help_text(task=TaskType.DETECTION, source=Source.IMAGENET)


def test_load_jsonl_object_errors(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    with pytest.raises(Exception, match="failed to parse JSON"):
        load_jsonl_object("{", path=path, lineno=1)
    with pytest.raises(Exception, match="must be an object"):
        load_jsonl_object("[]", path=path, lineno=2)


def test_ocr_manifest_tab_and_errors(tmp_path: Path) -> None:
    manifest = tmp_path / "anno.txt"
    manifest.write_text("img/a.jpg\tABC\n", encoding="utf-8")
    img = tmp_path / "img" / "a.jpg"
    img.parent.mkdir()
    img.write_bytes(b"x")
    rows = iter_ocr_lines(manifest, root=tmp_path)
    assert tokenize_label("AB") == ["A", "B"]
    assert rows[0][2] == ["A", "B", "C"]

    bad = tmp_path / "bad.txt"
    bad.write_text("onlypath\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest line"):
        iter_ocr_lines(bad, root=tmp_path)

    empty_file = tmp_path / "none.txt"
    empty_file.write_text("# comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no OCR lines"):
        iter_ocr_lines(empty_file, root=tmp_path)


def test_yolo_utils(tmp_path: Path) -> None:
    classes = tmp_path / "classes.txt"
    classes.write_text("cat\ndog\n", encoding="utf-8")
    names = load_yolo_class_names(classes)
    assert names[0].name == "cat"

    empty_classes = tmp_path / "empty.txt"
    empty_classes.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty YOLO"):
        load_yolo_class_names(empty_classes)

    list_file = tmp_path / "train.txt"
    list_file.write_text("images/a.jpg\n", encoding="utf-8")
    img = tmp_path / "images" / "a.jpg"
    img.parent.mkdir()
    img.write_bytes(b"x")
    resolved, rel = resolve_image_path(
        "images/a.jpg", root=tmp_path, list_file=list_file
    )
    assert resolved.is_file()
    assert rel == "images/a.jpg"

    label_path = yolo_label_path_for_image(tmp_path / "datasets" / "images" / "a.jpg")
    assert "labels" in label_path.as_posix()

    labels = tmp_path / "a.txt"
    labels.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    inst = parse_yolo_label_file(labels, width=100, height=100)
    assert inst[0]["category_id"] == 0

    bad_label = tmp_path / "bad.txt"
    bad_label.write_text("0 0.1 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YOLO"):
        parse_yolo_label_file(bad_label, width=10, height=10)

    paths = iter_image_list(list_file, root=tmp_path)
    assert paths

    empty_list = tmp_path / "empty_train.txt"
    empty_list.write_text("#\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no image paths"):
        iter_image_list(empty_list, root=tmp_path)


def test_coco_dataset_errors_and_keypoints(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid COCO JSON"):
        load_coco_json(bad_json)

    no_cat = tmp_path / "no_cat.json"
    no_cat.write_text('{"images":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="categories"):
        load_category_map(no_cat, task=TaskType.DETECTION)

    raw = {
        "categories": [{"id": 1, "name": "cat", "keypoints": ["nose"]}],
        "annotations": [{"image_id": 1, "category_id": 1, "keypoints": [1, 2, 2]}],
    }
    cat_path = _write_json(tmp_path, "cats.json", raw)
    cats = load_category_map(cat_path, task=TaskType.KEYPOINT)
    assert cats[1].name == "cat"
    kps = coco_keypoints_flat(raw["annotations"][0])
    assert kps == [1.0, 2.0, 2]

    assert coco_segmentation_rle(None, height=10, width=10) is None
    rle = coco_segmentation_rle(
        {"size": [10, 10], "counts": "6320004"}, height=10, width=10
    )
    assert rle is not None


def _write_json(tmp_path: Path, name: str, payload: object) -> Path:
    import json

    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
