"""Tests for PASCAL VOC source converters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vdschema import AnnotationReader, Name, Source, TaskType, switch
from vdswitch.converters.registry import get_converter_class, supported_sources
from vdswitch.utils.voc_dataset import (
    collect_detection_class_names,
    parse_voc_detection_xml,
    resolve_voc_root,
    segmentation_instances_from_class_png,
    voc2012_label_dict,
)


@pytest.fixture
def mini_voc_root(tmp_path: Path) -> Path:
    voc = tmp_path / "VOC2007"
    (voc / "Annotations").mkdir(parents=True)
    (voc / "JPEGImages").mkdir()
    (voc / "SegmentationClass").mkdir()
    (voc / "ImageSets" / "Main").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(voc / "JPEGImages" / "000001.jpg")
    (voc / "Annotations" / "000001.xml").write_text(
        """<?xml version="1.0"?>
<annotation>
  <filename>000001.jpg</filename>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <object>
    <name>cat</name>
    <difficult>0</difficult>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox>
  </object>
  <object>
    <name>dog</name>
    <difficult>1</difficult>
    <bndbox><xmin>60</xmin><ymin>10</ymin><xmax>90</xmax><ymax>40</ymax></bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )
    seg = np.zeros((80, 100), dtype=np.uint8)
    seg[10:30, 10:40] = 1
    seg[50:70, 50:90] = 2
    Image.fromarray(seg, mode="L").save(voc / "SegmentationClass" / "000001.png")
    (voc / "ImageSets" / "Main" / "train.txt").write_text("000001\n", encoding="utf-8")
    return voc


def test_voc_parse_xml(mini_voc_root: Path) -> None:
    record = parse_voc_detection_xml(mini_voc_root / "Annotations" / "000001.xml")
    assert record.width == 100
    assert len(record.objects) == 2
    assert record.objects[1].difficult is True


def test_voc_detection_switch(mini_voc_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.VOC,
        input_data=mini_voc_root / "ImageSets" / "Main" / "train.txt",
        output=out,
        input_root=mini_voc_root,
    )
    data, label = AnnotationReader(
        TaskType.DETECTION, out, task_data_filename="train.jsonl"
    ).load()
    assert len(data) == 1
    assert len(data[0].instances) == 2
    assert data[0].instances[1].is_ignored is True
    assert label[1].name in {"cat", "dog"}


def test_voc_segmentation_switch(mini_voc_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.SEGMENTATION,
        source=Source.VOC,
        input_data=mini_voc_root / "ImageSets" / "Main" / "train.txt",
        output=out,
        input_root=mini_voc_root,
    )
    data, label = AnnotationReader(
        TaskType.SEGMENTATION, out, task_data_filename="train.jsonl"
    ).load()
    assert len(data) == 1
    assert len(data[0].instances) == 2
    assert data[0].instances[0].rle_mask is not None
    assert label[1].name == "aeroplane"


def test_voc_segmentation_custom_label(tmp_path: Path) -> None:
    label = {1: Name("a"), 2: Name("b")}
    png = tmp_path / "m.png"
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[0:5, 0:5] = 1
    Image.fromarray(arr, mode="L").save(png)
    instances = segmentation_instances_from_class_png(png, label=label)
    assert len(instances) == 1
    assert instances[0]["category_id"] == 1


def test_voc_resolve_root(mini_voc_root: Path) -> None:
    assert resolve_voc_root(mini_voc_root, None) == mini_voc_root.resolve()
    assert resolve_voc_root(
        mini_voc_root / "ImageSets" / "Main" / "train.txt", None
    ) == mini_voc_root.resolve()


def test_voc_collect_names(mini_voc_root: Path) -> None:
    names = collect_detection_class_names(mini_voc_root, ["000001"])
    assert names["cat"] == 1
    assert names["dog"] == 2


def test_voc_registry() -> None:
    assert Source.VOC in supported_sources(TaskType.DETECTION)
    assert get_converter_class(TaskType.SEGMENTATION, Source.VOC)


def test_voc2012_label_count() -> None:
    assert len(voc2012_label_dict()) == 20


def test_voc_missing_seg_dir(mini_voc_root: Path, tmp_path: Path) -> None:
    import shutil

    broken = tmp_path / "voc"
    shutil.copytree(mini_voc_root, broken / "VOC2007")
    seg = broken / "VOC2007" / "SegmentationClass"
    shutil.rmtree(seg)
    with pytest.raises(FileNotFoundError, match="SegmentationClass"):
        switch(
            task=TaskType.SEGMENTATION,
            source=Source.VOC,
            input_data=broken / "VOC2007" / "ImageSets" / "Main" / "train.txt",
            output=tmp_path / "out",
        )


def test_voc_detection_via_voc_root_dir(mini_voc_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.VOC,
        input_data=mini_voc_root,
        output=out,
    )
    data, _ = AnnotationReader(
        TaskType.DETECTION, out, task_data_filename="VOC2007.jsonl"
    ).load()
    assert data


def test_voc_single_xml_file(mini_voc_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.VOC,
        input_data=mini_voc_root / "Annotations" / "000001.xml",
        output=out,
        input_root=mini_voc_root,
    )
    data, _ = AnnotationReader(
        TaskType.DETECTION, out, task_data_filename="000001.jsonl"
    ).load()
    assert len(data) == 1


def test_voc_optional_label_file(mini_voc_root: Path, tmp_path: Path) -> None:
    classes = tmp_path / "classes.txt"
    classes.write_text("cat\ndog\n", encoding="utf-8")
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.VOC,
        input_data=mini_voc_root / "ImageSets" / "Main" / "train.txt",
        input_label=classes,
        output=out,
        input_root=mini_voc_root,
    )
    data, label = AnnotationReader(
        TaskType.DETECTION, out, task_data_filename="train.jsonl"
    ).load()
    assert label[1].name == "cat"


def test_voc_iter_and_parse_errors(tmp_path: Path) -> None:
    from vdswitch.utils.voc_dataset import (
        detection_instances_from_record,
        iter_voc_image_ids,
        load_optional_class_list,
        resolve_media_filename,
        resolve_voc_root,
    )

    voc = tmp_path / "VOC"
    (voc / "Annotations").mkdir(parents=True)
    (voc / "Annotations" / "a.xml").write_text(
        """<annotation><filename>a.jpg</filename><size><width>1</width><height>1</height></size></annotation>""",
        encoding="utf-8",
    )
    assert iter_voc_image_ids(voc / "Annotations", voc) == ["a"]

    bad = tmp_path / "bad.xml"
    bad.write_text("<annotation></annotation>", encoding="utf-8")
    with pytest.raises(ValueError, match="filename"):
        parse_voc_detection_xml(bad)

    json_label = tmp_path / "labels.json"
    json_label.write_text('{"1":"x"}', encoding="utf-8")
    assert load_optional_class_list(json_label)[1].name == "x"

    record = parse_voc_detection_xml(voc / "Annotations" / "a.xml")
    name = resolve_media_filename(voc, "a", record, input_root=tmp_path)
    assert "a.jpg" in name

    with pytest.raises(ValueError, match="cannot infer"):
        resolve_voc_root(tmp_path / "list.txt", None)

    empty_classes = tmp_path / "empty.txt"
    empty_classes.write_text("\n", encoding="utf-8")
    assert load_optional_class_list(empty_classes) is None


def test_voc_detection_skips_unknown_class(mini_voc_root: Path) -> None:
    from vdswitch.utils.voc_dataset import detection_instances_from_record

    record = parse_voc_detection_xml(mini_voc_root / "Annotations" / "000001.xml")
    instances = detection_instances_from_record(record, name_to_id={"cat": 1})
    assert len(instances) == 1


def test_voc_segmentation_without_xml(mini_voc_root: Path, tmp_path: Path) -> None:
    (mini_voc_root / "Annotations" / "000001.xml").unlink()
    out = tmp_path / "out"
    switch(
        task=TaskType.SEGMENTATION,
        source=Source.VOC,
        input_data=mini_voc_root / "ImageSets" / "Main" / "train.txt",
        output=out,
        input_root=mini_voc_root,
    )
    data, _ = AnnotationReader(
        TaskType.SEGMENTATION, out, task_data_filename="train.jsonl"
    ).load()
    assert data


def test_voc_segmentation_bad_mask(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4)).save(rgb)
    with pytest.raises(ValueError, match="single-channel"):
        segmentation_instances_from_class_png(rgb, label=voc2012_label_dict())


def test_voc_iter_root_dir_and_empty_labels(mini_voc_root: Path, tmp_path: Path) -> None:
    from vdswitch.utils.voc_dataset import (
        collect_detection_class_names,
        iter_voc_image_ids,
        resolve_media_filename,
    )

    ids = iter_voc_image_ids(mini_voc_root, mini_voc_root)
    assert "000001" in ids

    empty_voc = tmp_path / "empty"
    (empty_voc / "Annotations").mkdir(parents=True)
    (empty_voc / "Annotations" / "z.xml").write_text(
        """<annotation><filename>z.jpg</filename><size><width>2</width><height>2</height></size></annotation>""",
        encoding="utf-8",
    )
    names = collect_detection_class_names(empty_voc, ["z"])
    assert names["object"] == 1

    rel = resolve_media_filename(
        mini_voc_root, "000001", None, input_root=mini_voc_root
    )
    assert rel.startswith("JPEGImages/")


def test_voc_detection_skips_missing_annotation(mini_voc_root: Path, tmp_path: Path) -> None:
    (mini_voc_root / "ImageSets" / "Main" / "train.txt").write_text(
        "000001\nmissing_id\n", encoding="utf-8"
    )
    out = tmp_path / "out"
    switch(
        task=TaskType.DETECTION,
        source=Source.VOC,
        input_data=mini_voc_root / "ImageSets" / "Main" / "train.txt",
        output=out,
        input_root=mini_voc_root,
    )
    data, _ = AnnotationReader(
        TaskType.DETECTION, out, task_data_filename="train.jsonl"
    ).load()
    assert len(data) == 1
