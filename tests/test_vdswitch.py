"""Tests for vdswitch CLI converters."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vdschema import AnnotationReader, Source, TaskType, switch


def test_vdswitch_det_up(det_legacy_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "vdschema_det"
    switch(
        task=TaskType.DETECTION,
        source=Source.UP,
        input_data=det_legacy_dir / "meta/train_baseline.jsonl",
        input_label=det_legacy_dir / "meta/label_dict.json",
        output=out,
        input_root=det_legacy_dir,
    )
    data, label = AnnotationReader(
        TaskType.DETECTION,
        out,
        task_data_filename="train_baseline.jsonl",
        task_meta_filename="label_dict.json",
    ).load()
    assert data
    assert label
    assert data[0].width == 320
    assert data[0].height == 240


def test_vdswitch_rejects_monolith_source(det_legacy_dir: Path) -> None:
    with pytest.raises(ValueError, match="monolith"):
        Source.parse("monolith")


def test_vdswitch_det_same_dir(det_legacy_dir: Path, tmp_path: Path) -> None:
    data = tmp_path / "train_baseline.jsonl"
    label = tmp_path / "label_dict.json"
    original_data = (det_legacy_dir / "meta/train_baseline.jsonl").read_text(
        encoding="utf-8"
    )
    original_label = (det_legacy_dir / "meta/label_dict.json").read_text(
        encoding="utf-8"
    )
    data.write_text(original_data, encoding="utf-8")
    label.write_text(original_label, encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    shutil.copy(det_legacy_dir / "images/sample.jpg", images / "sample.jpg")

    switch(
        task=TaskType.DETECTION,
        source=Source.UP,
        input_data=data,
        input_label=label,
        output=tmp_path,
    )
    assert (tmp_path / "train_baseline_vdschema.jsonl").is_file()
    assert (tmp_path / "label_dict_vdschema.json").is_file()


def test_vdswitch_det_multiple_input_data(det_legacy_dir: Path, tmp_path: Path) -> None:
    train = det_legacy_dir / "meta/train_baseline.jsonl"
    test = det_legacy_dir / "meta/test_baseline.jsonl"
    shutil.copy(train, test)
    out = tmp_path / "vdschema_det_multi"
    switch(
        task=TaskType.DETECTION,
        source=Source.UP,
        input_data=[train, test],
        input_label=det_legacy_dir / "meta/label_dict.json",
        output=out,
        input_root=det_legacy_dir,
    )
    assert (out / "train_baseline.jsonl").is_file()
    assert (out / "test_baseline.jsonl").is_file()
    assert (out / "test_baseline.jsonl").is_file()


def test_vdswitch_cls_up(cls_legacy_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "vdschema_cls"
    switch(
        task=TaskType.CLASSIFICATION,
        source=Source.UP,
        input_data=cls_legacy_dir / "meta/train_baseline.jsonl",
        input_label=cls_legacy_dir / "meta/label_dict.json",
        output=out,
        input_root=cls_legacy_dir,
    )
    data, label = AnnotationReader(
        TaskType.CLASSIFICATION,
        out,
        task_data_filename="train_baseline.jsonl",
        task_meta_filename="label_dict.json",
    ).load()
    assert data
    assert label
    assert data[0].width == 100
    assert data[0].height == 80
    assert data[0].categories[0].category_attr == "gender"
    assert data[0].categories[0].category_ids == [2]


def test_vdswitch_help_classification() -> None:
    from vdswitch.cli import main

    assert main(["help", "--task", "classification"]) == 0


def test_vdswitch_help_list_tasks() -> None:
    from vdswitch.cli import main

    assert main(["help"]) == 0


def test_vdswitch_help_requires_task_with_source() -> None:
    from vdswitch.cli import main

    assert main(["help", "--source", "up"]) == 2


def test_vdswitch_cli_multiple_input_data(det_legacy_dir: Path, tmp_path: Path) -> None:
    from vdswitch.cli import main

    train = det_legacy_dir / "meta/train_baseline.jsonl"
    test = det_legacy_dir / "meta/test_baseline.jsonl"
    shutil.copy(train, test)
    out = tmp_path / "cli_out"
    assert (
        main(
            [
                "--task",
                "detection",
                "--source",
                "up",
                "--input-data",
                str(train),
                str(test),
                "--input-label",
                str(det_legacy_dir / "meta/label_dict.json"),
                "--output",
                str(out),
                "--input-root",
                str(det_legacy_dir),
            ]
        )
        == 0
    )
    assert (out / "train_baseline.jsonl").is_file()
    assert (out / "test_baseline.jsonl").is_file()


def test_vdswitch_action(act_legacy_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "vdschema_act"
    switch(
        task=TaskType.ACTION,
        source=Source.UP,
        input_data=act_legacy_dir / "meta/video_train.txt",
        input_label=act_legacy_dir / "meta/label_dict.json",
        output=out,
        input_root=act_legacy_dir,
    )
    data, label = AnnotationReader(
        TaskType.ACTION,
        out,
        task_data_filename="video_train.txt",
        task_meta_filename="label_dict.json",
    ).load()
    assert data
    assert label
    assert data[0].width == 640
    assert data[0].height == 480
    assert data[0].actions[0].category_id == 1
    assert data[0].actions[0].start_idx == 442
    assert data[0].actions[0].end_idx == 445
    assert len(data) == 1


def test_vdswitch_action_one_jsonl_per_meta_line(
    act_legacy_dir: Path, tmp_path: Path
) -> None:
    """同一视频多条 meta 行应各自输出一条 jsonl。"""
    meta_path = act_legacy_dir / "meta/video_train.txt"
    meta_path.write_text(
        meta_path.read_text(encoding="utf-8")
        + "video/sample.avi;4;100;103;0;kmot/sample.txt\n",
        encoding="utf-8",
    )
    out = tmp_path / "vdschema_act_multi"
    switch(
        task=TaskType.ACTION,
        source=Source.UP,
        input_data=meta_path,
        input_label=act_legacy_dir / "meta/label_dict.json",
        output=out,
        input_root=act_legacy_dir,
    )
    data, _ = AnnotationReader(
        TaskType.ACTION,
        out,
        task_data_filename="video_train.txt",
        task_meta_filename="label_dict.json",
    ).load()
    assert len(data) == 2
    assert all(item.filename == "video/sample.avi" for item in data)
    assert all(len(item.actions) == 1 for item in data)


def test_vdswitch_sequence(seq_legacy_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "vdschema_seq"
    switch(
        task=TaskType.SEQUENCE,
        source=Source.UP,
        input_data=seq_legacy_dir / "meta/train_baseline.jsonl",
        input_label=seq_legacy_dir / "meta/vocab.txt",
        output=out,
        input_root=seq_legacy_dir,
    )
    data, label = AnnotationReader(
        TaskType.SEQUENCE,
        out,
        task_data_filename="train_baseline.jsonl",
        task_meta_filename="vocab.txt",
    ).load()
    assert data
    assert label == {"vocab": "vocab.txt"}
    assert (out / "vocab.txt").read_text(encoding="utf-8") == (
        seq_legacy_dir / "meta/vocab.txt"
    ).read_text(encoding="utf-8")
    assert data[0].width == 224
    assert data[0].height == 128
    assert data[0].sequences == ["B", "1", "0"]


def test_vdswitch_sequence_same_dir(seq_legacy_dir: Path, tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    images = tmp_path / "images"
    meta.mkdir()
    images.mkdir()
    shutil.copy(seq_legacy_dir / "images/sample.jpg", images / "sample.jpg")
    shutil.copy(seq_legacy_dir / "meta/vocab.txt", meta / "vocab.txt")
    shutil.copy(
        seq_legacy_dir / "meta/train_baseline.jsonl", meta / "train_baseline.jsonl"
    )

    switch(
        task=TaskType.SEQUENCE,
        source=Source.UP,
        input_data=meta / "train_baseline.jsonl",
        input_label=meta / "vocab.txt",
        output=meta,
        input_root=tmp_path,
    )
    assert (meta / "train_baseline_vdschema.jsonl").is_file()
    assert (meta / "vocab_vdschema.txt").is_file()
    assert not (meta / "annotation_vocab.txt").is_file()


def test_vdswitch_sequence_mixed_input_dirs(
    seq_legacy_dir: Path, tmp_path: Path
) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    vocab = labels / "vocab.txt"
    vocab.write_text(
        (seq_legacy_dir / "meta/vocab.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    out = seq_legacy_dir / "meta"
    switch(
        task=TaskType.SEQUENCE,
        source=Source.UP,
        input_data=seq_legacy_dir / "meta/train_baseline.jsonl",
        input_label=vocab,
        output=out,
        input_root=seq_legacy_dir,
    )
    assert (out / "train_baseline_vdschema.jsonl").is_file()
    assert (out / "vocab.txt").is_file()
    assert not (out / "vocab_vdschema.txt").is_file()
