"""Tests for ImageNet-style classification source."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdschema import AnnotationReader, Source, TaskType, switch
from vdswitch.converters.registry import get_converter_class
from vdswitch.utils.imagenet_dataset import build_imagenet_head


def test_build_imagenet_head(imagenet_legacy_dir: Path) -> None:
    heads, name_to_id = build_imagenet_head(imagenet_legacy_dir)
    assert "class" in heads
    assert name_to_id["cat"] == 1
    assert name_to_id["dog"] == 2


def test_vdswitch_imagenet_minimal_args(
    imagenet_legacy_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.CLASSIFICATION,
        source=Source.IMAGENET,
        input_data=imagenet_legacy_dir,
        output=out,
    )
    data, _ = AnnotationReader(
        TaskType.CLASSIFICATION,
        out,
        task_data_filename="train.jsonl",
        task_meta_filename="label_dict.json",
    ).load()
    assert len(data) == 2


def test_vdswitch_imagenet_classification(
    imagenet_legacy_dir: Path, tmp_path: Path
) -> None:
    dataset_root = imagenet_legacy_dir.parent
    out = tmp_path / "out"
    switch(
        task=TaskType.CLASSIFICATION,
        source=Source.IMAGENET,
        input_data=imagenet_legacy_dir,
        input_label=imagenet_legacy_dir,
        output=out,
        input_root=dataset_root,
    )
    data, label = AnnotationReader(
        TaskType.CLASSIFICATION,
        out,
        task_data_filename="train.jsonl",
        task_meta_filename="label_dict.json",
    ).load()
    assert len(data) == 2
    assert label


def test_imagenet_unsupported_for_detection() -> None:
    with pytest.raises(ValueError, match="unsupported source='imagenet'"):
        get_converter_class(TaskType.DETECTION, Source.IMAGENET)
