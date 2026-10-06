"""Tests for OCR sequence source."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import vdswitch_options
from vdschema import AnnotationReader, Source, TaskType, switch
from vdswitch.converters.registry import get_converter_class
from vdswitch.utils.ocr_dataset import iter_ocr_lines, tokenize_label


def test_tokenize_label() -> None:
    assert tokenize_label("hello") == ["h", "e", "l", "l", "o"]


def test_iter_ocr_lines(ocr_legacy_dir: Path) -> None:
    rows = iter_ocr_lines(ocr_legacy_dir / "anno.txt", root=ocr_legacy_dir)
    assert len(rows) == 1
    assert rows[0][1] == "images/001.jpg"
    assert rows[0][2] == ["h", "e", "l", "l", "o"]


def test_vdswitch_ocr_sequence(ocr_legacy_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    switch(
        task=TaskType.SEQUENCE,
        source=Source.OCR,
        input=ocr_legacy_dir / "anno.txt",
        output=out,
        options=vdswitch_options(
            category=ocr_legacy_dir / "vocab.txt",
            root=ocr_legacy_dir,
        ),
    )
    data, label = AnnotationReader(
        TaskType.SEQUENCE,
        out,
        task_data_filename="anno.jsonl",
        task_meta_filename="vocab.txt",
    ).load()
    assert len(data) == 1
    assert data[0].sequences == ["h", "e", "l", "l", "o"]
    assert label


def test_ocr_unsupported_for_detection() -> None:
    with pytest.raises(ValueError, match="unsupported source='ocr'"):
        get_converter_class(TaskType.DETECTION, Source.OCR)
