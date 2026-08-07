"""Tests for vdswitch output filename rules."""

from __future__ import annotations

from pathlib import Path

from vdswitch.utils import output_filename, resolve_output_filenames


def test_output_filename_different_dir() -> None:
    source = Path("/data/train_baseline.jsonl")
    assert output_filename(source, same_dir=False) == "train_baseline.jsonl"


def test_output_filename_same_dir() -> None:
    source = Path("/data/label_dict.json")
    assert output_filename(source, same_dir=True) == "label_dict_vdschema.json"


def test_resolve_output_filenames_different_dir(tmp_path: Path) -> None:
    data = tmp_path / "train_baseline.jsonl"
    label = tmp_path / "label_dict.json"
    out = tmp_path / "vdschema"
    data.touch()
    label.touch()

    data_name, label_name = resolve_output_filenames(
        input_data=data,
        input_label=label,
        output_dir=out,
    )
    assert data_name == "train_baseline.jsonl"
    assert label_name == "label_dict.json"


def test_resolve_output_filenames_same_dir(tmp_path: Path) -> None:
    data = tmp_path / "train_baseline.jsonl"
    label = tmp_path / "label_dict.json"
    data.touch()
    label.touch()

    data_name, label_name = resolve_output_filenames(
        input_data=data,
        input_label=label,
        output_dir=tmp_path,
    )
    assert data_name == "train_baseline_vdschema.jsonl"
    assert label_name == "label_dict_vdschema.json"
