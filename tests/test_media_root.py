"""Tests for legacy media root resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdswitch.utils import find_image_root, find_video_root


def test_find_image_root_with_explicit_root(
    det_legacy_dir: Path, tmp_path: Path
) -> None:
    root = find_image_root(
        output_dir=tmp_path / "out",
        input=det_legacy_dir / "meta/train_baseline.jsonl",
        root=det_legacy_dir,
    )
    assert root == det_legacy_dir / "images"


def test_find_video_root_with_explicit_root(
    act_legacy_dir: Path, tmp_path: Path
) -> None:
    root = find_video_root(
        output_dir=tmp_path / "out",
        input=act_legacy_dir / "meta/video_train.txt",
        root=act_legacy_dir,
    )
    assert root == act_legacy_dir / "video"


def test_find_image_root_missing_without_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="root="):
        find_image_root(
            output_dir=tmp_path / "out",
            input=tmp_path / "meta/data.jsonl",
        )
