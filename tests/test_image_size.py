"""Tests for image size resolution."""

from __future__ import annotations

import json
from pathlib import Path

from vdswitch.utils import ImageSizeResolver, find_image_root


def test_resolve_size_from_image_when_missing(det_legacy_dir: Path) -> None:
    image_root = find_image_root(
        output_dir=det_legacy_dir,
        input_data=det_legacy_dir / "meta/train_baseline.jsonl",
    )
    assert image_root == det_legacy_dir / "images"

    record = json.loads(
        (det_legacy_dir / "meta/train_baseline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    record["image_width"] = 0
    record["image_height"] = 0

    width, height = ImageSizeResolver(image_root).resolve(record)
    assert width == 320
    assert height == 240
