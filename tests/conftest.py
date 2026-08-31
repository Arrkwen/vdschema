"""Shared fixtures for vdswitch tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def det_legacy_dir(tmp_path: Path) -> Path:
    root = tmp_path / "det"
    images = root / "images"
    meta = root / "meta"
    images.mkdir(parents=True)
    meta.mkdir(parents=True)

    Image.new("RGB", (320, 240), color=(128, 64, 32)).save(images / "sample.jpg")
    (meta / "label_dict.json").write_text('{"1": "person", "2": "car"}\n', encoding="utf-8")
    (meta / "train_baseline.jsonl").write_text(
        '{"filename":"sample.jpg","image_width":0,"image_height":0,'
        '"instances":[{"id":0,"label":1,"bbox":[10,10,100,100]}]}\n',
        encoding="utf-8",
    )
    return root


@pytest.fixture
def seq_legacy_dir(tmp_path: Path) -> Path:
    root = tmp_path / "seq"
    images = root / "images"
    meta = root / "meta"
    images.mkdir(parents=True)
    meta.mkdir(parents=True)

    Image.new("RGB", (224, 128), color=(64, 128, 192)).save(images / "sample.jpg")
    (meta / "vocab.txt").write_text("0\n1\nB\n", encoding="utf-8")
    (meta / "train_baseline.jsonl").write_text(
        '{"filename":"sample.jpg","sequences":["B","1","0"],'
        '"image_width":224,"image_height":128}\n',
        encoding="utf-8",
    )
    return root


@pytest.fixture
def act_legacy_dir(tmp_path: Path) -> Path:
    root = tmp_path / "act"
    video_dir = root / "video"
    kmot_dir = root / "meta" / "kmot"
    video_dir.mkdir(parents=True)
    kmot_dir.mkdir(parents=True)

    video_path = video_dir / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (640, 480),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(500):
        writer.write(frame)
    writer.release()

    (kmot_dir / "sample.txt").write_text(
        "442,18,76.0,280.0,104.0,230.0,221488,1.0,1,package_tossing\n"
        "443,18,80.0,274.0,100.0,236.0,221488,1.0,1,package_tossing\n"
        "444,18,84.0,268.0,96.0,242.0,221488,1.0,1,package_tossing\n"
        "445,18,88.0,262.0,92.0,248.0,221488,1.0,1,package_tossing\n",
        encoding="utf-8",
    )
    (root / "meta" / "label_dict.json").write_text(
        '{"package_tossing":["normal","package_tossing"]}\n',
        encoding="utf-8",
    )
    (root / "meta" / "video_train.txt").write_text(
        "video/sample.mp4;4;443;446;1;kmot/sample.txt\n",
        encoding="utf-8",
    )
    return root
