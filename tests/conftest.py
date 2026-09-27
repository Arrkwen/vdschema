"""Shared fixtures for vdswitch tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_VIDEO_REL = "video/sample.avi"


def _write_sample_video(path: Path) -> None:
    """Write a short test clip that OpenCV can open on CI and macOS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25,
        (640, 480),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create test video: {path}")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(8):
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV cannot read test video: {path}")
    capture.release()


@pytest.fixture
def cls_legacy_dir(tmp_path: Path) -> Path:
    root = tmp_path / "cls"
    images = root / "images"
    meta = root / "meta"
    images.mkdir(parents=True)
    meta.mkdir(parents=True)

    Image.new("RGB", (100, 80), color=(200, 100, 50)).save(images / "sample.jpg")
    (meta / "label_dict.json").write_text(
        '{"gender":["male","female"]}\n', encoding="utf-8"
    )
    (meta / "train_baseline.jsonl").write_text(
        '{"filename":"sample.jpg","image_width":100,"image_height":80,'
        '"attribute":{"gender":{"male":0,"female":1}}}\n',
        encoding="utf-8",
    )
    return root


@pytest.fixture
def det_legacy_dir(tmp_path: Path) -> Path:
    root = tmp_path / "det"
    images = root / "images"
    meta = root / "meta"
    images.mkdir(parents=True)
    meta.mkdir(parents=True)

    Image.new("RGB", (320, 240), color=(128, 64, 32)).save(images / "sample.jpg")
    (meta / "label_dict.json").write_text(
        '{"1": "person", "2": "car"}\n', encoding="utf-8"
    )
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

    video_path = video_dir / "sample.avi"
    _write_sample_video(video_path)

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
        f"{SAMPLE_VIDEO_REL};4;443;446;1;kmot/sample.txt\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def coco_instances_path(tmp_path: Path) -> Path:
    root = tmp_path / "coco"
    images = root / "images"
    ann_dir = root / "annotations"
    images.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    Image.new("RGB", (200, 100), color=(10, 20, 30)).save(images / "sample.jpg")
    payload = {
        "images": [
            {"id": 1, "file_name": "sample.jpg", "width": 200, "height": 100},
        ],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 50.0, 40.0],
                "iscrowd": 0,
                "segmentation": [[10, 10, 60, 10, 60, 50, 10, 50]],
                "keypoints": [20.0, 20.0, 2, 30.0, 25.0, 2],
            },
            {
                "id": 11,
                "image_id": 1,
                "category_id": 1,
                "bbox": [80.0, 20.0, 20.0, 20.0],
                "iscrowd": 1,
            },
        ],
    }
    path = ann_dir / "instances.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def coco_keypoint_instances_path(coco_instances_path: Path) -> Path:
    payload = json.loads(coco_instances_path.read_text(encoding="utf-8"))
    payload["categories"] = [
        {
            "id": 1,
            "name": "person",
            "keypoints": ["nose", "left_eye"],
            "skeleton": [[1, 2]],
        }
    ]
    payload["annotations"] = [
        {
            "id": 10,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 50.0, 40.0],
            "iscrowd": 0,
            "num_keypoints": 2,
            "keypoints": [20.0, 20.0, 2, 30.0, 25.0, 2],
        }
    ]
    path = coco_instances_path.with_name("instances_keypoints.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def yolo_legacy_dir(tmp_path: Path) -> Path:
    root = tmp_path / "yolo"
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (200, 100), color=(255, 128, 64)).save(images / "sample.jpg")
    (labels / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (root / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
    (root / "train.txt").write_text("images/train/sample.jpg\n", encoding="utf-8")
    return root


@pytest.fixture
def imagenet_legacy_dir(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    train = root / "train"
    cat = train / "cat"
    dog = train / "dog"
    cat.mkdir(parents=True)
    dog.mkdir(parents=True)
    Image.new("RGB", (64, 48), color=(100, 100, 100)).save(cat / "a.jpg")
    Image.new("RGB", (64, 48), color=(200, 200, 200)).save(dog / "b.jpg")
    return train


@pytest.fixture
def ocr_legacy_dir(tmp_path: Path) -> Path:
    root = tmp_path / "ocr"
    images = root / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (80, 32), color=(50, 100, 150)).save(images / "001.jpg")
    (root / "vocab.txt").write_text("h\ne\nl\no\n", encoding="utf-8")
    (root / "anno.txt").write_text("images/001.jpg\thello\n", encoding="utf-8")
    return root
