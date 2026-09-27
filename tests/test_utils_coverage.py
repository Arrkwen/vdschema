"""Coverage for vdswitch utils edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vdschema import TaskType
from vdswitch.utils.coco_dataset import (
    group_annotations,
    iter_coco_images,
    load_category_map,
)
from vdswitch.utils.image_size import ImageSizeResolver
from vdswitch.utils.imagenet_dataset import build_imagenet_head, sorted_class_dirs
from vdswitch.utils.kmot import KmotTrack, parse_kmot_file
from vdswitch.utils.media_root import find_image_root
from vdswitch.utils.video_size import VideoSizeResolver
from vdswitch.utils.yolo_dataset import resolve_image_path


def test_imagenet_sorted_class_dirs_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sorted_class_dirs(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no class subdirectories"):
        sorted_class_dirs(empty)


def test_imagenet_build_head(imagenet_legacy_dir: Path) -> None:
    heads, name_to_id = build_imagenet_head(imagenet_legacy_dir)
    assert "class" in heads
    assert name_to_id


def test_coco_group_and_iter_errors(tmp_path: Path) -> None:

    with pytest.raises(ValueError, match="annotations must be a list"):
        group_annotations({"annotations": "bad"})

    bad_images = {
        "images": "bad",
        "categories": [{"id": 1, "name": "a"}],
        "annotations": [],
    }
    with pytest.raises(ValueError, match="images must be a list"):
        list(iter_coco_images(bad_images))


def test_coco_skip_non_dict_category(tmp_path: Path) -> None:
    import json

    path = tmp_path / "c.json"
    path.write_text(
        json.dumps({"categories": ["skip", {"id": 1, "name": "a"}]}),
        encoding="utf-8",
    )
    label = load_category_map(path, task=TaskType.DETECTION)
    assert 1 in label


def test_find_image_root_from_output_dir(det_legacy_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    shutil_copy = det_legacy_dir / "images" / "sample.jpg"
    if shutil_copy.is_file():
        import shutil

        shutil.copy(shutil_copy, out / "images" / "sample.jpg")
    root = find_image_root(
        output_dir=out,
        input_data=det_legacy_dir / "meta/train_baseline.jsonl",
    )
    assert root == out / "images"
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    found = find_image_root(
        output_dir=tmp_path / "out",
        input_data=root / "meta" / "data.jsonl",
        root=root,
    )
    assert found == root


def test_yolo_absolute_image_path(tmp_path: Path) -> None:
    img = tmp_path / "abs.jpg"
    img.write_bytes(b"x")
    resolved, _rel = resolve_image_path(
        str(img.resolve()), root=tmp_path, list_file=tmp_path / "t.txt"
    )
    assert resolved == img.resolve()


def test_media_root_bad_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--root not found"):
        find_image_root(
            output_dir=tmp_path / "out",
            input_data=tmp_path / "meta/data.jsonl",
            root=tmp_path / "nope",
        )


def test_image_size_absolute_path(tmp_path: Path) -> None:
    img = tmp_path / "abs.jpg"
    Image.new("RGB", (11, 12)).save(img)
    resolver = ImageSizeResolver(tmp_path)
    w, h = resolver.resolve({"filename": str(img.resolve())})
    assert (w, h) == (11, 12)


def test_yolo_resolve_empty_line(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty path"):
        resolve_image_path("  ", root=tmp_path, list_file=tmp_path / "train.txt")


def test_kmot_numeric_category_id(tmp_path: Path) -> None:
    kmot = tmp_path / "t.txt"
    kmot.write_text("0,1,0,0,1,1,0,1\n", encoding="utf-8")
    tracks = parse_kmot_file(kmot, name_to_id={"x": 0})
    assert tracks[1].category_id == 1


def test_kmot_empty_track_action_dict() -> None:
    track = KmotTrack(track_id=1)
    with pytest.raises(ValueError):
        track.to_action_dict()


def test_video_size_cache_and_missing(act_legacy_dir: Path, tmp_path: Path) -> None:
    resolver = VideoSizeResolver(act_legacy_dir / "video")
    first = resolver.resolve("video/sample.avi")
    second = resolver.resolve("video/sample.avi")
    assert first == second
    with pytest.raises(FileNotFoundError, match="video not found"):
        resolver.resolve("no_such_video.avi")
