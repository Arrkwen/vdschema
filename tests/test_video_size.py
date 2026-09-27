"""Tests for video size resolution."""

from __future__ import annotations

from pathlib import Path

from vdswitch.utils import VideoSizeResolver, find_video_root, read_video_size

_SAMPLE_VIDEO = "video/sample.avi"


def test_find_video_root(act_legacy_dir: Path) -> None:
    root = find_video_root(
        output_dir=act_legacy_dir,
        input_data=act_legacy_dir / "meta/video_train.txt",
    )
    assert root == act_legacy_dir / "video"


def test_resolve_video_size(act_legacy_dir: Path) -> None:
    resolver = VideoSizeResolver(act_legacy_dir / "video")
    width, height = resolver.resolve(_SAMPLE_VIDEO)
    assert width == 640
    assert height == 480


def test_read_video_size(act_legacy_dir: Path) -> None:
    path = act_legacy_dir / "video" / "sample.avi"
    assert read_video_size(path) == (640, 480)
