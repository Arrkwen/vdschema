"""Resolve video width/height for action conversion."""

from __future__ import annotations

from pathlib import Path

import cv2

from .media_root import find_video_root


class VideoSizeResolver:
    """Read video width/height with OpenCV and caching."""

    def __init__(self, video_root: Path) -> None:
        self.video_root = Path(video_root)
        self._cache: dict[str, tuple[int, int]] = {}

    def resolve(self, filename: str) -> tuple[int, int]:
        key = str(filename)
        if key in self._cache:
            return self._cache[key]

        path = self._resolve_video_path(filename)
        size = read_video_size(path)
        self._cache[key] = size
        return size

    def _resolve_video_path(self, filename: str) -> Path:
        rel = Path(filename)
        candidates = (
            self.video_root / rel,
            self.video_root / rel.name,
            self.video_root / "video" / rel.name,
            self.video_root / filename,
        )
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        raise FileNotFoundError(
            f"video not found for {filename!r} under {self.video_root}"
        )


def read_video_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for a local video file."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"failed to open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width < 1 or height < 1:
        raise ValueError(f"invalid video size for {path}: {width}x{height}")
    return width, height


__all__ = ["VideoSizeResolver", "find_video_root", "read_video_size"]
