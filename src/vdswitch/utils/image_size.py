"""Resolve annotation width/height from legacy records or image files."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .media_root import find_image_root


class ImageSizeResolver:
    """Read missing width/height from image files with caching."""

    def __init__(self, image_root: Path) -> None:
        self.image_root = Path(image_root)
        self._cache: dict[str, tuple[int, int]] = {}

    def resolve(self, record: dict) -> tuple[int, int]:
        width = int(record.get("image_width", record.get("width", 0)))
        height = int(record.get("image_height", record.get("height", 0)))
        if width >= 1 and height >= 1:
            return width, height

        filename = str(record["filename"])
        if filename in self._cache:
            return self._cache[filename]

        path = self._resolve_image_path(filename)
        with Image.open(path) as img:
            size = img.size
        self._cache[filename] = size
        return size

    def _resolve_image_path(self, filename: str) -> Path:
        rel = Path(filename)
        candidates = (
            self.image_root / rel.name,
            self.image_root / rel,
            self.image_root / filename,
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
            f"image not found for {filename!r} under {self.image_root}"
        )


__all__ = ["ImageSizeResolver", "find_image_root"]
