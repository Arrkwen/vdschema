"""Parse YOLO detection labels (Ultralytics / Darknet layout)."""

from __future__ import annotations

from pathlib import Path

from vdschema import Name, TaskType

from ..help import help_hint


def load_yolo_class_names(path: Path) -> dict[int, Name]:
    lines = path.read_text(encoding="utf-8").splitlines()
    names = [line.strip() for line in lines if line.strip()]
    if not names:
        raise ValueError(
            f"empty YOLO class list: {path}. {help_hint(TaskType.DETECTION)}"
        )
    return {idx: Name(name) for idx, name in enumerate(names)}


def resolve_image_path(
    line: str, *, root: Path | None, list_file: Path
) -> tuple[Path, str]:
    raw = line.strip()
    if not raw:
        raise ValueError(f"empty path line in {list_file}")
    rel = Path(raw).as_posix()
    path = Path(raw)
    if path.is_absolute():
        return path.resolve(), rel
    if root is not None:
        return (root / path).resolve(), rel
    return (list_file.parent / path).resolve(), rel


def yolo_label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_yolo_label_file(label_path: Path, *, width: int, height: int) -> list[dict]:
    if not label_path.is_file():
        return []
    instances: list[dict] = []
    text = label_path.read_text(encoding="utf-8")
    for idx, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"invalid YOLO label line in {label_path}: {line!r}")
        category_id = int(parts[0])
        cx, cy, bw, bh = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        instances.append(
            {
                "id": idx,
                "category_id": category_id,
                "bbox": [x1, y1, x2, y2],
            }
        )
    return instances


def iter_image_list(list_path: Path, *, root: Path | None) -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []
    with list_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                paths.append(resolve_image_path(line, root=root, list_file=list_path))
            except ValueError as exc:
                raise ValueError(f"{list_path}:{lineno}: {exc}") from exc
    if not paths:
        raise ValueError(f"no image paths in YOLO list file: {list_path}")
    return paths
