"""LabelMe JSON → vdschema helpers.

Format reference: https://github.com/wkentaro/labelme
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vdschema import Name

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")

DETECTION_SHAPE_TYPES = frozenset(
    {"rectangle", "polygon", "line", "linestrip", "point", "circle"}
)
SEGMENTATION_SHAPE_TYPES = frozenset({"polygon"})
KEYPOINT_SHAPE_TYPES = frozenset({"point"})


def load_labelme_document(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"LabelMe JSON must be an object: {path}")
    if "shapes" not in raw:
        raise ValueError(f"LabelMe JSON missing shapes[]: {path}")
    return raw


def iter_labelme_json_paths(input_data: Path) -> list[Path]:
    if input_data.is_file():
        return [input_data]
    return sorted(p for p in input_data.rglob("*.json") if p.is_file())


def iter_shapes(document: dict[str, Any]) -> Iterator[dict[str, Any]]:
    shapes = document.get("shapes")
    if not isinstance(shapes, list):
        return
    for shape in shapes:
        if isinstance(shape, dict):
            yield shape


def points_to_flat(points: Any) -> list[float]:
    if not isinstance(points, list) or len(points) < 1:
        raise ValueError("points must be a non-empty list of [x, y]")
    coords: list[float] = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise ValueError("each point must be [x, y]")
        coords.extend([float(pt[0]), float(pt[1])])
    return coords


def rectangle_to_bbox_xyxy(points: Any) -> list[float]:
    flat = points_to_flat(points)
    xs = flat[0::2]
    ys = flat[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def circle_to_bbox_xyxy(points: Any) -> list[float]:
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("circle requires two points (center and edge)")
    cx, cy = float(points[0][0]), float(points[0][1])
    ex, ey = float(points[1][0]), float(points[1][1])
    radius = math.hypot(ex - cx, ey - cy)
    return [cx - radius, cy - radius, cx + radius, cy + radius]


def document_size(document: dict[str, Any]) -> tuple[int, int]:
    width = int(document.get("imageWidth") or document.get("width") or 0)
    height = int(document.get("imageHeight") or document.get("height") or 0)
    if width < 1 or height < 1:
        raise ValueError("LabelMe JSON must include imageWidth and imageHeight")
    return width, height


def resolve_media_filename(
    json_path: Path,
    document: dict[str, Any],
    *,
    input_data: Path,
    input_root: Path | None,
) -> str:
    image_path = document.get("imagePath")
    if image_path:
        text = str(image_path).replace("\\", "/")
        path = Path(text)
        if path.is_absolute():
            root = (input_root or input_data.parent).resolve()
            try:
                return path.resolve().relative_to(root).as_posix()
            except ValueError:
                return path.name
        return text

    root = input_root or input_data.parent
    stem = json_path.stem
    search_roots = [
        json_path.parent,
        input_data,
        root,
        root / "images",
        json_path.parent.parent,
    ]
    seen: set[Path] = set()
    for base in search_roots:
        for suffix in _IMAGE_SUFFIXES:
            candidate = (base / f"{stem}{suffix}").resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                try:
                    return candidate.relative_to(root.resolve()).as_posix()
                except ValueError:
                    return candidate.name

    if json_path.is_file() and input_data.is_dir():
        return json_path.relative_to(input_data).with_suffix(".jpg").as_posix()
    return f"{stem}.jpg"


def _shape_labels(document: dict[str, Any], shape_types: frozenset[str]) -> set[str]:
    labels: set[str] = set()
    for shape in iter_shapes(document):
        if str(shape.get("shape_type") or "") not in shape_types:
            continue
        labels.add(str(shape.get("label") or ""))
    return labels


def collect_label_ids(paths: list[Path], shape_types: frozenset[str]) -> dict[str, int]:
    labels: set[str] = set()
    for path in paths:
        doc = load_labelme_document(path)
        labels |= _shape_labels(doc, shape_types)
    if not labels:
        labels.add("")
    ordered = sorted(labels, key=lambda s: (s == "", s))
    return {label: idx + 1 for idx, label in enumerate(ordered)}


def label_dict_from_label_map(label_to_id: dict[str, int]) -> dict[int, Name]:
    return {
        cid: Name(label or "object") for label, cid in label_to_id.items()
    }


def _shape_text(shape: dict[str, Any]) -> str | None:
    desc = shape.get("description")
    if desc:
        return str(desc)
    return None


def detection_instances_from_document(
    document: dict[str, Any],
    *,
    label_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for idx, shape in enumerate(iter_shapes(document)):
        shape_type = str(shape.get("shape_type") or "")
        if shape_type not in DETECTION_SHAPE_TYPES:
            continue
        points = shape.get("points")
        label = str(shape.get("label") or "")
        inst: dict[str, Any] = {
            "id": idx,
            "category_id": label_to_id[label],
        }
        try:
            if shape_type == "rectangle":
                inst["bbox"] = rectangle_to_bbox_xyxy(points)
            elif shape_type == "polygon":
                inst["polygon"] = points_to_flat(points)
            elif shape_type in ("line", "linestrip"):
                flat = points_to_flat(points)
                if len(flat) < 4:
                    continue
                inst["polyline"] = flat
            elif shape_type == "point":
                flat = points_to_flat(points)
                inst["point"] = flat[:2]
            elif shape_type == "circle":
                inst["bbox"] = circle_to_bbox_xyxy(points)
        except ValueError:
            continue
        text = _shape_text(shape)
        if text:
            inst["text"] = text
        instances.append(inst)
    return instances


def segmentation_instances_from_document(
    document: dict[str, Any],
    *,
    label_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    next_id = 0
    for shape in iter_shapes(document):
        if str(shape.get("shape_type") or "") != "polygon":
            continue
        try:
            polygon = points_to_flat(shape.get("points"))
        except ValueError:
            continue
        if len(polygon) < 6:
            continue
        label = str(shape.get("label") or "")
        inst: dict[str, Any] = {
            "id": next_id,
            "category_id": label_to_id[label],
            "polygon": polygon,
        }
        next_id += 1
        text = _shape_text(shape)
        if text:
            inst["text"] = text
        instances.append(inst)
    return instances


def keypoint_instances_from_document(
    document: dict[str, Any],
    *,
    label_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    next_id = 0
    for shape in iter_shapes(document):
        if str(shape.get("shape_type") or "") != "point":
            continue
        try:
            flat = points_to_flat(shape.get("points"))
        except ValueError:
            continue
        label = str(shape.get("label") or "")
        inst: dict[str, Any] = {
            "id": next_id,
            "category_id": label_to_id[label],
            "keypoints": [flat[0], flat[1], 2],
        }
        next_id += 1
        text = _shape_text(shape)
        if text:
            inst["text"] = text
        instances.append(inst)
    return instances
