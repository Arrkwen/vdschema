"""Load MS COCO instance JSON for vdswitch converters."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pycocotools import mask as mask_util

from vdschema import Name, SegmentationRLE, TaskType

from ..help import help_hint


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


def load_coco_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"invalid COCO JSON (expected object): {path}")
    return raw


def load_category_map(label_path: Path, *, task: TaskType) -> dict[int, Name]:
    raw = load_coco_json(label_path)
    categories = raw.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(
            f"missing categories[] in COCO label file: {label_path}. {help_hint(task)}"
        )
    label: dict[int, Name] = {}
    for item in categories:
        if not isinstance(item, dict):
            continue
        label[int(item["id"])] = Name(str(item["name"]))
    if not label:
        raise ValueError(f"empty categories in COCO label file: {label_path}")
    return label


def iter_coco_images(raw: dict[str, Any]) -> Iterator[CocoImage]:
    images = raw.get("images") or []
    if not isinstance(images, list):
        raise ValueError("COCO images must be a list")
    for item in images:
        if not isinstance(item, dict):
            continue
        yield CocoImage(
            id=int(item["id"]),
            file_name=str(item["file_name"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )


def group_annotations(raw: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    annotations = raw.get("annotations") or []
    if not isinstance(annotations, list):
        raise ValueError("COCO annotations must be a list")
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def coco_bbox_xyxy(ann: dict[str, Any]) -> list[float]:
    x, y, width, height = (float(v) for v in ann["bbox"])
    return [x, y, x + width, y + height]


def coco_keypoints_flat(ann: dict[str, Any]) -> list[float | int] | None:
    """Return COCO annotation ``keypoints`` flat list unchanged when valid."""
    flat = ann.get("keypoints")
    if not isinstance(flat, list) or len(flat) < 3:
        return None
    if len(flat) % 3 != 0:
        return None
    out: list[float | int] = []
    for i, value in enumerate(flat):
        if i % 3 == 2:
            out.append(int(value))
        else:
            out.append(float(value))
    return out


def coco_segmentation_rle(
    segm: Any, *, height: int, width: int
) -> SegmentationRLE | None:
    if segm is None:
        return None
    if isinstance(segm, dict):
        size = segm.get("size") or [height, width]
        return SegmentationRLE.from_dict({"size": size, "counts": segm["counts"]})
    if isinstance(segm, list) and segm:
        polys = segm if isinstance(segm[0], list) else [segm]
        rles = mask_util.frPyObjects(polys, height, width)
        merged = mask_util.merge(rles)
        counts = merged["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return SegmentationRLE._create(
            int(merged["size"][0]), int(merged["size"][1]), str(counts)
        )
    return None
