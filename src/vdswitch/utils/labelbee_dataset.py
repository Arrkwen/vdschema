"""LabelBee General Data JSON → vdschema helpers.

Format reference: https://github.com/open-mmlab/labelbee-client/tree/main/docs/annotation
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vdschema import Name

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


def load_labelbee_document(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"LabelBee JSON must be an object: {path}")
    return raw


def iter_labelbee_json_paths(input: Path) -> list[Path]:
    if input.is_file():
        return [input]
    return sorted(p for p in input.rglob("*.json") if p.is_file())


def iter_steps(document: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for key in sorted(document.keys()):
        if not key.startswith("step_"):
            continue
        step = document[key]
        if isinstance(step, dict):
            yield key, step


def step_tool_results(document: dict[str, Any], tool_name: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for _, step in iter_steps(document):
        if step.get("toolName") != tool_name:
            continue
        result = step.get("result")
        if isinstance(result, list):
            items.extend(item for item in result if isinstance(item, dict))
    return items


def point_list_to_flat(point_list: Any) -> list[float]:
    if not isinstance(point_list, list) or len(point_list) < 2:
        raise ValueError("pointList must contain at least two points")
    coords: list[float] = []
    for point in point_list:
        if not isinstance(point, dict):
            raise ValueError("pointList entries must be objects with x/y")
        coords.extend([float(point["x"]), float(point["y"])])
    return coords


def rect_to_bbox_xyxy(item: dict[str, Any]) -> list[float]:
    x = float(item["x"])
    y = float(item["y"])
    w = float(item["width"])
    h = float(item["height"])
    return [x, y, x + w, y + h]


def collect_detection_attributes(paths: list[Path]) -> dict[str, int]:
    attrs: set[str] = set()
    for path in paths:
        doc = load_labelbee_document(path)
        for tool in ("rectTool", "lineTool", "polygonTool"):
            for item in step_tool_results(doc, tool):
                if item.get("valid", True) is False:
                    continue
                attrs.add(str(item.get("attribute") or ""))
    if not attrs:
        attrs.add("")
    ordered = sorted(attrs, key=lambda s: (s == "", s))
    return {attr: idx + 1 for idx, attr in enumerate(ordered)}


def collect_tag_heads(paths: list[Path]) -> dict[str, dict[int, Name]]:
    options: dict[str, set[str]] = {}
    for path in paths:
        doc = load_labelbee_document(path)
        for item in step_tool_results(doc, "tagTool"):
            tag_map = item.get("result")
            if not isinstance(tag_map, dict):
                continue
            for attr, value in tag_map.items():
                head = str(attr)
                options.setdefault(head, set())
                for part in str(value).split(";"):
                    part = part.strip()
                    if part:
                        options[head].add(part)
    heads: dict[str, dict[int, Name]] = {}
    for head, values in sorted(options.items()):
        ordered = sorted(values)
        heads[head] = {idx + 1: Name(name) for idx, name in enumerate(ordered)}
    return heads


def collect_point_attributes(paths: list[Path]) -> dict[str, int]:
    attrs: set[str] = set()
    for path in paths:
        doc = load_labelbee_document(path)
        for item in step_tool_results(doc, "pointTool"):
            if item.get("valid", True) is False:
                continue
            attrs.add(str(item.get("attribute") or ""))
    if not attrs:
        attrs.add("")
    ordered = sorted(attrs, key=lambda s: (s == "", s))
    return {attr: idx + 1 for idx, attr in enumerate(ordered)}


def collect_polygon_attributes(paths: list[Path]) -> dict[str, int]:
    attrs: set[str] = set()
    for path in paths:
        doc = load_labelbee_document(path)
        for item in step_tool_results(doc, "polygonTool"):
            if item.get("valid", True) is False:
                continue
            attrs.add(str(item.get("attribute") or ""))
    if not attrs:
        attrs.add("")
    ordered = sorted(attrs, key=lambda s: (s == "", s))
    return {attr: idx + 1 for idx, attr in enumerate(ordered)}


def resolve_media_filename(
    json_path: Path,
    document: dict[str, Any],
    *,
    input: Path,
    root: Path | None,
) -> str:
    for key in ("file_name", "filename", "path", "img_path"):
        value = document.get(key)
        if value:
            return str(value).replace("\\", "/")

    root = root or input.parent
    stem = json_path.stem
    search_roots = [
        json_path.parent,
        input,
        root,
        root / "images",
        input.parent / "images",
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

    if json_path.is_file() and input.is_dir():
        rel = json_path.relative_to(input).with_suffix(".jpg")
        return rel.as_posix()
    return f"{stem}.jpg"


def detection_instances_from_document(
    document: dict[str, Any],
    *,
    attr_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    next_id = 0

    def append_instance(**fields: Any) -> None:
        nonlocal next_id
        inst = {"id": next_id, **fields}
        next_id += 1
        instances.append(inst)

    for item in step_tool_results(document, "rectTool"):
        attr = str(item.get("attribute") or "")
        inst: dict[str, Any] = {
            "category_id": attr_to_id[attr],
            "bbox": rect_to_bbox_xyxy(item),
        }
        if item.get("valid", True) is False:
            inst["is_ignored"] = True
        text = item.get("textAttribute")
        if text:
            inst["text"] = str(text)
        append_instance(**inst)

    for item in step_tool_results(document, "lineTool"):
        attr = str(item.get("attribute") or "")
        inst = {
            "category_id": attr_to_id[attr],
            "polyline": point_list_to_flat(item.get("pointList")),
        }
        if item.get("valid", True) is False:
            inst["is_ignored"] = True
        text = item.get("textAttribute")
        if text:
            inst["text"] = str(text)
        append_instance(**inst)

    for item in step_tool_results(document, "polygonTool"):
        attr = str(item.get("attribute") or "")
        inst = {
            "category_id": attr_to_id[attr],
            "polygon": point_list_to_flat(item.get("pointList")),
        }
        if item.get("valid", True) is False:
            inst["is_ignored"] = True
        text = item.get("textAttribute")
        if text:
            inst["text"] = str(text)
        append_instance(**inst)

    return instances


def segmentation_instances_from_document(
    document: dict[str, Any],
    *,
    attr_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for idx, item in enumerate(step_tool_results(document, "polygonTool")):
        attr = str(item.get("attribute") or "")
        inst: dict[str, Any] = {
            "id": idx,
            "category_id": attr_to_id[attr],
            "polygon": point_list_to_flat(item.get("pointList")),
        }
        if item.get("valid", True) is False:
            inst["is_ignored"] = True
        text = item.get("textAttribute")
        if text:
            inst["text"] = str(text)
        instances.append(inst)
    return instances


def keypoint_instances_from_document(
    document: dict[str, Any],
    *,
    attr_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for idx, item in enumerate(step_tool_results(document, "pointTool")):
        attr = str(item.get("attribute") or "")
        vis = 2 if item.get("valid", True) is not False else 0
        inst: dict[str, Any] = {
            "id": idx,
            "category_id": attr_to_id[attr],
            "keypoints": [float(item["x"]), float(item["y"]), vis],
        }
        text = item.get("textAttribute")
        if text:
            inst["text"] = str(text)
        instances.append(inst)
    return instances


def classification_categories_from_document(
    document: dict[str, Any],
    *,
    heads: dict[str, dict[int, Name]],
) -> list[dict[str, Any]]:
    name_to_id: dict[str, dict[str, int]] = {}
    for head, mapping in heads.items():
        name_to_id[head] = {name.name: cid for cid, name in mapping.items()}

    categories: list[dict[str, Any]] = []
    tag_items = step_tool_results(document, "tagTool")
    if not tag_items:
        return categories
    tag_map = tag_items[0].get("result")
    if not isinstance(tag_map, dict):
        return categories
    for attr, value in tag_map.items():
        head = str(attr)
        lookup = name_to_id.get(head)
        if not lookup:
            continue
        selected = [part.strip() for part in str(value).split(";") if part.strip()]
        category_ids = [lookup[name] for name in selected if name in lookup]
        if not category_ids:
            continue
        categories.append({"category_attr": head, "category_ids": category_ids})
    return categories


def label_dict_from_attr_map(attr_to_id: dict[str, int]) -> dict[int, Name]:
    id_to_name: dict[int, Name] = {}
    for attr, cid in attr_to_id.items():
        id_to_name[cid] = Name(attr or "object")
    return id_to_name
