"""PASCAL VOC layout → vdschema helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from defusedxml import ElementTree as ET
from PIL import Image

from vdschema import Name, SegmentationRLE

# Standard PASCAL VOC 2012 class names (segmentation mask ids 1..20).
VOC2012_CLASS_NAMES: tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG")


@dataclass(frozen=True)
class VocObject:
    name: str
    bbox_xyxy: list[float]
    difficult: bool


@dataclass(frozen=True)
class VocDetectionRecord:
    filename: str
    width: int
    height: int
    objects: tuple[VocObject, ...]


def voc2012_label_dict() -> dict[int, Name]:
    return {idx + 1: Name(name) for idx, name in enumerate(VOC2012_CLASS_NAMES)}


def resolve_voc_root(input_data: Path, input_root: Path | None) -> Path:
    if input_data.is_dir() and (input_data / "Annotations").is_dir():
        return input_data.resolve()
    if input_root is not None:
        root = input_root.expanduser().resolve()
        if (root / "Annotations").is_dir():
            return root
    if input_data.is_file() and "ImageSets" in input_data.parts:
        candidate = input_data.parent.parent.parent
        if (candidate / "Annotations").is_dir():
            return candidate.resolve()
    raise ValueError(
        "cannot infer PASCAL VOC root; pass --input-root to the VOC year folder "
        "(containing Annotations/, JPEGImages/)"
    )


def iter_voc_image_ids(input_data: Path, voc_root: Path) -> list[str]:
    if input_data.is_file() and input_data.suffix.lower() == ".txt":
        return [
            line.strip()
            for line in input_data.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if input_data.is_file() and input_data.suffix.lower() == ".xml":
        return [input_data.stem]
    ann_dir = voc_root / "Annotations"
    if input_data.is_dir() and input_data.resolve() == ann_dir.resolve():
        return sorted(p.stem for p in input_data.glob("*.xml"))
    if input_data.is_dir():
        return sorted(p.stem for p in ann_dir.glob("*.xml"))
    raise ValueError(f"unsupported VOC --input-data: {input_data}")


def voc_annotation_xml_path(voc_root: Path, image_id: str) -> Path:
    return voc_root / "Annotations" / f"{image_id}.xml"


def voc_segmentation_class_path(voc_root: Path, image_id: str) -> Path:
    return voc_root / "SegmentationClass" / f"{image_id}.png"


def parse_voc_detection_xml(path: Path) -> VocDetectionRecord:
    if not path.is_file():
        raise FileNotFoundError(f"missing VOC annotation XML: {path}")
    root = ET.parse(path).getroot()
    filename_el = root.find("filename")
    if filename_el is None or not filename_el.text:
        raise ValueError(f"missing filename in {path}")
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing size in {path}")
    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))
    if width < 1 or height < 1:
        raise ValueError(f"invalid size in {path}")

    objects: list[VocObject] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", default="0"))
        ymin = float(bnd.findtext("ymin", default="0"))
        xmax = float(bnd.findtext("xmax", default="0"))
        ymax = float(bnd.findtext("ymax", default="0"))
        difficult = int(obj.findtext("difficult", default="0")) == 1
        objects.append(
            VocObject(
                name=str(name_el.text).strip(),
                bbox_xyxy=[xmin, ymin, xmax, ymax],
                difficult=difficult,
            )
        )
    return VocDetectionRecord(
        filename=str(filename_el.text).strip(),
        width=width,
        height=height,
        objects=tuple(objects),
    )


def collect_detection_class_names(
    voc_root: Path, image_ids: list[str]
) -> dict[str, int]:
    names: set[str] = set()
    for image_id in image_ids:
        xml_path = voc_annotation_xml_path(voc_root, image_id)
        if not xml_path.is_file():
            continue
        record = parse_voc_detection_xml(xml_path)
        for obj in record.objects:
            if obj.name:
                names.add(obj.name)
    if not names:
        names.add("object")
    ordered = sorted(names)
    return {name: idx + 1 for idx, name in enumerate(ordered)}


def detection_label_dict(name_to_id: dict[str, int]) -> dict[int, Name]:
    return {cid: Name(name) for name, cid in name_to_id.items()}


def detection_instances_from_record(
    record: VocDetectionRecord,
    *,
    name_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for idx, obj in enumerate(record.objects):
        cid = name_to_id.get(obj.name)
        if cid is None:
            continue
        item: dict[str, Any] = {
            "id": idx,
            "category_id": cid,
            "bbox": list(obj.bbox_xyxy),
        }
        if obj.difficult:
            item["is_ignored"] = True
        instances.append(item)
    return instances


def resolve_media_filename(
    voc_root: Path,
    image_id: str,
    record: VocDetectionRecord | None,
    *,
    input_root: Path | None,
) -> str:
    if record is not None and record.filename:
        text = record.filename.replace("\\", "/")
        if not Path(text).is_absolute():
            candidate = voc_root / "JPEGImages" / Path(text).name
            if candidate.is_file():
                text = f"JPEGImages/{Path(text).name}"
        rel_root = input_root or voc_root.parent
        try:
            return (
                (voc_root / text).resolve().relative_to(rel_root.resolve()).as_posix()
            )
        except ValueError:
            return text

    for suffix in _IMAGE_SUFFIXES:
        candidate = voc_root / "JPEGImages" / f"{image_id}{suffix}"
        if candidate.is_file():
            rel = f"JPEGImages/{image_id}{suffix}"
            rel_root = input_root or voc_root.parent
            try:
                return candidate.resolve().relative_to(rel_root.resolve()).as_posix()
            except ValueError:
                return rel
    return f"JPEGImages/{image_id}.jpg"


def segmentation_instances_from_class_png(
    path: Path,
    *,
    label: dict[int, Name],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    mask = np.array(Image.open(path))
    if mask.ndim != 2:
        raise ValueError(f"SegmentationClass must be single-channel index map: {path}")
    instances: list[dict[str, Any]] = []
    next_id = 0
    for class_id in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
        if class_id not in label:
            continue
        binary = (mask == class_id).astype(np.uint8)
        if not binary.any():
            continue
        rle = SegmentationRLE.from_mask(binary)
        instances.append(
            {
                "id": next_id,
                "category_id": class_id,
                "rle_mask": rle.to_dict(),
            }
        )
        next_id += 1
    return instances


def load_optional_class_list(label_path: Path) -> dict[int, Name] | None:
    if not label_path.is_file():
        return None
    if label_path.suffix.lower() == ".json":
        import json

        raw = json.loads(label_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {int(k): Name(str(v)) for k, v in raw.items()}
        return None
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return None
    return {idx + 1: Name(name) for idx, name in enumerate(lines)}
