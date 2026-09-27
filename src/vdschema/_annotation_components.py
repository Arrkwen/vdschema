"""
Internal data structures aligned with annotation_data.

Application code usually does not need to instantiate these classes directly.
annotation_format converts plain bbox lists, dicts, mask arrays, and other
basic inputs into these structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


class AnnotationFormatError(ValueError):
    """Raised when annotation data does not match the expected schema."""


def _ensure_positive_int(value: Any, field_name: str) -> int:
    value = int(value)
    if value < 1:
        raise AnnotationFormatError(f"{field_name} must be >= 1")
    return value


@dataclass
class Bbox:
    """Axis-aligned box [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise AnnotationFormatError(
                f"bbox must satisfy x2>x1 and y2>y1, got "
                f"[{self.x1}, {self.y1}, {self.x2}, {self.y2}]"
            )

    def to_list(self) -> list[float]:
        self.validate()
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2)]

    @classmethod
    def from_list(cls, raw: list[float | int]) -> Bbox:
        if len(raw) != 4:
            raise AnnotationFormatError("bbox must contain exactly 4 values")
        return cls(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))

    @classmethod
    def from_xyxy(cls, raw: list[float | int]) -> Bbox:
        return cls.from_list(raw)

    @classmethod
    def from_xywh(cls, raw: list[float | int]) -> Bbox:
        if len(raw) != 4:
            raise AnnotationFormatError("bbox must contain exactly 4 values")
        x, y, width, height = [float(v) for v in raw]
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_cxcywh(cls, raw: list[float | int]) -> Bbox:
        if len(raw) != 4:
            raise AnnotationFormatError("bbox must contain exactly 4 values")
        cx, cy, width, height = [float(v) for v in raw]
        half_width = width / 2
        half_height = height / 2
        return cls(
            cx - half_width,
            cy - half_height,
            cx + half_width,
            cy + half_height,
        )


@dataclass
class SegmentationRLE:
    """
    pycocotools RLE: size=[height, width], counts=compressed string.

    - Encode: ``SegmentationRLE.from_mask(numpy_mask)``
    - Decode: ``rle.to_mask()``
    - Deserialize from JSON: ``from_dict(dict)``
    """

    height: int
    width: int
    counts: str
    _allow_init: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self._allow_init:
            raise AnnotationFormatError(
                "Use SegmentationRLE.from_mask() or from_dict(); "
                "do not construct with height/width/counts directly"
            )

    @classmethod
    def _create(cls, height: int, width: int, counts: str) -> SegmentationRLE:
        return cls(
            height=int(height),
            width=int(width),
            counts=counts,
            _allow_init=True,
        )

    def validate(self) -> None:
        if self.height < 1 or self.width < 1:
            raise AnnotationFormatError(
                "rle_mask.size must contain positive integers"
            )
        if not self.counts:
            raise AnnotationFormatError("rle_mask.counts must not be empty")

    def _rle_dict(self) -> dict[str, Any]:
        """RLE dict for pycocotools.decode."""
        counts = self.counts
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        return {"size": [int(self.height), int(self.width)], "counts": counts}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SegmentationRLE:
        size = raw["size"]
        counts = raw["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return cls._create(int(size[0]), int(size[1]), str(counts))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "size": [int(self.height), int(self.width)],
            "counts": self.counts,
        }

    @classmethod
    def from_mask(cls, mask_hw: Any) -> SegmentationRLE:
        """Convert a numpy ``[H, W]`` uint8 binary mask to RLE."""
        arr = np.asfortranarray(mask_hw)
        if arr.ndim != 2:
            raise AnnotationFormatError(
                "from_mask input must be a 2D [height, width] array"
            )
        rle = mask_utils.encode(arr)
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        size = rle["size"]
        return cls._create(int(size[0]), int(size[1]), counts)

    def to_mask(self) -> Any:
        """Convert RLE to a numpy uint8 mask with shape ``[height, width]``."""
        self.validate()
        return mask_utils.decode(self._rle_dict()).astype("uint8")


@dataclass
class Keypoint:
    """
    Single point [x, y, visibility].

    COCO-style visibility values:
    - 0: not labeled.
    - 1: labeled but not visible.
    - 2: labeled and visible.
    """

    x: float
    y: float
    visibility: int = 2

    def validate(self) -> None:
        if not 0 <= self.visibility <= 2:
            raise AnnotationFormatError("visibility must be in 0..2")

    def to_list(self) -> list[float | int]:
        self.validate()
        return [self.x, self.y, self.visibility]

    @classmethod
    def from_list(cls, raw: list[Any]) -> Keypoint:
        if len(raw) != 3:
            raise AnnotationFormatError("keypoint must be [x,y,v]")
        return cls(float(raw[0]), float(raw[1]), int(raw[2]))


@dataclass
class Instance:
    """
    annotation_data instance。
    Required: id / category_id. Optional: bbox / polygon / polyline / point / rle_mask / keypoints / text.
    Task-specific rules (e.g. detection requires bbox, polygon, polyline, or point) apply at read time.
    """

    id: int
    category_id: int
    bbox: Bbox | None = None
    polygon: list[float] | None = None
    polyline: list[float] | None = None
    point: list[float] | None = None
    keypoints: dict[str, list[Keypoint]] | None = None
    rle_mask: SegmentationRLE | None = None
    text: str | None = None
    is_ignored: bool = False

    def validate(self) -> None:
        if self.id < 0 or self.category_id < 0:
            raise AnnotationFormatError("id / category_id must be >= 0")
        if self.bbox is not None:
            self.bbox.validate()
        if self.polygon is not None:
            _validate_polygon_flat(self.polygon)
        if self.polyline is not None:
            _validate_polyline_flat(self.polyline)
        if self.point is not None:
            _validate_point_xy(self.point)
        if self.keypoints:
            for part, points in self.keypoints.items():
                if not part or not part[0].islower():
                    raise AnnotationFormatError(
                        f"keypoint part names should use snake_case: {part!r}"
                    )
                for kp in points:
                    kp.validate()
        if self.rle_mask is not None:
            self.rle_mask.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out: dict[str, Any] = {
            "id": int(self.id),
            "category_id": int(self.category_id),
        }
        if self.bbox is not None:
            out["bbox"] = self.bbox.to_list()
        if self.polygon is not None:
            out["polygon"] = list(self.polygon)
        if self.polyline is not None:
            out["polyline"] = list(self.polyline)
        if self.point is not None:
            out["point"] = list(self.point)
        if self.keypoints:
            out["keypoints"] = {
                part: [kp.to_list() for kp in points]
                for part, points in self.keypoints.items()
            }
        if self.rle_mask is not None:
            out["rle_mask"] = self.rle_mask.to_dict()
        if self.text is not None:
            out["text"] = self.text
        out["is_ignored"] = bool(self.is_ignored)
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Instance:
        kps = None
        if "keypoints" in raw:
            kps = {
                part: [Keypoint.from_list(p) for p in points]
                for part, points in raw["keypoints"].items()
            }
        bbox = Bbox.from_list(raw["bbox"]) if "bbox" in raw else None
        polygon = _polygon(raw.get("polygon"))
        polyline = _polyline(raw.get("polyline"))
        point = _point(raw.get("point"))
        rle = _rle_mask(_rle_mask_field(raw))
        return cls(
            id=int(raw["id"]),
            category_id=int(raw["category_id"]),
            bbox=bbox,
            polygon=polygon,
            polyline=polyline,
            point=point,
            keypoints=kps,
            rle_mask=rle,
            text=raw.get("text"),
            is_ignored=bool(raw.get("is_ignored", False)),
        )


@dataclass
class ClassificationHead:
    """One item in annotation_data categories[]."""

    category_attr: str
    category_ids: list[int]

    def validate(self) -> None:
        if not self.category_attr:
            raise AnnotationFormatError("category_attr must not be empty")
        if not self.category_ids:
            raise AnnotationFormatError("category_ids must not be empty")
        if len(set(self.category_ids)) != len(self.category_ids):
            raise AnnotationFormatError("category_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "category_attr": self.category_attr,
            "category_ids": [int(x) for x in self.category_ids],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ClassificationHead:
        return cls(
            category_attr=raw["category_attr"],
            category_ids=[int(x) for x in raw["category_ids"]],
        )


@dataclass
class Relationship:
    """One item in annotation_data relationships[]."""

    subject_id: int
    object_id: int
    relation_type: str

    def validate(self) -> None:
        if not self.relation_type:
            raise AnnotationFormatError("relation_type must not be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "subject_id": int(self.subject_id),
            "object_id": int(self.object_id),
            "relation_type": self.relation_type,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Relationship:
        return cls(
            subject_id=int(raw["subject_id"]),
            object_id=int(raw["object_id"]),
            relation_type=raw["relation_type"],
        )


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ContentImage:
    filename: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "image", "filename": self.filename}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ContentImage:
        return cls(filename=raw["filename"])


@dataclass
class ContentText:
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ContentText:
        return cls(text=raw["text"])


ContentPart = ContentImage | ContentText


@dataclass(init=False)
class ConversationTurn:
    """One item in annotation_data conversations[]."""

    role: ConversationRole
    content: list[ContentPart]

    def __init__(
        self,
        role: ConversationRole | str,
        image: str | ContentImage | None = None,
        text: str | ContentText | None = None,
        content: list[ContentPart] | None = None,
    ) -> None:
        self.role = ConversationRole(role)
        self.content = list(content) if content is not None else []
        if image is not None:
            self.content.append(
                image
                if isinstance(image, ContentImage)
                else ContentImage(filename=image)
            )
        if text is not None:
            self.content.append(
                text if isinstance(text, ContentText) else ContentText(text=text)
            )

    def validate(self) -> None:
        if not self.content:
            raise AnnotationFormatError("conversation content must not be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "role": self.role.value,
            "content": [p.to_dict() for p in self.content],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ConversationTurn:
        parts: list[ContentPart] = []
        for item in raw["content"]:
            if item["type"] == "image":
                parts.append(ContentImage.from_dict(item))
            elif item["type"] == "text":
                parts.append(ContentText.from_dict(item))
            else:
                raise AnnotationFormatError(f"Unknown content type: {item.get('type')}")
        return cls(role=ConversationRole(raw["role"]), content=parts)


@dataclass
class TrackItem:
    """annotation_data track_item。"""

    frame_idx: int
    bbox: Bbox
    rle_mask: SegmentationRLE | None = None

    def validate(self) -> None:
        if self.frame_idx < 0:
            raise AnnotationFormatError("frame_idx must be >= 0")
        self.bbox.validate()
        if self.rle_mask is not None:
            self.rle_mask.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out: dict[str, Any] = {
            "frame_idx": int(self.frame_idx),
            "bbox": self.bbox.to_list(),
        }
        if self.rle_mask is not None:
            out["rle_mask"] = self.rle_mask.to_dict()
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrackItem:
        rle = _rle_mask(_rle_mask_field(raw))
        return cls(
            frame_idx=int(raw["frame_idx"]),
            bbox=Bbox.from_list(raw["bbox"]),
            rle_mask=rle,
        )


@dataclass
class ActionEvent:
    """One item in annotation_data actions[]."""

    category_id: int
    track_id: int
    start_idx: int
    end_idx: int
    tracks: list[TrackItem] = field(default_factory=list)
    description: str | None = None

    def validate(self) -> None:
        if self.end_idx < self.start_idx:
            raise AnnotationFormatError("end_idx must be >= start_idx")
        if not self.tracks:
            raise AnnotationFormatError("tracks must not be empty")
        for tr in self.tracks:
            tr.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out: dict[str, Any] = {
            "category_id": int(self.category_id),
            "track_id": int(self.track_id),
            "start_idx": int(self.start_idx),
            "end_idx": int(self.end_idx),
            "tracks": [t.to_dict() for t in self.tracks],
        }
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ActionEvent:
        return cls(
            category_id=int(raw["category_id"]),
            track_id=int(raw["track_id"]),
            start_idx=int(raw["start_idx"]),
            end_idx=int(raw["end_idx"]),
            tracks=[TrackItem.from_dict(t) for t in raw["tracks"]],
            description=raw.get("description"),
        )


def _bbox(value: Any) -> Bbox:
    if isinstance(value, Bbox):
        return value
    if isinstance(value, dict):
        data = dict(value)
        bbox_format = data.get("format", "xyxy")
        raw = data.get("bbox", data.get("value"))
        if raw is None:
            raw = [data[k] for k in ("x1", "y1", "x2", "y2")]
        if bbox_format == "xyxy":
            return Bbox.from_xyxy(list(raw))
        if bbox_format == "xywh":
            return Bbox.from_xywh(list(raw))
        if bbox_format == "cxcywh":
            return Bbox.from_cxcywh(list(raw))
        raise AnnotationFormatError(f"Unknown bbox format: {bbox_format!r}")
    return Bbox.from_list(list(value))


def _rle_mask(value: Any) -> SegmentationRLE | None:
    if value is None:
        return None
    if isinstance(value, SegmentationRLE):
        return value
    if isinstance(value, dict):
        if "mask" in value:
            return SegmentationRLE.from_mask(value["mask"])
        if "size" in value and "counts" in value:
            return SegmentationRLE.from_dict(value)
    return SegmentationRLE.from_mask(value)


def _validate_polygon_flat(coords: list[float]) -> None:
    if len(coords) < 6 or len(coords) % 2 != 0:
        raise AnnotationFormatError(
            "polygon must be flat [x1,y1,x2,y2,...] with at least 3 vertices"
        )


def _validate_polyline_flat(coords: list[float]) -> None:
    if len(coords) < 4 or len(coords) % 2 != 0:
        raise AnnotationFormatError(
            "polyline must be flat [x1,y1,x2,y2,...] with at least 2 vertices"
        )


def _validate_point_xy(coords: list[float]) -> None:
    if len(coords) != 2:
        raise AnnotationFormatError("point must be [x, y]")


def _polygon(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise AnnotationFormatError("polygon must be a flat list of coordinates")
    coords = [float(v) for v in value]
    _validate_polygon_flat(coords)
    return coords


def _polyline(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise AnnotationFormatError("polyline must be a flat list of coordinates")
    coords = [float(v) for v in value]
    _validate_polyline_flat(coords)
    return coords


def _point(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise AnnotationFormatError("point must be a list [x, y]")
    coords = [float(v) for v in value]
    _validate_point_xy(coords)
    return coords


def _rle_mask_field(data: dict[str, Any]) -> Any:
    for key in ("rle_mask", "segmentation", "segmentation_mask", "mask"):
        if key in data:
            return data[key]
    return None


def _keypoint(value: Any) -> Keypoint:
    if isinstance(value, Keypoint):
        return value
    return Keypoint.from_list(list(value))


def _keypoints(value: Any) -> dict[str, list[Keypoint]] | None:
    if value is None:
        return None
    return {
        part: [_keypoint(point) for point in points]
        for part, points in dict(value).items()
    }


def _instance(raw: Any) -> Instance:
    if isinstance(raw, Instance):
        return raw
    data = dict(raw)
    bbox = _bbox(data["bbox"]) if "bbox" in data else None
    return Instance(
        id=int(data["id"]),
        category_id=int(data["category_id"]),
        bbox=bbox,
        polygon=_polygon(data.get("polygon")),
        polyline=_polyline(data.get("polyline")),
        point=_point(data.get("point")),
        keypoints=_keypoints(data.get("keypoints")),
        rle_mask=_rle_mask(_rle_mask_field(data)),
        text=data.get("text"),
        is_ignored=bool(data.get("is_ignored", False)),
    )


def _classification_head(raw: Any) -> ClassificationHead:
    if isinstance(raw, ClassificationHead):
        return raw
    data = dict(raw)
    return ClassificationHead(
        category_attr=data["category_attr"],
        category_ids=[int(x) for x in data["category_ids"]],
    )


def _relationship(raw: Any) -> Relationship:
    if isinstance(raw, Relationship):
        return raw
    data = dict(raw)
    return Relationship(
        subject_id=int(data["subject_id"]),
        object_id=int(data["object_id"]),
        relation_type=data["relation_type"],
    )


def _content_part(raw: Any) -> ContentPart:
    if isinstance(raw, (ContentImage, ContentText)):
        return raw
    data = dict(raw)
    part_type = data.get("type")
    if part_type == "image" or "filename" in data:
        return ContentImage(filename=data["filename"])
    if part_type == "text" or "text" in data:
        return ContentText(text=data["text"])
    raise AnnotationFormatError(f"Unknown content part: {raw!r}")


def _conversation_turn(raw: Any) -> ConversationTurn:
    if isinstance(raw, ConversationTurn):
        return raw
    data = dict(raw)
    if "content" not in data:
        return ConversationTurn(
            role=ConversationRole(data["role"]),
            image=data.get("image"),
            text=data.get("text"),
        )
    return ConversationTurn(
        role=ConversationRole(data["role"]),
        content=[_content_part(part) for part in data["content"]],
    )


def _track_item(raw: Any) -> TrackItem:
    if isinstance(raw, TrackItem):
        return raw
    data = dict(raw)
    return TrackItem(
        frame_idx=int(data["frame_idx"]),
        bbox=_bbox(data["bbox"]),
        rle_mask=_rle_mask(_rle_mask_field(data)),
    )


def _action_event(raw: Any) -> ActionEvent:
    if isinstance(raw, ActionEvent):
        return raw
    data = dict(raw)
    return ActionEvent(
        category_id=int(data["category_id"]),
        track_id=int(data["track_id"]),
        start_idx=int(data["start_idx"]),
        end_idx=int(data["end_idx"]),
        tracks=[_track_item(item) for item in data["tracks"]],
        description=data.get("description"),
    )
