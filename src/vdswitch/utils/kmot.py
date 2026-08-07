"""Parse legacy UP/monolith kmot track files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KmotFrame:
    frame_idx: int
    bbox_xyxy: list[float]
    category_raw: Any


@dataclass
class KmotTrack:
    track_id: int
    frames: list[KmotFrame] = field(default_factory=list)
    category_id: int | None = None

    @property
    def start_idx(self) -> int:
        if not self.frames:
            return 0
        return min(item.frame_idx for item in self.frames)

    @property
    def end_idx(self) -> int:
        if not self.frames:
            return 0
        return max(item.frame_idx for item in self.frames)

    def to_action_dict(self) -> dict[str, Any]:
        if not self.frames or self.category_id is None:
            raise ValueError(f"track {self.track_id} has no frames or category")
        tracks = [
            {
                "frame_idx": item.frame_idx,
                "bbox": item.bbox_xyxy,
            }
            for item in sorted(self.frames, key=lambda row: row.frame_idx)
        ]
        return {
            "category_id": self.category_id,
            "track_id": self.track_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "tracks": tracks,
        }


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> list[float]:
    return [x, y, x + w, y + h]


def _parse_kmot_line(parts: list[str]) -> tuple[int, int, list[float], Any] | None:
    if len(parts) < 7:
        return None
    frame_idx = int(float(parts[0]))
    track_id = int(float(parts[1]))
    x = float(parts[2])
    y = float(parts[3])
    w = float(parts[4])
    h = float(parts[5])
    if w <= 0 or h <= 0:
        return None
    category_raw = parts[-1]
    return frame_idx, track_id, _xywh_to_xyxy(x, y, w, h), category_raw


def _resolve_category_id(raw: Any, name_to_id: dict[str, int]) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"invalid action category: {raw!r}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    text = str(raw).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    try:
        return name_to_id[text]
    except KeyError as exc:
        allowed = ", ".join(sorted(name_to_id))
        raise ValueError(f"unknown action category={text!r}, allowed: {allowed}") from exc


def parse_kmot_file(
    kmot_path: Path,
    *,
    name_to_id: dict[str, int],
) -> dict[int, KmotTrack]:
    """Parse kmot txt into track_id -> :class:`KmotTrack`."""
    grouped: dict[int, KmotTrack] = {}
    category_votes: dict[int, Counter[int]] = defaultdict(Counter)

    with kmot_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = _parse_kmot_line(line.split(","))
            if parsed is None:
                continue
            frame_idx, track_id, bbox_xyxy, category_raw = parsed
            track = grouped.get(track_id)
            if track is None:
                track = KmotTrack(track_id=track_id)
                grouped[track_id] = track
            track.frames.append(
                KmotFrame(
                    frame_idx=frame_idx,
                    bbox_xyxy=bbox_xyxy,
                    category_raw=category_raw,
                )
            )
            category_votes[track_id][_resolve_category_id(category_raw, name_to_id)] += 1

    for track_id, track in grouped.items():
        track.category_id = category_votes[track_id].most_common(1)[0][0]
    return grouped
