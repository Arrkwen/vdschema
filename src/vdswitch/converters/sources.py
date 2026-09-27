"""Legacy source identifiers."""

from __future__ import annotations

from enum import Enum


class Source(str, Enum):
    UP = "up"
    COCO = "coco"
    YOLO = "yolo"
    IMAGENET = "imagenet"

    @classmethod
    def parse(cls, value: str | Source) -> Source:
        if isinstance(value, cls):
            return value
        normalized = value.lower()
        if normalized == "monolith":
            raise ValueError(
                "source 'monolith' is no longer supported; use 'up' (same layout)"
            ) from None
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown source={value!r}, allowed: {allowed}") from exc


# Backward-compatible alias.
LegacySource = Source
