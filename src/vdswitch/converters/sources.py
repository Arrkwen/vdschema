"""Legacy source identifiers."""

from __future__ import annotations

from enum import Enum


class Source(str, Enum):
    MONOLITH = "monolith"
    COCO = "coco"
    YOLO = "yolo"
    IMAGENET = "imagenet"
    OCR = "ocr"

    @classmethod
    def parse(cls, value: str | Source) -> Source:
        if isinstance(value, cls):
            return value
        normalized = value.lower()
        if normalized == "up":
            return cls.MONOLITH
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown source={value!r}, allowed: {allowed}") from exc


# Backward-compatible alias.
LegacySource = Source
