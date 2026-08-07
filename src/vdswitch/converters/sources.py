"""Legacy source identifiers."""

from __future__ import annotations

from enum import Enum


class Source(str, Enum):
    MONOLITH = "monolith"
    UP = "up"

    @classmethod
    def parse(cls, value: str | Source) -> Source:
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown source={value!r}, allowed: {allowed}") from exc


# Backward-compatible alias.
LegacySource = Source
