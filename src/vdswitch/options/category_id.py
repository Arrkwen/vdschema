"""Category-id tuning options for converters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vdschema import Name


def parse_option_pairs(pairs: list[str] | None) -> dict[str, str]:
    """Parse repeatable ``KEY=VALUE`` tokens (keys validated per converter)."""
    if not pairs:
        return {}
    options: dict[str, str] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"expected KEY=VALUE, got {raw!r}")
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"expected KEY=VALUE, got {raw!r}")
        if not value:
            raise ValueError(f"option {key!r} requires a non-empty value")
        options[key] = value
    return options


def _parse_bool(value: str, *, key: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be 0 or 1 (or true/false), got {value!r}")


def _parse_start(value: str) -> int:
    if value not in {"0", "1"}:
        raise ValueError(f"category_id_start must be 0 or 1, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class ConvertOptions:
    category_id_contiguous: bool = False
    category_id_start: int = 1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, str] | None) -> ConvertOptions:
        if not raw:
            return cls()
        contiguous = False
        start = 1
        if "category_id_contiguous" in raw:
            contiguous = _parse_bool(
                raw["category_id_contiguous"], key="category_id_contiguous"
            )
        if "category_id_start" in raw:
            start = _parse_start(raw["category_id_start"])
        if start not in (0, 1):
            raise ValueError("category_id_start must be 0 or 1")
        if (
            not contiguous
            and "category_id_start" in raw
            and raw.get("category_id_start") not in (None, "1")
        ):
            pass  # ignore start when not contiguous
        return cls(category_id_contiguous=contiguous, category_id_start=start)


def densify_label_map(
    label: Mapping[int, Name],
    *,
    start: int,
) -> tuple[dict[int, Name], dict[int, int]]:
    if start not in (0, 1):
        raise ValueError("category_id_start must be 0 or 1")
    if not label:
        raise ValueError("cannot densify empty label map")
    old_ids = sorted(label.keys())
    id_map = {old_id: start + index for index, old_id in enumerate(old_ids)}
    new_label = {id_map[old_id]: label[old_id] for old_id in old_ids}
    return new_label, id_map


def prepare_category_label(
    label: dict[int, Name],
    options: ConvertOptions,
) -> tuple[dict[int, Name], dict[int, int] | None]:
    if not options.category_id_contiguous:
        return label, None
    new_label, id_map = densify_label_map(label, start=options.category_id_start)
    return new_label, id_map


def map_category_id(raw_id: int, id_map: dict[int, int] | None) -> int:
    if id_map is None:
        return raw_id
    if raw_id not in id_map:
        raise ValueError(
            f"category_id={raw_id} is not in the source vocabulary; "
            f"known ids: {sorted(id_map.keys())}"
        )
    return id_map[raw_id]
