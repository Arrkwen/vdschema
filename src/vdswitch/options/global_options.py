"""``--option`` keys shared by every converter (merged into each class's option list)."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class GlobalOptions:
    """Runtime values for global ``--option`` keys (``prefix``; see ``presets._GLOBAL``)."""

    prefix: str | None = None

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(f.name for f in fields(cls))

    @classmethod
    def consume(cls, raw: dict[str, str]) -> tuple[GlobalOptions, dict[str, str]]:
        """Pop global keys from ``raw`` and return ``(global, remaining)``."""
        remaining = dict(raw)
        kwargs: dict[str, str] = {}
        for name in cls.field_names():
            if name not in remaining:
                continue
            value = remaining.pop(name).strip()
            if not value:
                raise ValueError(f"option {name!r} requires a non-empty value")
            kwargs[name] = value
        return cls(**kwargs), remaining
