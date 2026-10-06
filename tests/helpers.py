"""Test helpers."""

from __future__ import annotations

from pathlib import Path


def vdswitch_options(
    *,
    category: str | Path | None = None,
    root: str | Path | None = None,
    **extra: str,
) -> dict[str, str]:
    opts: dict[str, str] = {key: str(value) for key, value in extra.items()}
    if category is not None:
        opts["category"] = str(category)
    if root is not None:
        opts["root"] = str(root)
    return opts
