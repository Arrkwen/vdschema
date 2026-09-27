"""JSONL line parsing helpers for vdswitch converters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vdschema import AnnotationFormatError


def load_jsonl_object(line: str, *, path: Path, lineno: int) -> dict[str, Any]:
    """Parse one JSONL line into an object; raise ``AnnotationFormatError`` on failure."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AnnotationFormatError(f"{path}:{lineno} failed to parse JSON") from exc
    if not isinstance(record, dict):
        raise AnnotationFormatError(f"{path}:{lineno} JSONL record must be an object")
    return record
