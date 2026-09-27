"""Converter registry and Source parsing."""

from __future__ import annotations

import pytest

from vdschema import TaskType
from vdswitch.converters.registry import (
    get_converter_class,
    parse_task,
    supported_sources,
)
from vdswitch.converters.sources import Source


def test_parse_task_invalid_string() -> None:
    with pytest.raises(ValueError, match="unsupported task"):
        parse_task("not_a_real_task")


def test_get_converter_invalid_source_for_task() -> None:
    with pytest.raises(ValueError, match="unsupported source"):
        get_converter_class(TaskType.DETECTION, Source.OCR)


def test_get_converter_no_registered_converters(monkeypatch) -> None:
    from vdswitch.converters import registry

    monkeypatch.setattr(registry, "_CONVERTERS", {})
    with pytest.raises(ValueError, match="unsupported task"):
        get_converter_class(TaskType.DETECTION, Source.MONOLITH)


def test_supported_sources_detection() -> None:
    sources = supported_sources(TaskType.DETECTION)
    assert Source.MONOLITH in sources
    assert Source.COCO in sources
