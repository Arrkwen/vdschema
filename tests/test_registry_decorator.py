"""Extra registry registration coverage."""

from __future__ import annotations

from vdschema import TaskType
from vdswitch.converters.base import BaseConverter
from vdswitch.converters.registry import _CONVERTERS, register_converter_for_sources
from vdswitch.converters.sources import Source


def test_register_converter_for_sources_decorator() -> None:
    @register_converter_for_sources(task=TaskType.VLM, sources=(Source.MONOLITH,))
    class _VlmStubConverter(BaseConverter):
        def _convert(self) -> None:
            raise NotImplementedError

    assert (TaskType.VLM, Source.MONOLITH) in _CONVERTERS
    assert _VlmStubConverter.task_type is TaskType.VLM
