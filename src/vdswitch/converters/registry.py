"""Converter registry keyed by (TaskType, Source)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from vdschema import TaskType

from .base import BaseConverter
from .sources import Source

_CONVERTERS: dict[tuple[TaskType, Source], type[BaseConverter]] = {}
ConverterT = TypeVar("ConverterT", bound=type[BaseConverter])


def register_converter(*, task: TaskType, source: Source) -> Callable[[ConverterT], ConverterT]:
    """Register a converter class for one task/source pair."""

    def decorator(cls: ConverterT) -> ConverterT:
        cls.task_type = task
        cls.source = source
        _CONVERTERS[(task, source)] = cls
        return cls

    return decorator


def register_converter_for_sources(
    *, task: TaskType, sources: Iterable[Source]
) -> Callable[[ConverterT], ConverterT]:
    """Register the same converter class for multiple sources of one task."""

    source_list = tuple(sources)

    def decorator(cls: ConverterT) -> ConverterT:
        cls.task_type = task
        for source in source_list:
            _CONVERTERS[(task, source)] = cls
        return cls

    return decorator


def get_converter_class(task: TaskType, source: Source) -> type[BaseConverter]:
    try:
        return _CONVERTERS[(task, source)]
    except KeyError as exc:
        task_sources = supported_sources(task)
        if task_sources:
            allowed = ", ".join(item.value for item in sorted(task_sources, key=lambda s: s.value))
            raise ValueError(
                f"unsupported source={source.value!r} for task={task.value!r}, "
                f"allowed: {allowed}"
            ) from exc
        allowed = ", ".join(item.value for item in sorted(supported_tasks(), key=lambda t: t.value))
        raise ValueError(f"unsupported task={task.value!r}, allowed: {allowed}") from exc


def supported_tasks() -> frozenset[TaskType]:
    return frozenset(task for task, _ in _CONVERTERS)


def supported_sources(task: TaskType | None = None) -> frozenset[Source]:
    if task is None:
        return frozenset(source for _, source in _CONVERTERS)
    return frozenset(source for task_type, source in _CONVERTERS if task_type is task)


def parse_task(task: str | TaskType) -> TaskType:
    """Parse CLI/API task into a registered :class:`TaskType`."""
    if isinstance(task, TaskType):
        task_type = task
    else:
        try:
            task_type = TaskType(task.lower())
        except ValueError as exc:
            allowed = ", ".join(
                item.value for item in sorted(supported_tasks(), key=lambda t: t.value)
            )
            raise ValueError(f"unsupported task={task!r}, allowed: {allowed}") from exc

    if task_type not in supported_tasks():
        allowed = ", ".join(
            item.value for item in sorted(supported_tasks(), key=lambda t: t.value)
        )
        raise ValueError(f"unsupported task={task!r}, allowed: {allowed}")
    return task_type
