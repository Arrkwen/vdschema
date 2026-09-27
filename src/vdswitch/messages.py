"""Shared user-facing CLI messages."""

from __future__ import annotations

from vdschema import TaskType


def help_hint(task: TaskType) -> str:
    return f"Run: vdswitch help --task {task.value}"
