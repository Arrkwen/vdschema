"""ImageNet-style folder layout → classification labels."""

from __future__ import annotations

from pathlib import Path

from vdschema import Name, TaskType

from ..messages import help_hint

DEFAULT_HEAD = "class"


def sorted_class_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet root not found: {root}")
    dirs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda item: item.name,
    )
    if not dirs:
        raise ValueError(
            f"no class subdirectories under {root}. {help_hint(TaskType.CLASSIFICATION)}"
        )
    return dirs


def build_imagenet_head(root: Path) -> tuple[dict[str, dict[int, Name]], dict[str, int]]:
    class_dirs = sorted_class_dirs(root)
    name_to_id = {path.name: idx + 1 for idx, path in enumerate(class_dirs)}
    head = {name_to_id[name]: Name(name) for name in name_to_id}
    return {DEFAULT_HEAD: head}, name_to_id
