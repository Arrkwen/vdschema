"""Per-converter ``--option`` specs and resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .category_id import ConvertOptions

if TYPE_CHECKING:
    from vdswitch.converters.base import BaseConverter


@dataclass(frozen=True)
class ConverterOptionSpec:
    """One ``--option KEY=VALUE`` supported by a converter class."""

    key: str
    help: str
    required: bool = False
    example: str | None = None


@dataclass(frozen=True)
class ResolvedConverterInputs:
    category: Path
    root: Path | None
    convert_options: ConvertOptions


def _spec_map(cls: type[BaseConverter]) -> dict[str, ConverterOptionSpec]:
    return {item.key: item for item in cls.converter_options}


def resolve_converter_inputs(
    cls: type[BaseConverter],
    *,
    input: Path,
    options: Mapping[str, str] | None,
) -> ResolvedConverterInputs:
    specs = _spec_map(cls)
    raw = dict(options or {})
    for key in raw:
        if key not in specs:
            allowed = ", ".join(sorted(specs)) or "(none)"
            raise ValueError(
                f"unknown option {key!r} for --source {cls.source.value!r} "
                f"task={cls.task_type.value!r}; allowed: {allowed}"
            )
    for spec in cls.converter_options:
        if spec.required and spec.key not in raw:
            raise ValueError(
                f"missing required --option {spec.key}=… for --source "
                f"{cls.source.value!r}; see: vdswitch help --task "
                f"{cls.task_type.value} --source {cls.source.value}"
            )

    if "category" in specs:
        if "category" in raw:
            category = Path(raw.pop("category")).expanduser()
        elif specs["category"].required:
            raise ValueError("missing required --option category=…")
        else:
            category = input
    else:
        category = input

    if "root" in specs:
        if "root" in raw:
            root: Path | None = Path(raw.pop("root")).expanduser()
        elif specs["root"].required:
            raise ValueError("missing required --option root=…")
        else:
            root = None
    else:
        root = None

    convert_options = ConvertOptions.from_mapping(raw)

    return ResolvedConverterInputs(
        category=category.resolve(),
        root=root.resolve() if root is not None else None,
        convert_options=convert_options,
    )


def format_option_help_lines(cls: type[BaseConverter]) -> list[str]:
    return [f"  {spec.key}=… — {spec.help}" for spec in cls.converter_options]


def example_option_command_lines(
    cls: type[BaseConverter],
    *,
    input_path: str,
    category_path: str | None,
) -> list[str]:
    defaults = ConvertOptions()
    lines: list[str] = []
    for spec in cls.converter_options:
        if spec.key == "category_id_contiguous":
            value = "1" if defaults.category_id_contiguous else "0"
        elif spec.key == "category_id_start":
            value = str(defaults.category_id_start)
        elif spec.key == "category":
            value = spec.example or category_path
            if value is None:
                continue
        elif spec.key == "root":
            value = spec.example
            if value is None and not spec.required:
                continue
            value = value or "/path/to/dataset"
        else:
            value = spec.example
            if value is None:
                if spec.required:
                    continue
                continue
        lines.append(f"  --option {spec.key}={value} \\")
    return lines
