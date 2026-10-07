"""Per-converter ``--option`` specs and resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .category_id import ConvertOptions
from .global_options import GlobalOptions

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
    global_options: GlobalOptions = GlobalOptions()

    @property
    def prefix(self) -> str | None:
        return self.global_options.prefix


def _effective_options(cls: type[BaseConverter]) -> tuple[ConverterOptionSpec, ...]:
    return cls.converter_options


def _spec_map(cls: type[BaseConverter]) -> dict[str, ConverterOptionSpec]:
    return {item.key: item for item in _effective_options(cls)}


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
    for spec in _effective_options(cls):
        if spec.required and spec.key not in raw:
            raise ValueError(
                f"missing required --option {spec.key}=… for --source "
                f"{cls.source.value!r}; see: vdswitch help --task "
                f"{cls.task_type.value} --source {cls.source.value}"
            )

    if "category" in specs and "category" in raw:
        category = Path(raw.pop("category")).expanduser()
    else:
        category = input

    if "root" in specs and "root" in raw:
        root: Path | None = Path(raw.pop("root")).expanduser()
    else:
        root = None

    global_options, raw = GlobalOptions.consume(raw)

    convert_options = ConvertOptions.from_mapping(raw)

    return ResolvedConverterInputs(
        category=category.resolve(),
        root=root.resolve() if root is not None else None,
        convert_options=convert_options,
        global_options=global_options,
    )


def partition_converter_options(
    cls: type[BaseConverter],
) -> tuple[tuple[ConverterOptionSpec, ...], tuple[ConverterOptionSpec, ...]]:
    """Required ``--option`` specs first, then optional (including globals)."""
    required: list[ConverterOptionSpec] = []
    optional: list[ConverterOptionSpec] = []
    for spec in _effective_options(cls):
        if spec.required:
            required.append(spec)
        else:
            optional.append(spec)
    return tuple(required), tuple(optional)


def format_option_spec_help_lines(spec: ConverterOptionSpec) -> list[str]:
    """Two-line block: parameter name, then indented description."""
    global_keys = GlobalOptions.field_names()
    if spec.key in global_keys:
        lines = [
            f"  {spec.key}=VALUE",
            f"      {spec.help} Omit to leave filenames unchanged.",
        ]
    else:
        lines = [f"  {spec.key}=VALUE", f"      {spec.help}"]
    if spec.example:
        lines.append(f"      Example value: {spec.example}")
    return lines


def example_value_for_option(
    spec: ConverterOptionSpec,
    *,
    category_path: str | None,
    defaults: ConvertOptions | None = None,
) -> str | None:
    """Sample VALUE for help examples; ``None`` only when no sensible placeholder."""
    defaults = defaults or ConvertOptions()
    global_keys = GlobalOptions.field_names()
    if spec.key in global_keys:
        return spec.example or "split/v1"
    if spec.key == "category_id_contiguous":
        return "1" if defaults.category_id_contiguous else "0"
    if spec.key == "category_id_start":
        return str(defaults.category_id_start)
    if spec.key == "category":
        return spec.example or category_path
    if spec.key == "root":
        return spec.example or "/path/to/dataset"
    return spec.example


def example_option_command_lines(
    cls: type[BaseConverter],
    *,
    input_path: str,
    category_path: str | None,
) -> list[str]:
    defaults = ConvertOptions()
    lines: list[str] = []
    required, optional = partition_converter_options(cls)
    for spec in (*required, *optional):
        value = example_value_for_option(
            spec, category_path=category_path, defaults=defaults
        )
        if value is None:
            continue
        lines.append(f"  --option {spec.key}={value} \\")
    return lines


def format_option_help_lines(cls: type[BaseConverter]) -> list[str]:
    """Flat option help blocks (required, then optional)."""
    required, optional = partition_converter_options(cls)
    lines: list[str] = []
    for spec in (*required, *optional):
        lines.extend(format_option_spec_help_lines(spec))
    return lines
