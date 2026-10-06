"""CLI ``--option`` parsing, specs, and category-id helpers."""

from .category_id import (
    ConvertOptions,
    densify_label_map,
    map_category_id,
    parse_option_pairs,
    prepare_category_label,
)
from .spec import (
    ConverterOptionSpec,
    ResolvedConverterInputs,
    example_option_command_lines,
    format_option_help_lines,
    resolve_converter_inputs,
)

__all__ = [
    "ConvertOptions",
    "ConverterOptionSpec",
    "ResolvedConverterInputs",
    "densify_label_map",
    "example_option_command_lines",
    "format_option_help_lines",
    "map_category_id",
    "parse_option_pairs",
    "prepare_category_label",
    "resolve_converter_inputs",
]
