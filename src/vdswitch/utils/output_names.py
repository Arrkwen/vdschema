"""Resolve vdswitch output filenames from input paths."""

from __future__ import annotations

from pathlib import Path

_VDSCHEMA_SUFFIX = "_vdschema"


def output_filename(source: Path, *, same_dir: bool) -> str:
    """Return output filename, optionally inserting ``_vdschema`` before the suffix."""
    if not same_dir:
        return source.name
    return f"{source.stem}{_VDSCHEMA_SUFFIX}{source.suffix}"


def resolve_output_filenames(
    *,
    input: Path,
    category: Path,
    output_dir: Path,
) -> tuple[str, str]:
    """Map legacy input files to vdschema output filenames under ``output_dir``."""
    input = Path(input).expanduser().resolve()
    category = Path(category).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    return (
        output_filename(input, same_dir=output_dir == input.parent),
        output_filename(category, same_dir=output_dir == category.parent),
    )
