"""Parse OCR manifest files."""

from __future__ import annotations

from pathlib import Path

from vdschema import TaskType

from ..help import help_hint


def _split_manifest_line(line: str) -> tuple[str, str]:
    if "\t" in line:
        path_part, label = line.split("\t", 1)
        return path_part.strip(), label.strip()
    parts = line.split(maxsplit=1)
    if len(parts) < 2:
        raise ValueError(f"manifest line must contain path and label: {line!r}")
    return parts[0].strip(), parts[1].strip()


def tokenize_label(text: str) -> list[str]:
    return list(text.strip())


def iter_ocr_lines(
    manifest: Path, *, root: Path | None
) -> list[tuple[Path, str, list[str]]]:
    rows: list[tuple[Path, str, list[str]]] = []
    with manifest.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rel_path, label = _split_manifest_line(line)
            except ValueError as exc:
                raise ValueError(f"{manifest}:{lineno}: {exc}") from exc
            if not label:
                raise ValueError(
                    f"{manifest}:{lineno}: empty label. {help_hint(TaskType.SEQUENCE)}"
                )
            path = Path(rel_path)
            if not path.is_absolute():
                resolved = (
                    (root / path).resolve()
                    if root is not None
                    else (manifest.parent / path).resolve()
                )
            else:
                resolved = path.resolve()
            rows.append((resolved, Path(rel_path).as_posix(), tokenize_label(label)))
    if not rows:
        raise ValueError(f"no OCR lines in manifest: {manifest}")
    return rows
