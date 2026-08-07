"""Resolve legacy dataset media directories."""

from __future__ import annotations

from pathlib import Path

_MEDIA_ROOT_HINT = (
    "pass --root as the dataset root; it is joined with media paths in "
    "annotation files to resolve absolute image/video paths"
)


def _normalize_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"--root not found: {path}")
    return path


def _pick_existing_dir(
    candidates: tuple[Path, ...],
    *,
    kind: str,
    root: Path | None,
    allow_empty: bool = False,
) -> Path:
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path
    if allow_empty:
        for path in candidates:
            if path.is_dir():
                return path
    if root is not None:
        raise FileNotFoundError(f"no {kind} directory under --root={root}")
    raise FileNotFoundError(f"could not locate {kind} root; {_MEDIA_ROOT_HINT}")


def find_image_root(
    *,
    output_dir: Path,
    input_data: Path,
    root: str | Path | None = None,
) -> Path:
    """Locate the directory used to join annotation media paths into absolute image paths."""
    if root is not None:
        dataset_root = _normalize_root(root)
        return _pick_existing_dir(
            (dataset_root / "images", dataset_root),
            kind="image",
            root=dataset_root,
            allow_empty=True,
        )

    source_root = input_data.parent.parent
    return _pick_existing_dir(
        (
            output_dir / "images",
            source_root / "images",
            source_root,
            input_data.parent / "images",
            output_dir.parent / "images",
        ),
        kind="image",
        root=None,
    )


def find_video_root(
    *,
    output_dir: Path,
    input_data: Path,
    root: str | Path | None = None,
) -> Path:
    """Locate the directory used to join annotation media paths into absolute video paths."""
    if root is not None:
        dataset_root = _normalize_root(root)
        return _pick_existing_dir(
            (dataset_root / "video", dataset_root),
            kind="video",
            root=dataset_root,
            allow_empty=True,
        )

    source_root = input_data.parent.parent
    return _pick_existing_dir(
        (
            output_dir / "video",
            source_root / "video",
            source_root,
            output_dir,
            output_dir.parent / "video",
        ),
        kind="video",
        root=None,
    )
