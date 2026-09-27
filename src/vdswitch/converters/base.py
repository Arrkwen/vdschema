"""Base converter for legacy → vdschema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from vdschema import AnnotationReader, TaskType

from ..utils.image_size import ImageSizeResolver, find_image_root
from ..utils.output_names import resolve_output_filenames
from .sources import Source


class BaseConverter(ABC):
    """Convert one legacy source layout into a vdschema task directory."""

    task_type: ClassVar[TaskType]
    source: ClassVar[Source]

    #: Shown by ``vdswitch help``.
    source_note: ClassVar[str] = "Monolith legacy on-disk layout (JSONL / meta / kmot)."
    input_data_help: ClassVar[str] = ""
    input_label_help: ClassVar[str] = ""
    input_data_sample: ClassVar[str] = ""
    input_label_sample: ClassVar[str] = ""
    typical_layout: ClassVar[str] = ""

    def __init__(
        self,
        *,
        input_data: str | Path,
        input_label: str | Path,
        output_dir: str | Path,
        input_root: str | Path | None = None,
    ) -> None:
        self.input_data = Path(input_data).expanduser().resolve()
        self.input_label = Path(input_label).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.input_root = (
            Path(input_root).expanduser().resolve() if input_root is not None else None
        )
        self.output_data_filename, self.output_meta_filename = resolve_output_filenames(
            input_data=self.input_data,
            input_label=self.input_label,
            output_dir=self.output_dir,
        )

    @cached_property
    def image_size_resolver(self) -> ImageSizeResolver:
        image_root = find_image_root(
            output_dir=self.output_dir,
            input_data=self.input_data,
            root=self.input_root,
        )
        return ImageSizeResolver(image_root)

    def run(self) -> Path:
        """Validate inputs, convert, and verify vdschema output."""
        self._ensure_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._convert()
        AnnotationReader(
            self.task_type,
            self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        ).load()
        return self.output_dir

    def _ensure_inputs(self) -> None:
        if not self.input_data.is_file():
            raise FileNotFoundError(f"input data not found: {self.input_data}")
        if not self.input_label.is_file():
            raise FileNotFoundError(f"input label not found: {self.input_label}")

    @abstractmethod
    def _convert(self) -> None:
        """Write ``annotation_data.jsonl`` and label meta under ``output_dir``."""
