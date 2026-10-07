"""Base converter for legacy → vdschema."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from vdschema import AnnotationReader, Name, TaskType

from ..options import ConvertOptions, map_category_id, prepare_category_label
from ..options.presets import GLOBAL_ONLY_OPTIONS
from ..options.spec import ConverterOptionSpec
from ..utils.image_size import ImageSizeResolver, find_image_root
from ..utils.output_names import resolve_output_filenames
from .sources import Source


class BaseConverter(ABC):
    """Convert one legacy source layout into a vdschema task directory."""

    task_type: ClassVar[TaskType]
    source: ClassVar[Source]

    converter_options: ClassVar[tuple[ConverterOptionSpec, ...]] = GLOBAL_ONLY_OPTIONS

    source_note: ClassVar[str] = "Monolith legacy on-disk layout (JSONL / meta / kmot)."
    input_help: ClassVar[str] = ""
    input_sample: ClassVar[str] = ""
    typical_layout: ClassVar[str] = ""
    example_input: ClassVar[str] = "/path/to/data"
    example_category: ClassVar[str | None] = None
    input_is_dir: ClassVar[bool] = False

    def __init__(
        self,
        *,
        input: str | Path,
        category: str | Path,
        output_dir: str | Path,
        root: str | Path | None = None,
        convert_options: ConvertOptions | None = None,
        prefix: str | None = None,
    ) -> None:
        self.convert_options = convert_options or ConvertOptions()
        self.input = Path(input).expanduser().resolve()
        self.category = Path(category).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.prefix = prefix
        self.output_data_filename, self.output_meta_filename = resolve_output_filenames(
            input=self.input,
            category=self.category,
            output_dir=self.output_dir,
        )

    @classmethod
    def category_defaults_to_input(cls) -> bool:
        return not any(spec.key == "category" for spec in cls.converter_options)

    @cached_property
    def image_size_resolver(self) -> ImageSizeResolver:
        image_root = find_image_root(
            output_dir=self.output_dir,
            input=self.input,
            root=self.root,
        )
        return ImageSizeResolver(image_root)

    def prefix_media_filename(self, filename: str) -> str:
        """Prepend ``prefix`` to annotation ``filename`` fields when set."""
        if not self.prefix:
            return filename
        return os.path.join(self.prefix, filename)

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
        if type(self).input_is_dir:
            if not self.input.is_dir():
                raise FileNotFoundError(f"input not found: {self.input}")
        elif not self.input.is_file():
            raise FileNotFoundError(f"input not found: {self.input}")
        if type(self).category_defaults_to_input():
            return
        if self.category.resolve() == self.input.resolve():
            return
        if not self.category.is_file():
            raise FileNotFoundError(f"category file not found: {self.category}")

    def _prepare_category_label(
        self, label: dict[int, Name]
    ) -> tuple[dict[int, Name], dict[int, int] | None]:
        return prepare_category_label(label, self.convert_options)

    @staticmethod
    def _map_category_id(raw_id: int, id_map: dict[int, int] | None) -> int:
        return map_category_id(raw_id, id_map)

    @abstractmethod
    def _convert(self) -> None:
        """Write ``annotation_data.jsonl`` and label meta under ``output_dir``."""
