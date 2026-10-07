"""LabelBee General Data JSON → vdschema."""

from __future__ import annotations

from pathlib import Path

from vdschema import AnnotationWriter, TaskType

from ..options.presets import OPTIONAL_ROOT
from ..utils.labelbee_dataset import (
    classification_categories_from_document,
    collect_detection_attributes,
    collect_point_attributes,
    collect_polygon_attributes,
    collect_tag_heads,
    detection_instances_from_document,
    iter_labelbee_json_paths,
    keypoint_instances_from_document,
    label_dict_from_attr_map,
    load_labelbee_document,
    resolve_media_filename,
    segmentation_instances_from_document,
)
from .base import BaseConverter
from .registry import register_converter
from .sources import Source

_DOC_LINK = "https://github.com/open-mmlab/labelbee-client/tree/main/docs/annotation"


class _LabelBeeConverterBase(BaseConverter):
    source_note = f"LabelBee General Data JSON exports ({_DOC_LINK})."

    input_help = (
        "Directory tree of LabelBee `.json` files (one per image), or a single "
        "annotation JSON. Each file contains width, height, and `step_N` blocks "
        "with `toolName` / `result`."
    )
    category_help = (
        "Optional. Defaults to --input; category names are taken from each "
        "result's `attribute` (or tagTool option strings for classification)."
    )
    input_sample = (
        '{"width":640,"height":480,"step_1":{"toolName":"rectTool","result":[]}}'
    )
    category_sample = "(same as --input; attributes define label_dict)"
    typical_layout = (
        "  labelbee/json/**/*.json   — General Data export, one file per image\n"
        "  images/…                  — paired by filename stem (--option root=)"
    )
    converter_options = OPTIONAL_ROOT
    example_input = "/path/to/labelbee/json"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        same_dir = self.output_dir == self.input.parent
        stem = self.input.stem if self.input.is_file() else self.input.name
        self.output_data_filename = (
            f"{stem}_vdschema.jsonl" if same_dir else f"{stem}.jsonl"
        )
        self.output_meta_filename = "annotation_meta.json"

    def _ensure_inputs(self) -> None:
        if not self.input.exists():
            raise FileNotFoundError(f"input data not found: {self.input}")
        if self.input.is_file():
            if self.input.suffix.lower() != ".json":
                raise ValueError(
                    f"LabelBee --input must be a .json file or directory: {self.input}"
                )
        elif not self.input.is_dir():
            raise FileNotFoundError(f"input data not found: {self.input}")

    def _json_paths(self) -> list[Path]:
        paths = iter_labelbee_json_paths(self.input)
        if not paths:
            raise ValueError(f"no LabelBee JSON files under {self.input}")
        return paths


@register_converter(task=TaskType.DETECTION, source=Source.LABELBEE)
class LabelBeeDetectionConverter(_LabelBeeConverterBase):
    """rectTool / lineTool / polygonTool → vdschema detection."""

    def _convert(self) -> None:
        paths = self._json_paths()
        attr_to_id = collect_detection_attributes(paths)
        writer = AnnotationWriter(
            TaskType.DETECTION,
            label=label_dict_from_attr_map(attr_to_id),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for json_path in paths:
            doc = load_labelbee_document(json_path)
            instances = detection_instances_from_document(doc, attr_to_id=attr_to_id)
            if not instances:
                continue
            filename = resolve_media_filename(
                json_path,
                doc,
                input=self.input,
                root=self.root,
            )
            writer.append(
                filename=self.prefix_media_filename(filename),
                width=int(doc["width"]),
                height=int(doc["height"]),
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.SEGMENTATION, source=Source.LABELBEE)
class LabelBeeSegmentationConverter(_LabelBeeConverterBase):
    """polygonTool → vdschema segmentation (polygon)."""

    def _convert(self) -> None:
        paths = self._json_paths()
        attr_to_id = collect_polygon_attributes(paths)
        writer = AnnotationWriter(
            TaskType.SEGMENTATION,
            label=label_dict_from_attr_map(attr_to_id),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for json_path in paths:
            doc = load_labelbee_document(json_path)
            instances = segmentation_instances_from_document(doc, attr_to_id=attr_to_id)
            if not instances:
                continue
            filename = resolve_media_filename(
                json_path,
                doc,
                input=self.input,
                root=self.root,
            )
            writer.append(
                filename=self.prefix_media_filename(filename),
                width=int(doc["width"]),
                height=int(doc["height"]),
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.KEYPOINT, source=Source.LABELBEE)
class LabelBeeKeypointConverter(_LabelBeeConverterBase):
    """pointTool → vdschema keypoint."""

    def _convert(self) -> None:
        paths = self._json_paths()
        attr_to_id = collect_point_attributes(paths)
        writer = AnnotationWriter(
            TaskType.KEYPOINT,
            label=label_dict_from_attr_map(attr_to_id),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for json_path in paths:
            doc = load_labelbee_document(json_path)
            instances = keypoint_instances_from_document(doc, attr_to_id=attr_to_id)
            if not instances:
                continue
            filename = resolve_media_filename(
                json_path,
                doc,
                input=self.input,
                root=self.root,
            )
            writer.append(
                filename=self.prefix_media_filename(filename),
                width=int(doc["width"]),
                height=int(doc["height"]),
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.CLASSIFICATION, source=Source.LABELBEE)
class LabelBeeClassificationConverter(_LabelBeeConverterBase):
    """tagTool → vdschema classification."""

    def _convert(self) -> None:
        paths = self._json_paths()
        heads = collect_tag_heads(paths)
        if not heads:
            raise ValueError(
                f"no tagTool results found in LabelBee JSON under {self.input}"
            )
        writer = AnnotationWriter(
            TaskType.CLASSIFICATION,
            label=heads,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for json_path in paths:
            doc = load_labelbee_document(json_path)
            categories = classification_categories_from_document(doc, heads=heads)
            if not categories:
                continue
            filename = resolve_media_filename(
                json_path,
                doc,
                input=self.input,
                root=self.root,
            )
            writer.append(
                filename=self.prefix_media_filename(filename),
                width=int(doc["width"]),
                height=int(doc["height"]),
                categories=categories,
            )
        writer.save()
