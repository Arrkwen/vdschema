"""LabelMe JSON → vdschema."""

from __future__ import annotations

from pathlib import Path

from vdschema import AnnotationWriter, TaskType

from ..utils.labelme_dataset import (
    DETECTION_SHAPE_TYPES,
    KEYPOINT_SHAPE_TYPES,
    SEGMENTATION_SHAPE_TYPES,
    collect_label_ids,
    detection_instances_from_document,
    document_size,
    iter_labelme_json_paths,
    keypoint_instances_from_document,
    label_dict_from_label_map,
    load_labelme_document,
    resolve_media_filename,
    segmentation_instances_from_document,
)
from .base import BaseConverter
from .registry import register_converter
from .sources import Source

_DOC_LINK = "https://github.com/wkentaro/labelme"


class _LabelMeConverterBase(BaseConverter):
    source_note = f"LabelMe per-image JSON (`shapes[]`, `imagePath`) — {_DOC_LINK}."

    input_data_help = (
        "Directory of LabelMe `.json` files (typically one JSON per image), or a "
        "single annotation JSON with `shapes`, `imagePath`, `imageWidth`, "
        "`imageHeight`."
    )
    input_label_help = (
        "Optional. Defaults to --input-data; class names come from each shape's "
        "`label` field."
    )
    input_data_sample = (
        '{"imagePath":"img.jpg","imageWidth":640,"imageHeight":480,'
        '"shapes":[{"label":"cat","shape_type":"rectangle","points":[[0,0],[10,10]]}]}'
    )
    input_label_sample = "(same as --input-data; shape labels define annotation_meta)"
    typical_layout = (
        "  annotations/*.json   — LabelMe export (paired with images)\n"
        "  images/…           — paths in imagePath (--input-root when relative)"
    )
    input_label_same_as_data = True
    input_root_optional = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        same_dir = self.output_dir == self.input_data.parent
        stem = self.input_data.stem if self.input_data.is_file() else self.input_data.name
        self.output_data_filename = (
            f"{stem}_vdschema.jsonl" if same_dir else f"{stem}.jsonl"
        )
        self.output_meta_filename = "annotation_meta.json"

    def _ensure_inputs(self) -> None:
        if not self.input_data.exists():
            raise FileNotFoundError(f"input data not found: {self.input_data}")
        if self.input_data.is_file():
            if self.input_data.suffix.lower() != ".json":
                raise ValueError(
                    f"LabelMe --input-data must be a .json file or directory: "
                    f"{self.input_data}"
                )
        elif not self.input_data.is_dir():
            raise FileNotFoundError(f"input data not found: {self.input_data}")

    def _json_paths(self) -> list[Path]:
        paths = iter_labelme_json_paths(self.input_data)
        if not paths:
            raise ValueError(f"no LabelMe JSON files under {self.input_data}")
        return paths


@register_converter(task=TaskType.DETECTION, source=Source.LABELME)
class LabelMeDetectionConverter(_LabelMeConverterBase):
    """rectangle / polygon / line / point / circle → vdschema detection."""

    def _convert(self) -> None:
        paths = self._json_paths()
        label_to_id = collect_label_ids(paths, DETECTION_SHAPE_TYPES)
        writer = AnnotationWriter(
            TaskType.DETECTION,
            label=label_dict_from_label_map(label_to_id),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for json_path in paths:
            doc = load_labelme_document(json_path)
            instances = detection_instances_from_document(
                doc, label_to_id=label_to_id
            )
            if not instances:
                continue
            width, height = document_size(doc)
            filename = resolve_media_filename(
                json_path,
                doc,
                input_data=self.input_data,
                input_root=self.input_root,
            )
            writer.append(
                filename=filename,
                width=width,
                height=height,
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.SEGMENTATION, source=Source.LABELME)
class LabelMeSegmentationConverter(_LabelMeConverterBase):
    """polygon shapes → vdschema segmentation."""

    def _convert(self) -> None:
        paths = self._json_paths()
        label_to_id = collect_label_ids(paths, SEGMENTATION_SHAPE_TYPES)
        writer = AnnotationWriter(
            TaskType.SEGMENTATION,
            label=label_dict_from_label_map(label_to_id),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for json_path in paths:
            doc = load_labelme_document(json_path)
            instances = segmentation_instances_from_document(
                doc, label_to_id=label_to_id
            )
            if not instances:
                continue
            width, height = document_size(doc)
            filename = resolve_media_filename(
                json_path,
                doc,
                input_data=self.input_data,
                input_root=self.input_root,
            )
            writer.append(
                filename=filename,
                width=width,
                height=height,
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.KEYPOINT, source=Source.LABELME)
class LabelMeKeypointConverter(_LabelMeConverterBase):
    """point shapes → vdschema keypoint."""

    def _convert(self) -> None:
        paths = self._json_paths()
        label_to_id = collect_label_ids(paths, KEYPOINT_SHAPE_TYPES)
        writer = AnnotationWriter(
            TaskType.KEYPOINT,
            label=label_dict_from_label_map(label_to_id),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for json_path in paths:
            doc = load_labelme_document(json_path)
            instances = keypoint_instances_from_document(
                doc, label_to_id=label_to_id
            )
            if not instances:
                continue
            width, height = document_size(doc)
            filename = resolve_media_filename(
                json_path,
                doc,
                input_data=self.input_data,
                input_root=self.input_root,
            )
            writer.append(
                filename=filename,
                width=width,
                height=height,
                instances=instances,
            )
        writer.save()
