"""MS COCO JSON → vdschema (detection / keypoint / segmentation)."""

from __future__ import annotations

from vdschema import AnnotationWriter, TaskType

from ..options.presets import COCO_OPTIONS
from ..utils.coco_dataset import (
    coco_bbox_xyxy,
    coco_keypoints_flat,
    coco_segmentation_rle,
    group_annotations,
    iter_coco_images,
    load_category_map,
    load_coco_json,
)
from .base import BaseConverter
from .registry import register_converter
from .sources import Source

_COCO_INPUT_DATA_HELP = (
    "COCO instances JSON: images[], annotations[], categories[] "
    "(standard MS COCO detection / keypoint / segmentation export)."
)
_COCO_TYPICAL = (
    "  annotations/instances_*.json  — COCO export (categories inside)\n"
    "  images/…  — paths in JSON are often absolute; categories live in the JSON"
)


class _CocoConverter(BaseConverter):
    source_note = "MS COCO instance JSON (pycocotools-compatible layout)."

    converter_options = COCO_OPTIONS

    input_help = _COCO_INPUT_DATA_HELP
    input_sample = (
        '{"images":[{"id":1,"file_name":"a.jpg","width":640,"height":480}],'
        '"categories":[{"id":1,"name":"person"}],'
        '"annotations":[{"id":1,"image_id":1,"category_id":1,'
        '"bbox":[10,20,100,200]}]}'
    )
    typical_layout = _COCO_TYPICAL

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        same_dir = self.output_dir == self.input.parent
        stem = self.input.stem
        self.output_data_filename = (
            f"{stem}_vdschema.jsonl" if same_dir else f"{stem}.jsonl"
        )
        self.output_meta_filename = "label_dict.json"


@register_converter(task=TaskType.DETECTION, source=Source.COCO)
class CocoDetectionConverter(_CocoConverter):
    """COCO instances JSON → vdschema detection."""

    example_input = "/path/to/annotations/instances_train2017.json"

    def _convert(self) -> None:
        raw = load_coco_json(self.input)
        label, id_map = self._prepare_category_label(
            load_category_map(self.category, task=TaskType.DETECTION)
        )
        by_image = group_annotations(raw)
        writer = AnnotationWriter(
            TaskType.DETECTION,
            label=label,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for image in iter_coco_images(raw):
            instances = []
            for idx, ann in enumerate(by_image.get(image.id, [])):
                if "bbox" not in ann:
                    continue
                item: dict = {
                    "id": int(ann.get("id", idx)),
                    "category_id": self._map_category_id(
                        int(ann["category_id"]), id_map
                    ),
                    "bbox": coco_bbox_xyxy(ann),
                }
                if int(ann.get("iscrowd", 0)) == 1:
                    item["is_ignored"] = True
                instances.append(item)
            writer.append(
                filename=self.prefix_media_filename(image.file_name),
                width=image.width,
                height=image.height,
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.KEYPOINT, source=Source.COCO)
class CocoKeypointConverter(_CocoConverter):
    """COCO person keypoints JSON → vdschema keypoint."""

    example_input = "/path/to/annotations/person_keypoints_train2017.json"

    def _convert(self) -> None:
        raw = load_coco_json(self.input)
        label, id_map = self._prepare_category_label(
            load_category_map(self.category, task=TaskType.KEYPOINT)
        )
        by_image = group_annotations(raw)
        writer = AnnotationWriter(
            TaskType.KEYPOINT,
            label=label,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for image in iter_coco_images(raw):
            instances = []
            for idx, ann in enumerate(by_image.get(image.id, [])):
                keypoints = coco_keypoints_flat(ann)
                if not keypoints:
                    continue
                item: dict = {
                    "id": int(ann.get("id", idx)),
                    "category_id": self._map_category_id(
                        int(ann["category_id"]), id_map
                    ),
                    "keypoints": keypoints,
                }
                if "bbox" in ann:
                    item["bbox"] = coco_bbox_xyxy(ann)
                if int(ann.get("iscrowd", 0)) == 1:
                    item["is_ignored"] = True
                instances.append(item)
            writer.append(
                filename=self.prefix_media_filename(image.file_name),
                width=image.width,
                height=image.height,
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.SEGMENTATION, source=Source.COCO)
class CocoSegmentationConverter(_CocoConverter):
    """COCO instance segmentation JSON → vdschema segmentation."""

    example_input = "/path/to/annotations/instances_train2017.json"

    def _convert(self) -> None:
        raw = load_coco_json(self.input)
        label, id_map = self._prepare_category_label(
            load_category_map(self.category, task=TaskType.SEGMENTATION)
        )
        by_image = group_annotations(raw)
        writer = AnnotationWriter(
            TaskType.SEGMENTATION,
            label=label,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for image in iter_coco_images(raw):
            instances = []
            for idx, ann in enumerate(by_image.get(image.id, [])):
                if "bbox" not in ann:
                    continue
                segmentation = coco_segmentation_rle(
                    ann.get("segmentation"),
                    height=image.height,
                    width=image.width,
                )
                if segmentation is None:
                    continue
                item: dict = {
                    "id": int(ann.get("id", idx)),
                    "category_id": self._map_category_id(
                        int(ann["category_id"]), id_map
                    ),
                    "bbox": coco_bbox_xyxy(ann),
                    "rle_mask": segmentation.to_dict(),
                }
                if int(ann.get("iscrowd", 0)) == 1:
                    item["is_ignored"] = True
                instances.append(item)
            writer.append(
                filename=self.prefix_media_filename(image.file_name),
                width=image.width,
                height=image.height,
                instances=instances,
            )
        writer.save()
