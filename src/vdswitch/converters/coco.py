"""MS COCO JSON → vdschema (detection / keypoint / segmentation)."""

from __future__ import annotations

from vdschema import AnnotationWriter, TaskType

from ..utils.coco_dataset import (
    coco_bbox_xyxy,
    coco_keypoints_dict,
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
_COCO_INPUT_LABEL_HELP = (
    "Optional. Defaults to --input-data (categories[] in the same JSON). "
    "Or a JSON file with only categories[]."
)
_COCO_TYPICAL = (
    "  annotations/instances_*.json  — COCO export (categories inside)\n"
    "  images/…  — paths in JSON; use --input-root only if paths are relative"
)


class _CocoConverter(BaseConverter):
    source_note = "MS COCO instance JSON (pycocotools-compatible layout)."

    input_label_same_as_data = True
    input_root_optional = True

    input_data_help = _COCO_INPUT_DATA_HELP
    input_label_help = _COCO_INPUT_LABEL_HELP
    input_data_sample = (
        '{"images":[{"id":1,"file_name":"a.jpg","width":640,"height":480}],'
        '"categories":[{"id":1,"name":"person"}],'
        '"annotations":[{"id":1,"image_id":1,"category_id":1,'
        '"bbox":[10,20,100,200]}]}'
    )
    input_label_sample = "Same file as --input-data, or {\"categories\":[…]}"
    typical_layout = _COCO_TYPICAL

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        same_dir = self.output_dir == self.input_data.parent
        stem = self.input_data.stem
        self.output_data_filename = (
            f"{stem}_vdschema.jsonl" if same_dir else f"{stem}.jsonl"
        )
        self.output_meta_filename = "label_dict.json"


@register_converter(task=TaskType.DETECTION, source=Source.COCO)
class CocoDetectionConverter(_CocoConverter):
    """COCO instances JSON → vdschema detection."""

    def _convert(self) -> None:
        raw = load_coco_json(self.input_data)
        label = load_category_map(self.input_label, task=TaskType.DETECTION)
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
                    "category_id": int(ann["category_id"]),
                    "bbox": coco_bbox_xyxy(ann),
                }
                if int(ann.get("iscrowd", 0)) == 1:
                    item["is_ignored"] = True
                instances.append(item)
            writer.append(
                filename=image.file_name,
                width=image.width,
                height=image.height,
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.KEYPOINT, source=Source.COCO)
class CocoKeypointConverter(_CocoConverter):
    """COCO person keypoints JSON → vdschema keypoint."""

    def _convert(self) -> None:
        raw = load_coco_json(self.input_data)
        label = load_category_map(self.input_label, task=TaskType.KEYPOINT)
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
                if "bbox" not in ann:
                    continue
                keypoints = coco_keypoints_dict(
                    ann, categories=label, raw=raw
                )
                if not keypoints:
                    continue
                item: dict = {
                    "id": int(ann.get("id", idx)),
                    "category_id": int(ann["category_id"]),
                    "bbox": coco_bbox_xyxy(ann),
                    "keypoints": keypoints,
                }
                if int(ann.get("iscrowd", 0)) == 1:
                    item["is_ignored"] = True
                instances.append(item)
            writer.append(
                filename=image.file_name,
                width=image.width,
                height=image.height,
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.SEGMENTATION, source=Source.COCO)
class CocoSegmentationConverter(_CocoConverter):
    """COCO instance segmentation JSON → vdschema segmentation."""

    def _convert(self) -> None:
        raw = load_coco_json(self.input_data)
        label = load_category_map(self.input_label, task=TaskType.SEGMENTATION)
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
                    "category_id": int(ann["category_id"]),
                    "bbox": coco_bbox_xyxy(ann),
                    "segmentation": segmentation.to_dict(),
                }
                if int(ann.get("iscrowd", 0)) == 1:
                    item["is_ignored"] = True
                instances.append(item)
            writer.append(
                filename=image.file_name,
                width=image.width,
                height=image.height,
                instances=instances,
            )
        writer.save()
