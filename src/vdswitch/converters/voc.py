"""PASCAL VOC → vdschema (detection XML / segmentation class PNG)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vdschema import AnnotationWriter, Name, TaskType

from ..options.presets import VOC_OPTIONS
from ..utils.voc_dataset import (
    collect_detection_class_names,
    detection_instances_from_record,
    detection_label_dict,
    iter_voc_image_ids,
    load_optional_class_list,
    parse_voc_detection_xml,
    resolve_media_filename,
    resolve_voc_root,
    segmentation_instances_from_class_png,
    voc2012_label_dict,
    voc_annotation_xml_path,
    voc_segmentation_class_path,
)
from .base import BaseConverter
from .registry import register_converter
from .sources import Source

_VOC_LAYOUT = (
    "  VOC2007/ (or VOC2012/)\n"
    "    Annotations/{id}.xml     — detection boxes\n"
    "    JPEGImages/{id}.jpg\n"
    "    SegmentationClass/{id}.png — semantic labels (seg task)\n"
    "    ImageSets/Main/train.txt   — image ids, one per line"
)


class _VocConverterBase(BaseConverter):
    source_note = "PASCAL VOC devkit (XML detection + SegmentationClass PNG)."

    input_help = (
        "VOC ImageSets list (e.g. ImageSets/Main/train.txt), a single "
        "Annotations/*.xml, or the Annotations/ directory."
    )
    input_sample = "ImageSets/Main/train.txt"
    typical_layout = _VOC_LAYOUT
    converter_options = VOC_OPTIONS
    example_input = "/path/to/VOC2007/ImageSets/Main/train.txt"

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

    def _voc_root(self) -> Path:
        return resolve_voc_root(self.input, self.root)

    def _image_ids(self, voc_root: Path) -> list[str]:
        ids = iter_voc_image_ids(self.input, voc_root)
        if not ids:
            raise ValueError(f"no VOC image ids from {self.input}")
        return ids

    def _optional_class_list(self) -> dict[int, Name] | None:
        if self.category.resolve() == self.input.resolve():
            return None
        return load_optional_class_list(self.category)


@register_converter(task=TaskType.DETECTION, source=Source.VOC)
class VocDetectionConverter(_VocConverterBase):
    """VOC Annotations/*.xml → vdschema detection."""

    def _convert(self) -> None:
        voc_root = self._voc_root()
        image_ids = self._image_ids(voc_root)
        optional = self._optional_class_list()
        if optional is not None:
            name_to_id = {name.name: cid for cid, name in optional.items()}
        else:
            name_to_id = collect_detection_class_names(voc_root, image_ids)
        writer = AnnotationWriter(
            TaskType.DETECTION,
            label=detection_label_dict(name_to_id),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for image_id in image_ids:
            xml_path = voc_annotation_xml_path(voc_root, image_id)
            if not xml_path.is_file():
                continue
            record = parse_voc_detection_xml(xml_path)
            instances = detection_instances_from_record(record, name_to_id=name_to_id)
            if not instances:
                continue
            filename = resolve_media_filename(
                voc_root,
                image_id,
                record,
                root=self.root,
            )
            writer.append(
                filename=self.prefix_media_filename(filename),
                width=record.width,
                height=record.height,
                instances=instances,
            )
        writer.save()


@register_converter(task=TaskType.SEGMENTATION, source=Source.VOC)
class VocSegmentationConverter(_VocConverterBase):
    """VOC SegmentationClass/*.png → vdschema segmentation (RLE per class)."""

    def _convert(self) -> None:
        voc_root = self._voc_root()
        seg_dir = voc_root / "SegmentationClass"
        if not seg_dir.is_dir():
            raise FileNotFoundError(
                f"SegmentationClass/ not found under VOC root: {voc_root}"
            )
        image_ids = self._image_ids(voc_root)
        label = self._optional_class_list() or voc2012_label_dict()
        writer = AnnotationWriter(
            TaskType.SEGMENTATION,
            label=label,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )
        for image_id in image_ids:
            png_path = voc_segmentation_class_path(voc_root, image_id)
            instances = segmentation_instances_from_class_png(png_path, label=label)
            if not instances:
                continue
            xml_path = voc_annotation_xml_path(voc_root, image_id)
            if xml_path.is_file():
                record = parse_voc_detection_xml(xml_path)
                width, height = record.width, record.height
                filename = resolve_media_filename(
                    voc_root,
                    image_id,
                    record,
                    root=self.root,
                )
            else:
                jpeg = voc_root / "JPEGImages" / f"{image_id}.jpg"
                if not jpeg.is_file():
                    continue
                with Image.open(jpeg) as img:
                    width, height = img.size
                filename = resolve_media_filename(
                    voc_root, image_id, None, root=self.root
                )
            writer.append(
                filename=self.prefix_media_filename(filename),
                width=width,
                height=height,
                instances=instances,
            )
        writer.save()
