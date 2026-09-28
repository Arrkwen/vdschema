"""Monolith legacy JSONL / meta layouts → vdschema."""

from __future__ import annotations

import json
from pathlib import Path

from vdschema import AnnotationWriter, Name, TaskType

from ..messages import help_hint
from ..utils.jsonl import load_jsonl_object
from ..utils.kmot import parse_kmot_file
from ..utils.video_size import VideoSizeResolver, find_video_root
from .base import BaseConverter
from .registry import register_converter
from .sources import Source

_MONOLITH_JSONL_LAYOUT = (
    "  meta/train_baseline.jsonl  — annotation JSONL\n"
    "  meta/label_dict.json or vocab.txt  — labels / vocabulary\n"
    "  images/…                   — media (use --input-root)"
)


def _load_det_label(label_path: Path) -> dict[int, Name]:
    with label_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"invalid detection label file: {label_path} "
            f'(expected JSON {{"1":"name1","2":"name2", ...}}). '
            f"{help_hint(TaskType.DETECTION)}"
        )
    return {int(key): Name(str(name)) for key, name in raw.items()}


def _load_cls_label(label_path: Path) -> dict[str, dict[int, Name]]:
    with label_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"invalid classification label file: {label_path} "
            f'(expected JSON {{"attr": ["class_a", "class_b", ...]}}). '
            f"{help_hint(TaskType.CLASSIFICATION)}"
        )
    heads: dict[str, dict[int, Name]] = {}
    for attr, names in raw.items():
        if not isinstance(names, list) or not names:
            raise ValueError(
                f"label[{attr!r}] must be a non-empty list of class names in {label_path}. "
                f"{help_hint(TaskType.CLASSIFICATION)}"
            )
        heads[str(attr)] = {idx + 1: Name(str(name)) for idx, name in enumerate(names)}
    return heads


def _load_act_label(label_path: Path) -> tuple[dict[int, Name], dict[str, int]]:
    with label_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"invalid action label file: {label_path}. {help_hint(TaskType.ACTION)}"
        )

    if "action" in raw and isinstance(raw["action"], list):
        label: dict[int, Name] = {}
        name_to_id: dict[str, int] = {}
        for item in raw["action"]:
            if not isinstance(item, dict):
                continue
            cid = int(item["category_id"])
            name = str(item["category_name"])
            label[cid] = Name(name)
            name_to_id[name] = cid
        if not label:
            raise ValueError(f"act label action[] is empty: {label_path}")
        return label, name_to_id

    attr_name = next(iter(raw.keys()))
    class_names = list(raw[attr_name] or [])
    if not class_names:
        raise ValueError(f"act label[{attr_name!r}] is empty")
    label = {idx: Name(str(name)) for idx, name in enumerate(class_names)}
    name_to_id = {str(name): idx for idx, name in enumerate(class_names)}
    return label, name_to_id


def _parse_legacy_meta_line(line: str) -> tuple[str, str] | None:
    """Parse legacy act meta line; ``None`` means skip (same rules as monolith datasets).

    Meta start/end/label are only used for legacy QC filtering; action fields
    come from kmot tracks.
    """
    parts = line.split(";")
    if len(parts) < 6:
        raise ValueError(f"invalid act meta line: {line!r}")

    video_path = parts[0]
    num_frames = int(parts[1])
    start = int(parts[2])
    end = int(parts[3])
    kmot_rel = parts[5]

    if num_frames < 0:
        num_frames = abs(num_frames)
        start, end = end, start
    if start > end:
        return None
    num_frames = end - start + 1
    if num_frames < 4:
        return None

    return video_path, kmot_rel


@register_converter(task=TaskType.DETECTION, source=Source.MONOLITH)
class MonolithUpDetectionConverter(BaseConverter):
    """Monolith baseline jsonl + label_dict.json → vdschema detection."""

    input_data_help = (
        "JSONL file: one JSON object per line with filename and instances "
        "(bbox, label or category_id, optional is_ignored)."
    )
    input_label_help = (
        "JSON file: string keys are category ids, values are class names "
        '(e.g. {"1":"person","2":"car"}).'
    )
    input_data_sample = (
        '{"filename":"img.jpg","instances":[{"id":0,"label":1,"bbox":[10,10,100,100]}]}'
    )
    input_label_sample = '{"1":"person","2":"car"}'
    typical_layout = _MONOLITH_JSONL_LAYOUT

    def _convert(self) -> None:
        writer = AnnotationWriter(
            TaskType.DETECTION,
            label=_load_det_label(self.input_label),
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )

        with self.input_data.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = load_jsonl_object(line, path=self.input_data, lineno=lineno)
                instances = []
                for idx, inst in enumerate(record.get("instances") or []):
                    item = {
                        "id": int(inst.get("id", idx)),
                        "category_id": int(inst.get("label", inst.get("category_id"))),
                        "bbox": inst["bbox"],
                    }
                    if inst.get("is_ignored") or inst.get("is_ignore"):
                        item["is_ignored"] = True
                    instances.append(item)
                width, height = self.image_size_resolver.resolve(record)
                writer.append(
                    filename=record["filename"],
                    width=width,
                    height=height,
                    instances=instances,
                )

        writer.save()


@register_converter(task=TaskType.CLASSIFICATION, source=Source.MONOLITH)
class MonolithUpClassificationConverter(BaseConverter):
    """Monolith baseline jsonl + label_dict.json → vdschema classification."""

    input_data_help = (
        "JSONL file: one JSON object per line with filename and attribute.\n"
        "attribute maps each head name to one-hot dict {class_name: 0|1, ...}."
    )
    input_label_help = (
        "JSON file: each key is a classification head (attribute) name; "
        "each value is an ordered list of class names (not numeric id maps)."
    )
    input_data_sample = (
        '{"filename":"img.jpg","attribute":{"gender":{"male":0,"female":1}}}'
    )
    input_label_sample = '{"gender":["male","female"]}'
    typical_layout = _MONOLITH_JSONL_LAYOUT

    def _convert(self) -> None:
        label_heads = _load_cls_label(self.input_label)
        with self.input_label.open(encoding="utf-8") as f:
            raw = json.load(f)
        ordered_heads = {str(k): list(v) for k, v in raw.items()}

        writer = AnnotationWriter(
            TaskType.CLASSIFICATION,
            label=label_heads,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )

        with self.input_data.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = load_jsonl_object(line, path=self.input_data, lineno=lineno)
                attribute = record.get("attribute") or {}
                categories = []
                for attr_name, one_hot in attribute.items():
                    if not isinstance(one_hot, dict):
                        continue
                    names = ordered_heads.get(str(attr_name))
                    if not names:
                        continue
                    category_ids = [
                        idx + 1
                        for idx, name in enumerate(names)
                        if int(one_hot.get(name, 0)) == 1
                    ]
                    if not category_ids:
                        continue
                    categories.append(
                        {"category_attr": str(attr_name), "category_ids": category_ids}
                    )
                width, height = self.image_size_resolver.resolve(record)
                writer.append(
                    filename=record["filename"],
                    width=width,
                    height=height,
                    categories=categories,
                )

        writer.save()


@register_converter(task=TaskType.SEQUENCE, source=Source.MONOLITH)
class MonolithUpSequenceConverter(BaseConverter):
    """Monolith baseline jsonl + vocab.txt → vdschema sequence."""

    input_data_help = (
        "JSONL file: one JSON object per line with filename and sequences "
        "(list of token strings from the vocab file)."
    )
    input_label_help = (
        "Plain-text vocab file: one token per line (not JSON). "
        "Copied to vdschema meta as the sequence vocabulary."
    )
    input_data_sample = '{"filename":"img.jpg","sequences":["B","1","0"]}'
    input_label_sample = "0\n1\nB\n"
    typical_layout = _MONOLITH_JSONL_LAYOUT

    def _convert(self) -> None:
        writer = AnnotationWriter(
            TaskType.SEQUENCE,
            label=self.input_label,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )

        with self.input_data.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = load_jsonl_object(line, path=self.input_data, lineno=lineno)
                width, height = self.image_size_resolver.resolve(record)
                writer.append(
                    filename=record["filename"],
                    width=width,
                    height=height,
                    sequences=list(record.get("sequences") or []),
                )

        writer.save()


@register_converter(task=TaskType.ACTION, source=Source.MONOLITH)
class MonolithUpActionConverter(BaseConverter):
    """Monolith video meta txt + kmot → vdschema action."""

    input_data_help = (
        "Text meta file: one video per line, semicolon-separated fields.\n"
        "Format: video_path;num_frames;start;end;unused;kmot_relative_path\n"
        "kmot paths are resolved relative to the meta file directory."
    )
    input_label_help = (
        'JSON file: either {"action":[{"category_id":1,"category_name":"…"},…]} '
        'or a single head mapping {"head_name":["class_a","class_b",…]} '
        "(class names must match kmot track labels)."
    )
    input_data_sample = "video/sample.avi;4;443;446;1;kmot/sample.txt"
    input_label_sample = (
        '{"action":[{"category_id":1,"category_name":"package_tossing"}]}\n'
        'or {"package_tossing":["normal","package_tossing"]}'
    )
    typical_layout = (
        "  meta/video_train.txt       — video list + kmot pointers\n"
        "  meta/kmot/*.txt            — kmot tracks\n"
        "  meta/label_dict.json       — action categories\n"
        "  video/…                    — media (use --input-root)"
    )

    def _convert(self) -> None:
        label, name_to_id = _load_act_label(self.input_label)
        writer = AnnotationWriter(
            TaskType.ACTION,
            label=label,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )

        video_size = VideoSizeResolver(
            find_video_root(
                output_dir=self.output_dir,
                input_data=self.input_data,
                root=self.input_root,
            )
        )
        with self.input_data.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parsed = _parse_legacy_meta_line(line)
                if parsed is None:
                    continue
                video_path, kmot_rel = parsed

                kmot_path = (self.input_data.parent / kmot_rel).resolve()
                if not kmot_path.is_file():
                    continue

                grouped = parse_kmot_file(kmot_path, name_to_id=name_to_id)
                actions = [
                    track.to_action_dict()
                    for track_id in sorted(grouped)
                    if (track := grouped[track_id]).frames
                ]
                if not actions:
                    continue

                width, height = video_size.resolve(video_path)
                writer.append(
                    filename=video_path,
                    width=width,
                    height=height,
                    actions=actions,
                )

        writer.save()
