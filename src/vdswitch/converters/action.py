"""Action converters."""

from __future__ import annotations

import json
from pathlib import Path

from vdschema import AnnotationWriter, Name, TaskType

from .base import BaseConverter
from .registry import register_converter_for_sources
from .sources import Source
from ..utils.kmot import parse_kmot_file
from ..utils.video_size import VideoSizeResolver, find_video_root


def _load_act_label(label_path: Path) -> tuple[dict[int, Name], dict[str, int]]:
    with label_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"invalid act label_dict: {label_path}")
    attr_name = next(iter(raw.keys()))
    class_names = list(raw[attr_name] or [])
    if not class_names:
        raise ValueError(f"act label[{attr_name!r}] is empty")
    label = {idx: Name(str(name)) for idx, name in enumerate(class_names)}
    name_to_id = {str(name): idx for idx, name in enumerate(class_names)}
    return label, name_to_id


@register_converter_for_sources(
    task=TaskType.ACTION,
    sources=(Source.MONOLITH, Source.UP),
)
class MonolithUpActionConverter(BaseConverter):
    """monolith/up video meta txt + kmot → vdschema action."""

    def _convert(self) -> None:
        label, name_to_id = _load_act_label(self.input_label)
        writer = AnnotationWriter(
            TaskType.ACTION,
            label=label,
            task_dir=self.output_dir,
            task_data_filename=self.output_data_filename,
            task_meta_filename=self.output_meta_filename,
        )

        data_root = self.input_data.parent
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
                parts = line.split(";")
                if len(parts) < 6:
                    raise ValueError(f"invalid act meta line: {line!r}")

                # legacy meta 仅用于定位 video 与 kmot 文件；动作字段来自 kmot。
                video_path = parts[0]
                kmot_rel = parts[5]
                kmot_path = (data_root / kmot_rel).resolve()
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
