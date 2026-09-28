# vdschema usage

[← Back to README](../README.md) · [API reference](api.md) · [JSON examples](example.md)

VDschema provides typed **write** and **read** paths over a unified JSONL layout (`annotation_data.jsonl` plus optional label meta).

## Supported tasks


| Task           | `TaskType`                | `label`                                         | Python type                |
| -------------- | ------------------------- | ----------------------------------------------- | -------------------------- |
| manifest       | `TaskType.MANIFEST`       | —                                               | `ManifestAnnotation`       |
| detection      | `TaskType.DETECTION`      | `{id: Name(...)}`                               | `DetectionAnnotation`      |
| keypoint       | `TaskType.KEYPOINT`       | `{id: Name(...)}`                               | `KeypointAnnotation`       |
| segmentation   | `TaskType.SEGMENTATION`   | `{id: Name(...)}`                               | `SegmentationAnnotation`   |
| classification | `TaskType.CLASSIFICATION` | `{head: {id: Name(...)}}`                       | `ClassificationAnnotation` |
| relationship   | `TaskType.RELATIONSHIP`   | `{detection: {...}, relationship: {...}}`       | `RelationshipAnnotation`   |
| vlm            | `TaskType.VLM`            | —                                               | `VlmAnnotation`            |
| conversation   | `TaskType.CONVERSATION`   | —                                               | `ConversationAnnotation`   |
| sequence       | `TaskType.SEQUENCE`       | `"/path/to/vocab.txt"` → `annotation_vocab.txt` | `SequenceAnnotation`       |
| action         | `TaskType.ACTION`         | `{id: Name(...)}`                               | `ActionAnnotation`         |


Schema: [annotation_data.json](../src/vdschema/schema/annotation_data.json), [annotation_meta.json](../src/vdschema/schema/annotation_meta.json).

## Detection geometry

Each instance must include at least one of:


| Format                          | Field      | Shape                                   |
| ------------------------------- | ---------- | --------------------------------------- |
| Horizontal box                  | `bbox`     | `[x1, y1, x2, y2]` (xyxy)               |
| Oriented box / closed contour   | `polygon`  | flat `[x1, y1, x2, y2, …]`, ≥3 vertices |
| Open polyline (e.g. lane lines) | `polyline` | flat `[x1, y1, x2, y2, …]`, ≥2 vertices |
| Point target (e.g. counting)    | `point`    | `[x, y]`                                |


## Detection

[API reference](api.md)

### Axis-aligned bounding box (bbox)

```python
from vdschema import AnnotationReader, AnnotationWriter, Name, TaskType

writer = AnnotationWriter(
    TaskType.DETECTION,
    label={
        1: Name("person", alias=["human"], prompt=["a human", "人体"]),
        2: Name("car"),
    },
)
writer.append(
    filename="images/sample.jpg",
    width=640,
    height=480,
    instances=[
        {"id": 0, "category_id": 1, "bbox": [10, 20, 100, 200]},
        {"id": 1, "category_id": 1, "bbox": [160, 190, 300, 560], "is_ignored": True},
    ],
)
writer.save()
data, label = AnnotationReader(TaskType.DETECTION, writer.save_dir()).load()
print(label[1].name, label[1].alias)  # person ('human',)
```

### Oriented box (polygon)

```python
from vdschema import AnnotationReader, AnnotationWriter, Name, TaskType

writer = AnnotationWriter(TaskType.DETECTION, label={1: Name("ship")})
writer.append(
    filename="images/obb_001.jpg",
    width=640,
    height=480,
    instances=[
        {
            "id": 0,
            "category_id": 1,
            "polygon": [120, 80, 200, 60, 220, 140, 140, 160],
        }
    ],
)
writer.save()
data, _ = AnnotationReader(TaskType.DETECTION, writer.save_dir()).load()
```

### Open polyline

```python
from vdschema import AnnotationReader, AnnotationWriter, Name, TaskType

writer = AnnotationWriter(TaskType.DETECTION, label={1: Name("lane_line")})
writer.append(
    filename="images/lane_001.jpg",
    width=1280,
    height=720,
    instances=[
        {
            "id": 0,
            "category_id": 1,
            "polyline": [100, 650, 280, 520, 460, 410, 640, 340],
        }
    ],
)
writer.save()
data, _ = AnnotationReader(TaskType.DETECTION, writer.save_dir()).load()
```

### Point

```python
from vdschema import AnnotationReader, AnnotationWriter, Name, TaskType

writer = AnnotationWriter(TaskType.DETECTION, label={1: Name("defect")})
writer.append(
    filename="images/count_001.jpg",
    width=640,
    height=480,
    instances=[
        {"id": 0, "category_id": 1, "point": [120.5, 340.0]},
        {"id": 1, "category_id": 1, "point": [400, 200]},
    ],
)
writer.save()
data, _ = AnnotationReader(TaskType.DETECTION, writer.save_dir()).load()
```

## Manifest

Use manifest for **inference or evaluation media lists**: each line has only `filename`, `width`, and `height`. No `label` dictionary and no `annotation_meta.json`. The same on-disk directory can be read with another `TaskType` on `AnnotationReader` (for example, an inference manifest read as detection).

```python
from vdschema import (
    AnnotationReader,
    AnnotationWriter,
    DetectionAnnotation,
    TaskType,
    VlmAnnotation,
)

writer = AnnotationWriter(TaskType.MANIFEST)
writer.append(filename="images/infer_001.jpg", width=640, height=480)
writer.append(filename="videos/infer_002.mp4", width=1920, height=1080)
writer.save()

manifest_dir = writer.save_dir()

data, label = AnnotationReader(TaskType.MANIFEST, manifest_dir).load()
assert label is None
assert data[0].to_dict() == {
    "filename": "images/infer_001.jpg",
    "width": 640,
    "height": 480,
}

data_det, _ = AnnotationReader(TaskType.DETECTION, manifest_dir).load()
assert isinstance(data_det[0], DetectionAnnotation)
assert data_det[0].instances == []

data_vlm, _ = AnnotationReader(TaskType.VLM, manifest_dir).load()
assert isinstance(data_vlm[0], VlmAnnotation)
assert data_vlm[0].description == ""
```

## Keypoint

COCO-style **flat** array `[x1, y1, v1, x2, y2, v2, ...]` (length is a multiple of 3). `bbox` is optional.

```python
from vdschema import AnnotationReader, AnnotationWriter, Name, TaskType

writer = AnnotationWriter(TaskType.KEYPOINT, label={1: Name("person")})
writer.append(
    filename="images/keypoint_001.jpg",
    width=640,
    height=480,
    instances=[
        {
            "id": 0,
            "category_id": 1,
            "keypoints": [210, 120, 2, 230, 140, 2],
        }
    ],
)
writer.save()
data, label = AnnotationReader(
    TaskType.KEYPOINT, writer.save_dir()
).load()
```

## Segmentation

```python
from vdschema import (
    AnnotationReader,
    AnnotationWriter,
    Bbox,
    Name,
    SegmentationRLE,
    TaskType,
)
import numpy as np

mask = np.zeros((480, 640), dtype=np.uint8)
mask[20:80, 30:120] = 1
task_dir = "output/segmentation"
writer = AnnotationWriter(
    TaskType.SEGMENTATION,
    label={1: Name("person")},
    task_dir=task_dir,
)
writer.append(
    filename="images/segmentation_001.jpg",
    width=640,
    height=480,
    instances=[
        {
            "id": 0,
            "category_id": 1,
            "bbox": Bbox.from_xywh([120, 30, 200, 180]),
            "rle_mask": SegmentationRLE.from_mask(mask),
        }
    ],
)
writer.save()
data, label = AnnotationReader(TaskType.SEGMENTATION, task_dir).load()
```

## Classification

```python
from vdschema import AnnotationReader, AnnotationWriter, Name, TaskType

writer = AnnotationWriter(
    TaskType.CLASSIFICATION,
    label={
        "hair_color": {1: Name("black"), 2: Name("brown")},
        "age": {1: Name("young"), 2: Name("middle-aged")},
    },
)
writer.append(
    filename="images/classification_001.jpg",
    width=640,
    height=480,
    categories=[
        {"category_attr": "hair_color", "category_ids": [1]},
        {"category_attr": "age", "category_ids": [1]},
    ],
)
writer.save()
data, label = AnnotationReader(
    TaskType.CLASSIFICATION, writer.save_dir()
).load()
```

## Relationship

```python
from vdschema import AnnotationReader, AnnotationWriter, Name, TaskType

writer = AnnotationWriter(
    TaskType.RELATIONSHIP,
    label={
        "detection": {1: Name("person"), 2: Name("car")},
        "relationship": {0: Name("near"), 1: Name("left_of")},
    },
)
writer.append(
    filename="images/relationship_001.jpg",
    width=640,
    height=480,
    instances=[
        {"id": 0, "category_id": 1, "bbox": [10, 20, 100, 200]},
        {"id": 1, "category_id": 2, "bbox": [120, 30, 200, 180]},
    ],
    relationships=[{"subject_id": 0, "object_id": 1, "relation_type": "near"}],
)
writer.save()
data, label = AnnotationReader(
    TaskType.RELATIONSHIP, writer.save_dir()
).load()
```

## VLM

No label dictionary file. Omit `label` for tasks without vocabulary.

```python
from vdschema import AnnotationReader, AnnotationWriter, TaskType

writer = AnnotationWriter(TaskType.VLM)
writer.append(
    filename="images/vlm_001.jpg",
    width=640,
    height=480,
    description="A street intersection with three people crossing.",
)
writer.save()
data, label = AnnotationReader(
    TaskType.VLM, writer.save_dir()
).load()
assert label is None
```

## Conversation

```python
from vdschema import (
    AnnotationReader,
    AnnotationWriter,
    ConversationRole,
    ConversationTurn,
    TaskType,
)

writer = AnnotationWriter(TaskType.CONVERSATION)
writer.append(
    filename="images/conversation_001.jpg",
    width=640,
    height=480,
    conversations=[
        ConversationTurn(
            role=ConversationRole.USER,
            image="images/conversation_001.jpg",
            text="Describe the image.",
        ),
        ConversationTurn(role=ConversationRole.ASSISTANT, text="A street scene."),
    ],
)
writer.save()
data, label = AnnotationReader(
    TaskType.CONVERSATION, writer.save_dir()
).load()
```

## Sequence

Vocabulary is provided as an external file and copied to the output directory as `annotation_vocab.txt`. The file must contain **exactly one token per line** (no spaces/tabs within a line). See [assets/](../assets/) for examples such as `alphanumeric_vocab.txt`. Each token in `sequences` must appear in the vocabulary file; validation runs on `append()` and on read.

```python
from vdschema import AnnotationReader, AnnotationWriter, TaskType

writer = AnnotationWriter(
    TaskType.SEQUENCE,
    label="/path/to/vocab.txt",
)
writer.append(
    filename="batch1_crop_plate/27993412_car0_inst0.jpg",
    width=224,
    height=128,
    sequences=["B", "1", "0", "7", "7", "P", "D", "V"],
)
writer.save()
data, label = AnnotationReader(
    TaskType.SEQUENCE, writer.save_dir()
).load()
assert label == {"vocab": "annotation_vocab.txt"}
```

## Action

```python
from vdschema import AnnotationReader, AnnotationWriter, Name, TaskType

writer = AnnotationWriter(
    TaskType.ACTION,
    label={2: Name("fall"), 4: Name("walk")},
)
writer.append(
    filename="videos/action_001.mp4",
    width=1920,
    height=1080,
    description="An elderly person falls down.",
    actions=[
        {
            "category_id": 2,
            "track_id": 0,
            "start_idx": 4,
            "end_idx": 6,
            "description": "An elderly person falls down.",
            "tracks": [{"frame_idx": 0, "bbox": [520, 300, 620, 380]}],
        }
    ],
)
writer.save()
data, label = AnnotationReader(
    TaskType.ACTION, writer.save_dir()
).load()
```

## More examples

Full runnable tests: [tests/test_annotation_format.py](../tests/test_annotation_format.py).

```bash
uv run pytest tests/test_annotation_format.py
```

