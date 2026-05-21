# VDschema

**VDschema**: A unified annotation **schema** for **V**ision **D**atasets with JSONL I/O support.

It supports two primary workflows:

- **Write** — annotation pipelines produce unified JSONL annotations data
- **Read** — training pipelines load JSONL into typed Python annotation objects.

## Supported tasks


| Task           | `TaskType`                | `Task_dict`                               | Python type                |
| -------------- | ------------------------- | ----------------------------------------- | -------------------------- |
| detection      | `TaskType.DETECTION`      | `{id: name}`                              | `DetectionAnnotation`      |
| keypoint       | `TaskType.KEYPOINT`       | `{id: name}`                              | `KeypointAnnotation`       |
| segmentation   | `TaskType.SEGMENTATION`   | `{id: name}`                              | `SegmentationAnnotation`   |
| classification | `TaskType.CLASSIFICATION` | `{head: {id: name}}`                      | `ClassificationAnnotation` |
| relationship   | `TaskType.RELATIONSHIP`   | `{detection: {...}, relationship: {...}}` | `RelationshipAnnotation`   |
| vlm            | `TaskType.VLM`            | —                                         | `VlmAnnotation`            |
| conversation   | `TaskType.CONVERSATION`   | —                                         | `ConversationAnnotation`   |
| action         | `TaskType.ACTION`         | `{id: name}`                              | `ActionAnnotation`         |


JSON examples for each task: [src/vdschema/schema/example.md](src/vdschema/schema/example.md).

Schema definitions: [annotation_schema.json](src/vdschema/schema/annotation_schema.json), [annotation_dict.json](src/vdschema/schema/annotation_dict.json).

## Install

**PyPI (pip)**

```bash
pip install vdschema
```

**PyPI (uv)**

```bash
uv pip install vdschema

# Add as a project dependency if need
uv add vdschema
```

**From source (git)**

```bash
git clone https://github.com/Arrkwen/vdschema.git
cd vdschema
pip install -e .
```

## Basic Usage
Expand a task below for a full example:
<details open>
<summary><strong>Detection</strong></summary>

```python
from vdschema import AnnotationReader, AnnotationWriter, TaskType

# step1: create annotation object
writer = AnnotationWriter(
    TaskType.DETECTION,
    task_dict={1: "person", 2: "car"},
)

# step2: add annotation info
writer.append(
    filename="images/sample.jpg",
    width=640,
    height=480,
    instances=[
        {"id": 0, "category_id": 1, "bbox": [10, 20, 100, 200]},
    ],
)
# step3: save annotation info, by default, written to output/{task_type}/,can override by setting task_dir in AnnotationWriter.
writer.save()

# step4(option): load annotation info.
annotations, task_dict = AnnotationReader(
    TaskType.DETECTION, writer.save_dir()
).load()

print(task_dict)  # {1: "person", 2: "car"}
print(annotations) # [DetectionAnnotation(filename='images/sample.jpg', width=640, height=480, instances=[Instance(id=0, category_id=1, bbox=Bbox(x1=10.0, y1=20.0, x2=100.0, y2=200.0), keypoints=None, segmentation=None, text=None)], description=None)]
```

</details>

<details>
<summary><strong>Keypoint</strong></summary>

```python
from vdschema import AnnotationReader, AnnotationWriter, TaskType

writer = AnnotationWriter(TaskType.KEYPOINT, task_dict={1: "person"})
writer.append(
    filename="images/keypoint_001.jpg",
    width=640,
    height=480,
    instances=[
        {
            "id": 0,
            "category_id": 1,
            "bbox": [200, 100, 380, 420],
            "keypoints": {"body": [[210, 120, 2], [230, 140, 2]]},
        }
    ],
)
writer.save()
annotations, task_dict = AnnotationReader(
    TaskType.KEYPOINT, writer.save_dir()
).load()
```

</details>

<details>
<summary><strong>Segmentation</strong></summary>

```python
from vdschema import (
    AnnotationReader,
    AnnotationWriter,
    Bbox,
    SegmentationRLE,
    TaskType,
)
import numpy as np

mask = np.zeros((480, 640), dtype=np.uint8)
mask[20:80, 30:120] = 1
task_dir = "output/segmentation"
writer = AnnotationWriter(
    TaskType.SEGMENTATION,
    task_dict={1: "person"},
    task_dir=task_dir,
)
# if bbox is not xyxy format, support other format(xywh, cxxywh)
writer.append(
    filename="images/segmentation_001.jpg",
    width=640,
    height=480,
    instances=[
        {
            "id": 0,
            "category_id": 1,
            "bbox": Bbox.from_xywh([120, 30, 200, 180]),
            "segmentation": SegmentationRLE.from_mask(mask),
        }
    ],
)
writer.save()
annotations, task_dict = AnnotationReader(TaskType.SEGMENTATION, task_dir).load()
```

</details>

<details>
<summary><strong>Classification</strong></summary>

```python
from vdschema import AnnotationReader, AnnotationWriter, TaskType

writer = AnnotationWriter(
    TaskType.CLASSIFICATION,
    task_dict={
        "hair_color": {1: "black", 2: "brown"},
        "age": {1: "young", 2: "middle-aged"},
    },
)
writer.append(
    filename="images/classification_001.jpg",
    width=640,
    height=480,
    categories=[
        {"category_type": "hair_color", "category_ids": [1]},
        {"category_type": "age", "category_ids": [1]},
    ],
)
writer.save()
annotations, task_dict = AnnotationReader(
    TaskType.CLASSIFICATION, writer.save_dir()
).load()
```

</details>

<details>
<summary><strong>Relationship</strong></summary>

```python
from vdschema import AnnotationReader, AnnotationWriter, TaskType

writer = AnnotationWriter(
    TaskType.RELATIONSHIP,
    task_dict={
        "detection": {1: "person", 2: "car"},
        "relationship": {0: "near", 1: "left_of"},
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
annotations, task_dict = AnnotationReader(
    TaskType.RELATIONSHIP, writer.save_dir()
).load()
```

</details>

<details>
<summary><strong>VLM</strong></summary>

No label dictionary file. Omit `task_dict` for tasks without vocabulary.

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
annotations, task_dict = AnnotationReader(
    TaskType.VLM, writer.save_dir()
).load()
assert task_dict is None
```

</details>

<details>
<summary><strong>Conversation</strong></summary>

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
annotations, task_dict = AnnotationReader(
    TaskType.CONVERSATION, writer.save_dir()
).load()
```

</details>

<details>
<summary><strong>Action</strong></summary>

```python
from vdschema import AnnotationReader, AnnotationWriter, TaskType

writer = AnnotationWriter(
    TaskType.ACTION,
    task_dict={2: "fall", 4: "walk"},
)
writer.append(
    filename="videos/action_001.mp4",
    width=1920,
    height=1080,
    description="An elderly person falls down.",
    actions=[
        {
            "action_id": 2,
            "track_id": 0,
            "start_idx": 4,
            "end_idx": 6,
            "description": "An elderly person falls down.",
            "tracks": [{"frame_idx": 0, "bbox": [520, 300, 620, 380]}],
        }
    ],
)
writer.save()
annotations, task_dict = AnnotationReader(
    TaskType.ACTION, writer.save_dir()
).load()
```

</details>

### More examples

Full runnable tests: [tests/test_annotation_format.py](tests/test_annotation_format.py). Run:

```bash
uv run pytest
```

## Development

```bash
git clone https://github.com/Arrkwen/vdschema.git
cd vdschema
uv sync --group dev
uv run pytest
uv run ruff check .
uv build
```

## Publishing

1. Bump `__version__` in `src/vdschema/_version.py` (used at runtime and by `uv build`).
2. Create a **Published** GitHub Release with a new tag (e.g. `0.1.0`).

The [publish.yml](.github/workflows/publish.yml) workflow runs on release publish and uploads the build to PyPI. You can also re-run it manually from Actions → **vdschema-publisher**.