# VDschema

**VDschema**: A unified annotation **schema** for **V**ision **D**atasets with JSONL I/O support.

It supports two primary workflows:

- **Write** — annotation pipelines produce unified JSONL annotations data
- **Read** — training pipelines load JSONL into typed Python annotation objects.

## Supported tasks


| Task           | `TaskType`                | `label`                                         | Python type                |
| -------------- | ------------------------- | ----------------------------------------------- | -------------------------- |
| detection      | `TaskType.DETECTION`      | `{id: Name(...)}`                               | `DetectionAnnotation`      |
| keypoint       | `TaskType.KEYPOINT`       | `{id: Name(...)}`                               | `KeypointAnnotation`       |
| segmentation   | `TaskType.SEGMENTATION`   | `{id: Name(...)}`                               | `SegmentationAnnotation`   |
| classification | `TaskType.CLASSIFICATION` | `{head: {id: Name(...)}}`                       | `ClassificationAnnotation` |
| relationship   | `TaskType.RELATIONSHIP`   | `{detection: {...}, relationship: {...}}`       | `RelationshipAnnotation`   |
| vlm            | `TaskType.VLM`            | —                                               | `VlmAnnotation`            |
| conversation   | `TaskType.CONVERSATION`   | —                                               | `ConversationAnnotation`   |
| sequence       | `TaskType.SEQUENCE`       | `"/path/to/vocab.txt"` → `annotation_vocab.txt` | `SequenceAnnotation`       |
| action         | `TaskType.ACTION`         | `{id: Name(...)}`                               | `ActionAnnotation`         |


JSON examples for each task: [src/vdschema/schema/example.md](src/vdschema/schema/example.md).

Schema definitions: [annotation_meta.json](src/vdschema/schema/annotation_meta.json), [annotation_data.json](src/vdschema/schema/annotation_data.json). Sequence tasks use `**annotation_vocab.txt`** instead of `annotation_meta.json`.

## Install

**PyPI (pip)**

```bash
pip install vdschema

# Legacy conversion CLI (opencv, pillow)
pip install "vdschema[vdswitch]"
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
uv sync --extra vdswitch --group dev
pip install -e ".[vdswitch]"
```

### Paths on disk

Each task directory holds `annotation_data.jsonl` plus optional label meta:

- `**data_path**` — per-image/per-video JSONL records.
- `**meta_path**` — task-level label information (not always JSON). Detection uses `annotation_meta.json`; sequence uses a vocabulary text file (default `annotation_vocab.txt`). Override the meta filename with `task_meta_filename` when needed.

## Basic Usage

Expand a task below for a full example:

**Detection**

```python
from vdschema import AnnotationReader, AnnotationWriter, Name, TaskType

# step1: create annotation object
writer = AnnotationWriter(
    TaskType.DETECTION,
    label={
        1: Name("person", alias=["human"], prompt=["a human", "人体"]),
        2: Name("car"),
    },
)

# step2: add annotation info
writer.append(
    filename="images/sample.jpg",
    width=640,
    height=480,
    instances=[
        {"id": 0, "category_id": 1, "bbox": [10, 20, 100, 200]},
        {"id": 1, "category_id": 1, "bbox": [160, 190, 300, 560]， "is_ignored": True},
    ],
)
# step3: save annotation info, by default, written to output/{task_type}/,can override by setting task_dir in AnnotationWriter.
writer.save()

# step4(option): load annotation info.
data, label = AnnotationReader(
    TaskType.DETECTION, writer.save_dir()
).load()

print(label[1].name, label[1].alias)  # person ('human',)
print(data) # [DetectionAnnotation(filename='images/sample.jpg', width=640, height=480, instances=[Instance(id=0, category_id=1, bbox=Bbox(x1=10.0, y1=20.0, x2=100.0, y2=200.0), keypoints=None, segmentation=None, text=None)], description=None)]
```

**Keypoint**

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
            "bbox": [200, 100, 380, 420],
            "keypoints": {"body": [[210, 120, 2], [230, 140, 2]]},
        }
    ],
)
writer.save()
data, label = AnnotationReader(
    TaskType.KEYPOINT, writer.save_dir()
).load()
```



**Segmentation**

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
data, label = AnnotationReader(TaskType.SEGMENTATION, task_dir).load()
```



**Classification**

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



**Relationship**

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



**VLM**

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



**Conversation**

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



**Sequence**

Vocabulary is provided as an external file and copied to the output directory as `annotation_vocab.txt`. The file must contain **exactly one token per line** (no spaces/tabs within a line). See [assets/](assets/) for examples such as `alphanumeric_vocab.txt`. Each token in `sequences` must appear in the vocabulary file; validation runs on `append()` and on read.

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



**Action**

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



### More examples

Full runnable tests: [tests/test_annotation_format.py](tests/test_annotation_format.py). Run:

```bash
uv run pytest
```

## vdswitch

同步安装的一个标注数据转换工具，用于将以下格式的数据集，转为 vdschema 格式的标注数据集

- [x] unified-perception (up)
- [x] coco (detection / keypoint / segmentation)
- [x] yolo (detection)
- [x] imagenet (classification)

```bash
vdswitch \
  --task detection \
  --source up \
  --input-data /path/to/train.jsonl /path/to/test.jsonl \
  --input-label /path/to/label_dict.json \
  --input-root /path/to/dataset \
  --output output
```

各任务的 `--input-data` / `--input-label` 格式不同。查看说明：

```bash
vdswitch help                          # 列出任务摘要
vdswitch help --task classification    # 分类任务的文件格式与示例
vdswitch help --task detection --source up
```

| Task             | `--source up`                            | `--source coco`                                      |
| ---------------- | ---------------------------------------- | ---------------------------------------------------- |
| `detection`      | JSONL + `label_dict.json`                | COCO `instances_*.json`（`categories` 可与 data 同文件）     |
|                |                                          | YOLO `train.txt` + `classes.txt` + `labels/*.txt`        |
| `keypoint`       | —                                        | COCO instances JSON（含 `keypoints` / 类别 `keypoints` 名） |
| `segmentation`   | —                                        | COCO instances JSON（`segmentation` 多边形或 RLE）         |
| `classification` | JSONL + 多头 `label_dict.json`             | ImageNet 目录 `train/<class>/`（`--input-data`/`--input-label` 同路径） |
| `action`         | 视频 meta + kmot                         | 无标准 COCO 格式                                          |
| `sequence`       | JSONL + `vocab.txt`                      | 无标准 COCO 格式                                          |

COCO 示例：

```bash
vdswitch \
  --task detection \
  --source coco \
  --input-data /path/to/annotations/instances_train2017.json \
  --input-label /path/to/annotations/instances_train2017.json \
  --input-root /path/to/dataset \
  --output output
```


运行 `vdswitch -h` 会提示使用 `vdswitch help`；各任务完整样例见 `vdswitch help --task <name>`。

**Python API**

```python
from vdschema import AnnotationReader, Source, TaskType, switch

out = switch(
    task=TaskType.DETECTION,
    source=Source.UP,
    input_data=[
        "/path/to/meta/train.jsonl",
        "/path/to/meta/test.jsonl",
    ],
    input_label="/path/to/meta/label_dict.json",
    input_root="/path/to/dataset",  # 数据集根目录；与标注中的路径拼接，定位图片/视频绝对路径
    output="output",  # 默认值
)

data, label = AnnotationReader(TaskType.DETECTION, out).load()
```

更多用例见 [tests/test_vdswitch.py](tests/test_vdswitch.py)。


| Flag            | Values                                              | Notes                                                                                      |
| --------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `--task`        | `detection`, `classification`, `action`, `sequence` | vdschema task type                                                                         |
| `--source`      | `up`, `coco`                                        | legacy UP layout or MS COCO instances JSON                                                 |
| `--input-data`  | one or more file paths                              | annotation data file path(s)                                                               |
| `--input-label` | file path                                           | Label or vocab file; JSON or text depending on `--task` (see `vdswitch help --task …`)     |
| `--input-root`  | directory                                           | dataset root; joined with media paths in annotations to resolve absolute image/video paths |
| `--output`      | directory (default: `output`)                       | output directory; see output filename rules below                                          |


**Output filenames**

- If `--output` equals the directory of `--input-data`, write `{stem}_vdschema{suffix}` next to the source files (e.g. `train_vdschema.jsonl`, `label_dict_vdschema.json`).
- Otherwise, reuse the source names in the output directory (e.g. `train.jsonl`, `label_dict.json`).

## Development

```bash
git clone https://github.com/Arrkwen/vdschema.git
cd vdschema
uv sync --extra vdswitch --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build
```

## Publishing

1. Bump `version` in `pyproject.toml` (used at runtime via package metadata and by `uv build`).
2. Create a **Published** GitHub Release with a new tag (e.g. `0.1.0`).

The [publish.yml](.github/workflows/publish.yml) workflow runs on release publish and uploads the build to PyPI. You can also re-run it manually from Actions → **vdschema-publisher**.