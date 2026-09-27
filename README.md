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

**Detection geometry** — each instance must include at least one of:

| Format | Field | Shape |
| ------ | ----- | ----- |
| Horizontal box | `bbox` | `[x1, y1, x2, y2]` (xyxy) |
| Oriented box / closed contour | `polygon` | flat `[x1, y1, x2, y2, …]`, ≥3 vertices |
| Open polyline (e.g. lane lines) | `polyline` | flat `[x1, y1, x2, y2, …]`, ≥2 vertices |

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

Expand a task below for a full example (GitHub renders `<details>` as collapsible sections).

<details>
<summary><strong>Detection</strong></summary>

Each instance needs at least one geometry: horizontal **`bbox`**, oriented **`polygon`**, or open **`polyline`** (see table above).

<details>
<summary>Axis-aligned bbox</summary>

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

</details>

<details>
<summary>Oriented box (polygon)</summary>

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

</details>

<details>
<summary>Open polyline</summary>

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

</details>

</details>

<details>
<summary><strong>Keypoint</strong></summary>

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

</details>

<details>
<summary><strong>Segmentation</strong></summary>

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
            "rle_mask": SegmentationRLE.from_mask(mask),
        }
    ],
)
writer.save()
data, label = AnnotationReader(TaskType.SEGMENTATION, task_dir).load()
```

</details>

<details>
<summary><strong>Classification</strong></summary>

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

</details>

<details>
<summary><strong>Relationship</strong></summary>

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

</details>

<details>
<summary><strong>VLM</strong></summary>

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
data, label = AnnotationReader(
    TaskType.CONVERSATION, writer.save_dir()
).load()
```

</details>

<details>
<summary><strong>Sequence</strong></summary>

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

</details>

<details>
<summary><strong>Action</strong></summary>

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

</details>

### More examples

Full runnable tests: [tests/test_annotation_format.py](tests/test_annotation_format.py). Run:

```bash
uv run pytest
```

## vdswitch

同步安装的一个标注数据转换工具，用于将以下格式的数据集，转为 vdschema 格式的标注数据集

- [x] monolith
- [x] coco (detection / keypoint / segmentation)
- [x] yolo (detection)
- [x] imagenet (classification)
- [x] ocr (sequence)

```bash
vdswitch \
  --task detection \
  --source monolith \
  --input-data /path/to/train.jsonl /path/to/test.jsonl \
  --input-label /path/to/label_dict.json \
  --input-root /path/to/dataset \
  --output output
```

各任务的 `--input-data` / `--input-label` 格式不同。查看说明：

```bash
vdswitch help                          # 列出任务摘要
vdswitch help --task classification    # 分类任务的文件格式与示例
vdswitch help --task detection --source monolith
```

| Task             | `--source monolith`                      | other sources                                        |
| ---------------- | ---------------------------------------- | ---------------------------------------------------- |
| `detection`      | JSONL + `label_dict.json`                | COCO `instances_*.json`（类别在 JSON 内，可省略 `--input-label`；标注里已是绝对路径时可省略 `--input-root`）     |
|                |                                          | YOLO `train.txt` + `classes.txt` + `labels/*.txt`        |
| `keypoint`       | —                                        | COCO instances JSON（含 `keypoints` / 类别 `keypoints` 名） |
| `segmentation`   | —                                        | COCO instances JSON（`segmentation` 多边形或 RLE）         |
| `classification` | JSONL + 多头 `label_dict.json`             | ImageNet 目录 `train/<class>/`（可只传 `--input-data`，省略 `--input-label` / `--input-root`） |
| `action`         | 视频 meta + kmot                         | 无标准 COCO 格式                                          |
| `sequence`       | JSONL + `vocab.txt`                      | OCR 清单 `anno.txt`（路径 TAB 文本）+ `vocab.txt`              |

COCO 示例：

```bash
vdswitch \
  --task detection \
  --source coco \
  --input-data /path/to/annotations/instances_train2017.json \
  --output output
```

（`vdswitch help --task detection --source coco` 中的示例与上述最小参数一致；相对路径图片时需再加 `--input-root`。）


运行 `vdswitch -h` 会提示使用 `vdswitch help`；各任务完整样例见 `vdswitch help --task <name>`。

**Python API**

```python
from vdschema import AnnotationReader, Source, TaskType, switch

out = switch(
    task=TaskType.DETECTION,
    source=Source.MONOLITH,
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
| `--source`      | `monolith`, `coco`, `yolo`, `imagenet`, `ocr`       | legacy monolith layout or third-party dataset formats                                      |
| `--input-data`  | one or more file paths                              | annotation data file path(s)                                                               |
| `--input-label` | file path (optional for some `--source`)            | Label or vocab file; omit when native format embeds labels (COCO JSON, ImageNet layout); see `vdswitch help` |
| `--input-root`  | directory (optional for some `--source`)            | dataset root; omit when media paths in annotations are already absolute; see `vdswitch help` |
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