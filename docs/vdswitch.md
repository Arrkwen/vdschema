# vdswitch usage

[← Back to README](../README.md) ·  [vdschema writer/reader guide](vdschema.md)

**vdswitch** ships with the optional `vdschema[vdswitch]` extra. It converts common vision dataset formats into vdschema JSONL (and label meta when applicable). CLI: `vdswitch`; Python: `switch()`. API: [api.md](api.md).

Install:

```bash
pip install "vdschema[vdswitch]"
```

## Supported sources

- [x] monolith
- [x] coco (detection / keypoint / segmentation)
- [x] yolo (detection)
- [x] imagenet (classification)
- [x] ocr (sequence)
- [x] labelbee (detection / segmentation / keypoint / classification; [format docs](https://github.com/open-mmlab/labelbee-client/tree/main/docs/annotation))
- [x] labelme (detection / segmentation / keypoint; [LabelMe JSON](https://github.com/wkentaro/labelme))
- [x] voc (detection XML + SegmentationClass PNG; PASCAL VOC devkit)

## CLI quick start

```bash
vdswitch \
  --task detection \
  --source monolith \
  --input /path/to/train.jsonl /path/to/test.jsonl \
  --option category=/path/to/label_dict.json \
  --option root=/path/to/dataset \
  --output output
```

Per-task input formats differ. Use built-in help:

```bash
vdswitch help                          # task summary
vdswitch help --task classification    # classification file layout
vdswitch help --task detection --source monolith
```

Running `vdswitch -h` points to `vdswitch help`. Full examples: `vdswitch help --task <name>`.

### Converter options (`--option`)

Repeat **`--option KEY=VALUE`**. Allowed keys are defined on each converter class (see `vdswitch help --task … --source …`); unknown keys are rejected for that source.

Examples:

| Key | Typical sources | Meaning |
| --- | --- | --- |
| `category` | monolith, YOLO, OCR | Label or vocab file (required where listed). |
| `root` | monolith, YOLO, LabelMe, … | Dataset root for relative media paths. |
| `category_id_contiguous` | COCO det/kpt/seg | `1` remaps category ids to contiguous ids (default `0`). |
| `category_id_start` | COCO | First id when remapping (default `1`). |

```bash
vdswitch --task detection --source coco \
  --input instances.json --output out \
  --option category_id_contiguous=1 \
  --option category_id_start=1
```

## Task × source matrix


| Task             | `--source monolith`          | other sources                                                                        |
| ---------------- | ---------------------------- | ------------------------------------------------------------------------------------ |
| `detection`      | JSONL + `label_dict.json`    | COCO `instances_*.json`（类别在 JSON 内，无需 path options） |
|                  |                              | YOLO `train.txt` + `classes.txt` + `labels/*.txt`                                    |
|                  |                              | LabelBee `**/*.json` General Data（`rectTool` / `lineTool` / `polygonTool`）           |
|                  |                              | LabelMe `*.json`（`rectangle` / `polygon` / `linestrip` / `point` / `circle`→bbox）    |
|                  |                              | VOC `ImageSets/Main/*.txt` + `Annotations/*.xml`（检测）                                 |
| `keypoint`       | —                            | COCO instances JSON（含 flat `keypoints`）                                              |
|                  |                              | LabelBee `pointTool` JSON 目录                                                         |
|                  |                              | LabelMe `point` shapes JSON 目录                                                       |
| `segmentation`   | —                            | COCO instances JSON（`segmentation` 多边形或 RLE）                                         |
|                  |                              | LabelBee `polygonTool` JSON 目录                                                       |
|                  |                              | LabelMe `polygon` shapes JSON 目录                                                     |
|                  |                              | VOC `SegmentationClass/*.png`（每类一个 RLE 实例，默认 VOC2012 20 类）                           |
| `classification` | JSONL + 多头 `label_dict.json` | ImageNet 目录 `train/<class>/`（通常只需 `--input`） |
|                  |                              | LabelBee `tagTool` JSON 目录                                                           |
| `action`         | 视频 meta + kmot               | 无标准 COCO 格式                                                                          |
| `sequence`       | JSONL + `vocab.txt`          | OCR 清单 `anno.txt`（路径 TAB 文本）+ `vocab.txt`                                            |


### COCO example

```bash
vdswitch \
  --task detection \
  --source coco \
  --input /path/to/annotations/instances_train2017.json \
  --output output
```

(`vdswitch help --task detection --source coco` lists COCO-only tuning options.)

## CLI flags


| Flag           | Values                                                                      | Notes                                   |
| -------------- | --------------------------------------------------------------------------- | --------------------------------------- |
| `--task`       | `detection`, `classification`, `action`, `sequence`, …                      | vdschema task type                      |
| `--source`     | `monolith`, `coco`, `yolo`, …                                               | dataset format                          |
| `--input` | one or more paths                                                           | annotation data                         |
| `--option`     | `KEY=VALUE` (repeatable)                                                    | per-source; see `vdswitch help`         |
| `--output`     | directory (default: `output`)                                               | output directory                        |


## Output filenames

- If `--output` equals the directory of `--input`, write `{stem}_vdschema{suffix}` next to the source files (e.g. `train_vdschema.jsonl`, `label_dict_vdschema.json`).
- Otherwise, reuse the source names in the output directory (e.g. `train.jsonl`, `label_dict.json`).

## Python API

```python
from vdschema import AnnotationReader, Source, TaskType, switch

out = switch(
    task=TaskType.DETECTION,
    source=Source.MONOLITH,
    input=[
        "/path/to/meta/train.jsonl",
        "/path/to/meta/test.jsonl",
    ],
    options={
        "category": "/path/to/meta/label_dict.json",
        "root": "/path/to/dataset",
    },
    output="output",
)

data, label = AnnotationReader(TaskType.DETECTION, out).load()
```

More tests: [tests/test_vdswitch.py](../tests/test_vdswitch.py).
