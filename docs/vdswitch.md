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
  --input-data /path/to/train.jsonl /path/to/test.jsonl \
  --input-label /path/to/label_dict.json \
  --input-root /path/to/dataset \
  --output output
```

Per-task input formats differ. Use built-in help:

```bash
vdswitch help                          # task summary
vdswitch help --task classification    # classification file layout
vdswitch help --task detection --source monolith
```

Running `vdswitch -h` points to `vdswitch help`. Full examples: `vdswitch help --task <name>`.

## Task × source matrix


| Task             | `--source monolith`          | other sources                                                                        |
| ---------------- | ---------------------------- | ------------------------------------------------------------------------------------ |
| `detection`      | JSONL + `label_dict.json`    | COCO `instances_*.json`（类别在 JSON 内，可省略 `--input-label`；标注里已是绝对路径时可省略 `--input-root`） |
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
| `classification` | JSONL + 多头 `label_dict.json` | ImageNet 目录 `train/<class>/`（可只传 `--input-data`，省略 `--input-label` / `--input-root`） |
|                  |                              | LabelBee `tagTool` JSON 目录                                                           |
| `action`         | 视频 meta + kmot               | 无标准 COCO 格式                                                                          |
| `sequence`       | JSONL + `vocab.txt`          | OCR 清单 `anno.txt`（路径 TAB 文本）+ `vocab.txt`                                            |


### COCO example

```bash
vdswitch \
  --task detection \
  --source coco \
  --input-data /path/to/annotations/instances_train2017.json \
  --output output
```

(`vdswitch help --task detection --source coco` matches this minimal invocation; add `--input-root` when image paths in JSON are relative.)

## CLI flags


| Flag            | Values                                                                      | Notes                                                                                                        |
| --------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--task`        | `detection`, `classification`, `action`, `sequence`, …                      | vdschema task type (see `vdswitch help`)                                                                     |
| `--source`      | `monolith`, `coco`, `yolo`, `imagenet`, `ocr`, `labelbee`, `labelme`, `voc` | legacy monolith layout or third-party dataset formats                                                        |
| `--input-data`  | one or more file paths                                                      | annotation data file path(s)                                                                                 |
| `--input-label` | file path (optional for some `--source`)                                    | Label or vocab file; omit when native format embeds labels (COCO JSON, ImageNet layout); see `vdswitch help` |
| `--input-root`  | directory (optional for some `--source`)                                    | dataset root; omit when media paths in annotations are already absolute; see `vdswitch help`                 |
| `--output`      | directory (default: `output`)                                               | output directory; see output filename rules below                                                            |


## Output filenames

- If `--output` equals the directory of `--input-data`, write `{stem}_vdschema{suffix}` next to the source files (e.g. `train_vdschema.jsonl`, `label_dict_vdschema.json`).
- Otherwise, reuse the source names in the output directory (e.g. `train.jsonl`, `label_dict.json`).

## Python API

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
    input_root="/path/to/dataset",
    output="output",
)

data, label = AnnotationReader(TaskType.DETECTION, out).load()
```

More tests: [tests/test_vdswitch.py](../tests/test_vdswitch.py).
