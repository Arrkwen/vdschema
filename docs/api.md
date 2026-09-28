# Public API reference

[← README](../README.md) · [Usage guide](vdschema.md) · [vdswitch](vdswitch.md) · [JSON examples](example.md)

This page documents symbols exported from `vdschema` (`from vdschema import …`). Install `vdschema[vdswitch]` for `switch` and `Source`.

## JSONL I/O

### `AnnotationWriter`

Writes `annotation_data.jsonl` and label meta (when the task uses a vocabulary).

```python
AnnotationWriter(
    task_type: TaskType,
    label: dict | str | Path | None = None,
    task_dir: str | Path = "output",
    *,
    task_data_filename: str | None = None,
    task_meta_filename: str | None = None,
)
```


| Parameter            | Description                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `task_type`          | Task kind; default `task_dir` becomes `output/{task_type.value}/` when `task_dir=="output"`.                              |
| `label`              | Label dictionary input. Shape depends on task (see table below). Omit or `None` only when `not task_type.has_label_dict`. |
| `task_dir`           | Output directory root.                                                                                                    |
| `task_data_filename` | JSONL filename (default `annotation_data.jsonl`).                                                                         |
| `task_meta_filename` | Meta filename (default `annotation_meta.json`, or `annotation_vocab.txt` for sequence).                                   |


`label` **shapes by task**


| Task                                         | `label` type                                 | Example                                                |
| -------------------------------------------- | -------------------------------------------- | ------------------------------------------------------ |
| Detection / keypoint / segmentation / action | `dict[int, Name                              | str]`                                                  |
| Classification                               | `dict[str, dict[int, Name                    | str]]`                                                 |
| Relationship                                 | `dict` with keys `detection`, `relationship` | `{"detection": {1: "a"}, "relationship": {0: "near"}}` |
| Sequence                                     | `str                                         | Path` to vocab file                                    |


**Methods**


| Method                          | Description                                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------------------- |
| `append(**kwargs)`              | Build one record from keyword args matching the task’s annotation fields, validate, append. |
| `append_annotation(annotation)` | Append an existing annotation instance.                                                     |
| `save()`                        | Flush JSONL; write label meta when applicable.                                              |
| `save_dir()` → `Path`           | Resolved output directory.                                                                  |
| `data_path` / `meta_path`       | Paths to JSONL and meta/vocab files.                                                        |


### `AnnotationReader`

Reads and validates JSONL under `task_dir`.

```python
AnnotationReader(
    task_type: TaskType,
    task_dir: str | Path,
    *,
    task_data_filename: str | None = None,
    task_meta_filename: str | None = None,
)
```


| Parameter            | Description                                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `task_type`          | How to parse each JSON line (may differ from the task used when writing the same folder, e.g. manifest → detection). |
| `task_dir`           | Directory containing data (and optional meta). Same `output/{task}` resolution as writer when path is `output`.      |
| `task_data_filename` | JSONL filename override.                                                                                             |
| `task_meta_filename` | Meta / vocab filename override.                                                                                      |


**Methods**


| Method                    | Returns                       | Description                                                        |
| ------------------------- | ----------------------------- | ------------------------------------------------------------------ |
| `load()`                  | `(list[BaseAnnotation], label | None)`                                                             |
| `iter_raw()`              | `Iterator[dict]`              | Raw JSON objects per line.                                         |
| `iter_annotations()`      | `Iterator[BaseAnnotation]`    | Parsed and validated records.                                      |
| `validate()`              | `bool`                        | `True` if `load()` would succeed; `False` on format or I/O errors. |
| `data_path` / `meta_path` | `Path`                        | File paths.                                                        |


---

## Label names

### `Name`

Canonical category / label name with optional aliases and prompts (stored in meta JSON).

```python
Name(name: str, *, alias: Sequence[str] | None = None, prompt: Sequence[str] | None = None)
```


| Parameter | Type                        | Description                               |
| --------- | --------------------------- | ----------------------------------------- |
| `name`    | `str`                       | Primary name (non-empty).                 |
| `alias`   | sequence of `str`, optional | Alternate strings mapping to the same id. |
| `prompt`  | sequence of `str`, optional | Optional prompt strings for training.     |


---

## Enums

### `TaskType`

String enum. Selects record type, on-disk layout, and label meta handling.


| Member           | Value            | `annotation_class`         | Requires `label` on write |
| ---------------- | ---------------- | -------------------------- | ------------------------- |
| `MANIFEST`       | `manifest`       | `ManifestAnnotation`       | No                        |
| `DETECTION`      | `detection`      | `DetectionAnnotation`      | Yes                       |
| `KEYPOINT`       | `keypoint`       | `KeypointAnnotation`       | Yes                       |
| `SEGMENTATION`   | `segmentation`   | `SegmentationAnnotation`   | Yes                       |
| `CLASSIFICATION` | `classification` | `ClassificationAnnotation` | Yes                       |
| `RELATIONSHIP`   | `relationship`   | `RelationshipAnnotation`   | Yes                       |
| `VLM`            | `vlm`            | `VlmAnnotation`            | No                        |
| `CONVERSATION`   | `conversation`   | `ConversationAnnotation`   | No                        |
| `SEQUENCE`       | `sequence`       | `SequenceAnnotation`       | Yes (vocab file path)     |
| `ACTION`         | `action`         | `ActionAnnotation`         | Yes                       |


**Properties**

- `annotation_class` → concrete `BaseAnnotation` subclass for this task.
- `has_label_dict` → `False` for manifest, VLM, and conversation (no label meta file on save).

### `ConversationRole`


| Member      | Value       |
| ----------- | ----------- |
| `USER`      | `user`      |
| `ASSISTANT` | `assistant` |
| `SYSTEM`    | `system`    |


---

## Geometry and instance helpers

### `Bbox`

Axis-aligned box **xyxy** `[x1, y1, x2, y2]`.


| Constructor               | Arguments                 |
| ------------------------- | ------------------------- |
| `from_list` / `from_xyxy` | 4 numbers                 |
| `from_xywh`               | `[x, y, width, height]`   |
| `from_cxcywh`             | `[cx, cy, width, height]` |


`to_list()` → xyxy list.

### `Keypoint`

Single point `**[x, y, visibility]`**; visibility `0` / `1` / `2` (COCO convention).


| Field        | Type                |
| ------------ | ------------------- |
| `x`, `y`     | `float`             |
| `visibility` | `int` (default `2`) |


Instance `keypoints` in JSON is a **flat** list `[x1,y1,v1, x2,y2,v2, …]`.

### `SegmentationRLE`


| Field             | Description            |
| ----------------- | ---------------------- |
| `height`, `width` | Mask size              |
| `counts`          | pycocotools RLE string |



| Method                  | Description               |
| ----------------------- | ------------------------- |
| `from_mask(ndarray)`    | Encode uint8 `[H,W]` mask |
| `to_mask()`             | Decode to numpy array     |
| `from_dict` / `to_dict` | JSON RLE object           |


### `ConversationTurn`

One dialog turn.

```python
ConversationTurn(
    role: ConversationRole | str,
    *,
    image: str | ContentImage | None = None,
    text: str | ContentText | None = None,
    content: list[ContentPart] | None = None,
)
```

Provide `content` **or** `image` / `text` shorthand (appended to `content`).
