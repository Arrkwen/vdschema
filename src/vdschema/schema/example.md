# Annotation format examples

Each task stores one JSON object per line in `annotation_data.jsonl`. When a task uses a label vocabulary, `annotation_meta.json` is saved in the same directory (sequence tasks use `annotation_vocab.txt` instead).

`bbox` is `[x1, y1, x2, y2]` (xyxy). `keypoints` entries are `[x, y, visibility]` (`visibility`: 0/1/2). `polygon` is a flat closed ring or oriented box `[x1,y1,...]` (≥3 vertices). `polyline` is an open line `[x1,y1,...]` (≥2 vertices). `point` is `[x, y]` for single-point detection. `rle_mask` follows pycocotools RLE (`size`: `[height, width]`).

---

## detection

**annotation_meta.json**

```json
{
  "annotation_schema_ref": "https://github.com/Arrkwen/vdschema/blob/main/src/vdschema/schema/annotation_data.json",
  "detection": [
    {
      "category_id": 1,
      "category_name": "person",
      "category_alias": ["human"],
      "category_prompt": ["a human", "人体"]
    },
    {"category_id": 2, "category_name": "car"},
    {"category_id": 3, "category_name": "ship"},
    {"category_id": 4, "category_name": "lane_line"},
    {"category_id": 5, "category_name": "defect"}
  ]
}
```

**annotation_data.jsonl** — axis-aligned `bbox` (one line)

```json
{
  "filename": "images/detection_001.jpg",
  "width": 640,
  "height": 480,
  "instances": [
    {"id": 0, "category_id": 1, "bbox": [10, 20, 100, 200]},
    {"id": 1, "category_id": 2, "bbox": [120, 30, 200, 180]}
  ]
}
```

**annotation_data.jsonl** — oriented box (`polygon`, one line)

```json
{
  "filename": "images/detection_obb_001.jpg",
  "width": 640,
  "height": 480,
  "instances": [
    {
      "id": 0,
      "category_id": 3,
      "polygon": [120, 80, 200, 60, 220, 140, 140, 160]
    }
  ]
}
```

**annotation_data.jsonl** — open `polyline` (e.g. lane line, one line)

```json
{
  "filename": "images/detection_lane_001.jpg",
  "width": 1280,
  "height": 720,
  "instances": [
    {
      "id": 0,
      "category_id": 4,
      "polyline": [100, 650, 280, 520, 460, 410, 640, 340]
    }
  ]
}
```

**annotation_data.jsonl** — `point` (single-point detection, one line)

```json
{
  "filename": "images/detection_point_001.jpg",
  "width": 640,
  "height": 480,
  "instances": [
    {"id": 0, "category_id": 5, "point": [120.5, 340.0]},
    {"id": 1, "category_id": 5, "point": [400, 200]}
  ]
}
```

---

## keypoint

**annotation_meta.json** — same structure as detection.

**annotation_data.jsonl**

```json
{
  "filename": "images/keypoint_001.jpg",
  "width": 640,
  "height": 480,
  "instances": [
    {
      "id": 0,
      "category_id": 1,
      "bbox": [200, 100, 380, 420],
      "keypoints": {
        "body": [[210, 120, 2], [230, 140, 2]]
      }
    }
  ]
}
```

---

## segmentation

**annotation_meta.json** — same structure as detection.

**annotation_data.jsonl**

```json
{
  "filename": "images/segmentation_001.jpg",
  "width": 640,
  "height": 480,
  "instances": [
    {
      "id": 0,
      "category_id": 1,
      "bbox": [10, 20, 100, 200],
      "rle_mask": {"size": [480, 640], "counts": "eNq..."}
    }
  ]
}
```

---

## classification

**annotation_meta.json**

```json
{
  "annotation_schema_ref": "https://github.com/Arrkwen/vdschema/blob/main/src/vdschema/schema/annotation_data.json",
  "classification": [
    {
      "category_attr": "scene",
      "category_label": [
        {"category_id": 10, "category_name": "intersection"},
        {"category_id": 11, "category_name": "highway"}
      ]
    },
    {
      "category_attr": "weather",
      "category_label": [
        {"category_id": 3, "category_name": "sunny"},
        {"category_id": 4, "category_name": "cloudy"}
      ]
    }
  ]
}
```

**annotation_data.jsonl**

```json
{
  "filename": "images/classification_001.jpg",
  "width": 640,
  "height": 480,
  "categories": [
    {"category_attr": "scene", "category_ids": [10]},
    {"category_attr": "weather", "category_ids": [3, 4]}
  ]
}
```

---

## relationship

**annotation_meta.json**

```json
{
  "annotation_schema_ref": "https://github.com/Arrkwen/vdschema/blob/main/src/vdschema/schema/annotation_data.json",
  "detection": [
    {"category_id": 1, "category_name": "person"},
    {"category_id": 2, "category_name": "car"}
  ],
  "relationship": [
    {"relationship_id": 0, "relationship_name": "near"},
    {"relationship_id": 1, "relationship_name": "left_of"}
  ]
}
```

**annotation_data.jsonl**

```json
{
  "filename": "images/relationship_001.jpg",
  "width": 640,
  "height": 480,
  "instances": [
    {"id": 0, "category_id": 1, "bbox": [10, 20, 100, 200]},
    {"id": 1, "category_id": 2, "bbox": [120, 30, 200, 180]}
  ],
  "relationships": [
    {"subject_id": 0, "object_id": 1, "relation_type": "near"}
  ]
}
```

---

## vlm

No `annotation_meta.json`.

**annotation_data.jsonl**

```json
{
  "filename": "images/vlm_001.jpg",
  "width": 640,
  "height": 480,
  "description": "A street intersection with three people crossing."
}
```

---

## conversation

No `annotation_meta.json`.

**annotation_data.jsonl**

```json
{
  "filename": "images/conversation_001.jpg",
  "width": 640,
  "height": 480,
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "image", "filename": "images/conversation_001.jpg"},
        {"type": "text", "text": "Describe the image."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "A street scene."}]
    }
  ]
}
```

---

## sequence

**annotation_vocab.txt** (exactly one token per line; copied from the Writer `label` path on save). Example vocab files: [assets/alphanumeric_vocab.txt](assets/alphanumeric_vocab.txt), [assets/digits_vocab.txt](assets/digits_vocab.txt), [assets/letters_vocab.txt](assets/letters_vocab.txt), [assets/province_vocab.txt](assets/province_vocab.txt).

```
0
1
B
P
D
V
7
```

**annotation_data.jsonl**

```json
{
  "filename": "batch1_crop_plate/27993412_car0_inst0.jpg",
  "width": 224,
  "height": 128,
  "sequences": ["B", "1", "0", "7", "7", "P", "D", "V"]
}
```

---

## action

**annotation_meta.json**

```json
{
  "annotation_schema_ref": "https://github.com/Arrkwen/vdschema/blob/main/src/vdschema/schema/annotation_data.json",
  "action": [
    {"category_id": 2, "category_name": "fall"},
    {"category_id": 4, "category_name": "walk"}
  ]
}
```

**annotation_data.jsonl**

```json
{
  "filename": "videos/action_001.mp4",
  "width": 1920,
  "height": 1080,
  "description": "An elderly person falls down.",
  "actions": [
    {
      "category_id": 2,
      "track_id": 0,
      "start_idx": 4,
      "end_idx": 6,
      "description": "An elderly person falls down.",
      "tracks": [
        {"frame_idx": 0, "bbox": [520, 300, 620, 380]},
        {"frame_idx": 4, "bbox": [552, 308, 652, 388]}
      ]
    }
  ]
}
```

