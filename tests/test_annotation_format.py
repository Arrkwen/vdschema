from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from vdschema.annotation_dict import ANNOTATION_VOCAB_FILENAME
from vdschema import (
    ActionAnnotation,
    AnnotationFormatError,
    AnnotationReader,
    AnnotationWriter,
    BaseAnnotation,
    Bbox,
    Name,
    ClassificationAnnotation,
    ConversationAnnotation,
    ConversationRole,
    ConversationTurn,
    DetectionAnnotation,
    KeypointAnnotation,
    RelationshipAnnotation,
    SegmentationRLE,
    SegmentationAnnotation,
    SequenceAnnotation,
    TaskType,
    VlmAnnotation,
)

TAnnotation = TypeVar("TAnnotation", bound=BaseAnnotation)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ALPHANUMERIC_VOCAB = ASSETS_DIR / "alphanumeric_vocab.txt"


def _read_annotations(
    writer: AnnotationWriter,
    *,
    expected_count: int,
) -> tuple[list[TAnnotation], dict | None, list[dict[str, Any]]]:
    """Exercise AnnotationReader.iter_raw / iter_annotations / load / validate."""
    reader = AnnotationReader(
        writer.task_type,
        writer.save_dir(),
        task_data_filename=writer.task_data_filename,
        task_meta_filename=writer.task_meta_filename,
    )
    raw_rows = list(reader.iter_raw())
    assert len(raw_rows) == expected_count
    via_iter = list(reader.iter_annotations())
    assert len(via_iter) == expected_count
    data, label = reader.load()
    assert len(data) == expected_count
    assert data == via_iter
    assert reader.validate() is True
    return data, label, raw_rows


def _mask(height: int = 480, width: int = 640) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[20:80, 30:120] = 1
    return mask


def test_detection_annotation_format():
    writer = AnnotationWriter(
        TaskType.DETECTION,
        label={
            1: Name("person", alias=["human"], prompt=["a human", "人体"]),
            2: Name("car"),
            3: Name("bus"),
            4: Name("truck"),
        },
    )

    # bbox convert from multiple formats, the default list parameter is xyxy format
    writer.append(
        filename="images/detection_001.jpg",
        width=640,
        height=480,
        instances=[
            {"id": 0, "category_id": 1, "bbox": [10, 20, 100, 200]},
            {
                "id": 1,
                "category_id": 1,
                "bbox": Bbox.from_xyxy([120, 30, 200, 180]),
                "is_ignored": True,
            },
            {"id": 2, "category_id": 2, "bbox": Bbox.from_xywh([120, 30, 80, 150])},
            {"id": 3, "category_id": 4, "bbox": Bbox.from_cxcywh([160, 105, 80, 150])},
        ],
    )
    writer.append(
        filename="images/detection_002.jpg",
        width=800,
        height=600,
        instances=[
            {"id": 0, "category_id": 3, "bbox": [50, 60, 180, 260], "text": "AB0000EF"},
            {
                "id": 1,
                "category_id": 4,
                "bbox": Bbox.from_cxcywh([290, 190, 140, 220]),
                "text": "AB0000FF"
            },
        ],
    )
    writer.append(
        filename="images/detection_002.jpg",
        width=800,
        height=600,
        instances=[],
    )
    writer.save()

    data, label, rows = _read_annotations(writer, expected_count=3)
    assert "task_type" not in rows[0]
    assert "schema_version" not in rows[0]
    assert rows[0]["instances"][1]["bbox"] == [120.0, 30.0, 200.0, 180.0]
    assert rows[0]["instances"][2]["bbox"] == [120.0, 30.0, 200.0, 180.0]
    assert rows[0]["instances"][3]["bbox"] == [120.0, 30.0, 200.0, 180.0]
    assert rows[1]["instances"][1]["bbox"] == [220.0, 80.0, 360.0, 300.0]
    assert isinstance(data[0], DetectionAnnotation)
    assert data[0].instances[1].bbox.to_list() == [120.0, 30.0, 200.0, 180.0]
    assert data[0].instances[0].is_ignored is False
    assert data[0].instances[1].is_ignored is True
    assert rows[0]["instances"][0]["is_ignored"] is False
    assert rows[0]["instances"][1]["is_ignored"] is True
    assert data[1].instances[1].text == "AB0000FF"
    assert data[2].instances == []
    assert label[1].name == "person"
    assert label[1].alias == ("human",)
    assert label[1].prompt == ("a human", "人体")
    assert label[2].name == "car"
    assert label[2].alias == ()
    assert label[4].name == "truck"
    meta = __import__("json").loads(writer.meta_path.read_text(encoding="utf-8"))
    person = next(item for item in meta["detection"] if item["category_id"] == 1)
    assert person["category_alias"] == ["human"]
    assert person["category_prompt"] == ["a human", "人体"]
    car = next(item for item in meta["detection"] if item["category_id"] == 2)
    assert "category_alias" not in car
    assert "category_prompt" not in car


def test_keypoint_annotation_format():
    writer = AnnotationWriter(
        TaskType.KEYPOINT,
        label={1: "person", 2: "car", 3: "bus", 4: "truck"},
    )

    # bbox convert from multiple formats, the default list parameter is xyxy format
    writer.append(
        filename="images/keypoint_001.jpg",
        width=640,
        height=480,
        instances=[
            {
                "id": 0,
                "category_id": 1,
                "bbox": [200, 100, 380, 420],
                "keypoints": {
                    "body": [
                        [210, 120, 2],
                        [230, 140, 2],
                    ]
                },
                "text": "body keypoints",
            }
        ],
    )
    writer.append(
        filename="images/keypoint_002.jpg",
        width=640,
        height=480,
        instances=[
            {
                "id": 1,
                "category_id": 1,
                "bbox": [100, 100, 220, 360],
                "keypoints": {"face": [[110, 120, 2], [130, 120, 2]]},
            }
        ],
    )
    writer.save()

    data, _, rows = _read_annotations(writer, expected_count=2)
    assert "task_type" not in rows[0]
    assert "schema_version" not in rows[0]
    assert rows[0]["instances"][0]["keypoints"]["body"] == [
        [210, 120, 2],
        [230.0, 140.0, 2],
    ]
    assert rows[1]["instances"][0]["keypoints"]["face"][1] == [130.0, 120.0, 2]
    assert isinstance(data[0], KeypointAnnotation)
    assert data[0].instances[0].keypoints["body"][0].visibility == 2
    assert data[0].instances[0].text == "body keypoints"
    assert list(data[1].instances[0].keypoints["face"][1].to_list()) == [
        130.0,
        120.0,
        2,
    ]


def test_segmentation_annotation_format():
    writer = AnnotationWriter(
        TaskType.SEGMENTATION,
        label={1: "person", 2: "car", 3: "bus", 4: "truck"},
    )
    seg_a = SegmentationRLE.from_mask(_mask())
    seg_b = SegmentationRLE.from_mask(_mask())

    # bbox convert from multiple formats, the default list parameter is xyxy format
    writer.append(
        filename="images/segmentation_001.jpg",
        width=640,
        height=480,
        instances=[
            {
                "id": 0,
                "category_id": 1,
                "bbox": [10, 20, 100, 200],
                "segmentation": seg_a,
            }
        ],
    )
    writer.append(
        filename="images/segmentation_002.jpg",
        width=640,
        height=480,
        instances=[
            {
                "id": 1,
                "category_id": 1,
                "bbox": Bbox.from_xywh([120, 30, 200, 180]),
                "segmentation": seg_b,
            }
        ],
    )
    writer.append(
        filename="images/segmentation_003.jpg",
        width=640,
        height=480,
        instances=[],
    )
    writer.save()

    data, _, rows = _read_annotations(writer, expected_count=3)
    assert "task_type" not in rows[0]
    assert "schema_version" not in rows[0]
    assert rows[0]["instances"][0]["segmentation"]["size"] == [480, 640]
    assert rows[1]["instances"][0]["segmentation"]["counts"]
    assert isinstance(data[0], SegmentationAnnotation)
    assert data[0].instances[0].segmentation.to_mask().shape == (480, 640)
    assert data[2].instances == []


def test_classification_annotation_format():
    writer = AnnotationWriter(
        TaskType.CLASSIFICATION,
        label={
            "hair_color": {1: "black", 2: "brown", 3: "blonde", 4: "red"},
            "age": {1: "young", 2: "middle-aged", 3: "elderly"},
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
    writer.append(
        filename="images/classification_002.jpg",
        width=800,
        height=600,
        categories=[{"category_attr": "age", "category_ids": [2]}],
    )
    writer.save()

    data, label, rows = _read_annotations(writer, expected_count=2)
    assert "task_type" not in rows[0]
    assert "schema_version" not in rows[0]
    assert rows[0]["categories"][1]["category_ids"] == [1]
    assert rows[1]["categories"][0]["category_ids"] == [2]
    assert isinstance(data[0], ClassificationAnnotation)
    assert data[0].categories[0].category_attr == "hair_color"
    assert data[0].categories[1].category_ids == [1]
    assert data[1].categories[0].category_ids == [2]

    assert label["hair_color"][1].name == "black"
    assert label["age"][2].name == "middle-aged"


def test_relationship_annotation_format():
    writer = AnnotationWriter(
        TaskType.RELATIONSHIP,
        label={
            "detection": {1: "person", 2: "car"},
            "relationship": {0: "near", 1: "left_of"},
        },
    )

    # bbox convert from multiple formats, the default list parameter is xyxy format
    writer.append(
        filename="images/relationship_001.jpg",
        width=640,
        height=480,
        instances=[
            {"id": 0, "category_id": 1, "bbox": [10, 20, 100, 200]},
            {"id": 1, "category_id": 2, "bbox": [120, 30, 200, 180]},
        ],
        relationships=[
            {"subject_id": 0, "object_id": 1, "relation_type": "near"}
        ],
    )
    writer.append(
        filename="images/relationship_002.jpg",
        width=640,
        height=480,
        instances=[
            {"id": 0, "category_id": 1, "bbox": [20, 20, 110, 210]},
            {"id": 1, "category_id": 2, "bbox": [140, 40, 240, 190]},
        ],
        relationships=[
            {"subject_id": 1, "object_id": 0, "relation_type": "left_of"}
        ],
    )
    writer.append(
        filename="images/relationship_003.jpg",
        width=640,
        height=480,
        instances=[],
        relationships=[],
    )
    writer.save()

    data, _, rows = _read_annotations(writer, expected_count=3)
    assert "task_type" not in rows[0]
    assert "schema_version" not in rows[0]
    assert rows[0]["relationships"][0]["relation_type"] == "near"
    assert rows[1]["relationships"][0]["relation_type"] == "left_of"
    assert isinstance(data[0], RelationshipAnnotation)
    assert data[0].relationships[0].relation_type == "near"
    assert data[1].relationships[0].relation_type == "left_of"
    assert data[2].instances == []
    assert data[2].relationships == []


def test_vlm_annotation_format():
    writer = AnnotationWriter(TaskType.VLM)

    writer.append(
        filename="images/vlm_001.jpg",
        width=640,
        height=480,
        description="A street intersection with three people crossing.",
    )
    writer.append(
        filename="images/vlm_002.jpg",
        width=800,
        height=600,
        description="A computer is on an indoor desk.",
    )
    writer.save()

    data, label, rows = _read_annotations(writer, expected_count=2)
    assert "task_type" not in rows[0]
    assert "schema_version" not in rows[0]
    assert rows[0]["description"].startswith("A street")
    assert rows[1]["description"].startswith("A computer")
    assert isinstance(data[0], VlmAnnotation)
    assert data[0].description.startswith("A street")
    assert data[1].width == 800
    assert label is None


def test_conversation_annotation_format():
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
            ConversationTurn(
                role=ConversationRole.ASSISTANT,
                text="A street scene.",
            ),
        ],
    )
    writer.append(
        filename="images/conversation_002.jpg",
        width=640,
        height=480,
        conversations=[
            ConversationTurn(
                role=ConversationRole.USER,
                image="images/conversation_002.jpg",
                text="Describe the image.",
            ),
            ConversationTurn(
                role=ConversationRole.ASSISTANT,
                text="A computer is on an indoor desk.",
            ),
        ],
    )
    writer.save()

    data, label, rows = _read_annotations(writer, expected_count=2)
    assert "task_type" not in rows[0]
    assert "schema_version" not in rows[0]
    assert rows[0]["conversations"][0]["content"][1]["type"] == "text"
    assert rows[1]["conversations"][1]["role"] == ConversationRole.ASSISTANT.value
    assert isinstance(data[0], ConversationAnnotation)
    assert data[0].conversations[0].role == ConversationRole.USER
    assert data[1].conversations[1].content[0].text.startswith("A computer")
    assert label is None


def test_action_annotation_format():
    writer = AnnotationWriter(
        TaskType.ACTION,
        label={2: "fall", 4: "walk", 5: "stand"},
    )

    # bbox convert from multiple formats, the default list parameter is xyxy format
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
                "tracks": [
                    {
                        "frame_idx": 0,
                        "bbox": [520, 300, 620, 380],
                        "segmentation": SegmentationRLE.from_mask(_mask(1080, 1920)),
                    },
                    {
                        "frame_idx": 1,
                        "bbox": Bbox.from_xyxy([528, 302, 628, 382]),
                        "segmentation": SegmentationRLE.from_mask(_mask(1080, 1920)),
                    },
                    {
                        "frame_idx": 2,
                        "bbox": Bbox.from_xyxy([536, 304, 636, 384]),
                        "segmentation": SegmentationRLE.from_mask(_mask(1080, 1920)),
                    },
                    {
                        "frame_idx": 3,
                        "bbox": Bbox.from_xyxy([544, 306, 644, 386]),
                        "segmentation": SegmentationRLE.from_mask(_mask(1080, 1920)),
                    },
                    {
                        "frame_idx": 4,
                        "bbox": Bbox.from_xyxy([552, 308, 652, 388]),
                        "segmentation": SegmentationRLE.from_mask(_mask(1080, 1920)),
                    },
                    {
                        "frame_idx": 5,
                        "bbox": Bbox.from_xyxy([560, 310, 660, 390]),
                        "segmentation": SegmentationRLE.from_mask(_mask(1080, 1920)),
                    },
                    {
                        "frame_idx": 6,
                        "bbox": Bbox.from_xyxy([568, 312, 668, 392]),
                        "segmentation": SegmentationRLE.from_mask(_mask(1080, 1920)),
                    },
                    {
                        "frame_idx": 7,
                        "bbox": Bbox.from_xyxy([576, 314, 676, 394]),
                        "segmentation": SegmentationRLE.from_mask(_mask(1080, 1920)),
                    },
                ],
            }
        ],
    )
    writer.append(
        filename="videos/action_002.mp4",
        width=1280,
        height=720,
        description="A pedestrian is walking.",
        actions=[
            {
                "category_id": 4,
                "track_id": 1,
                "start_idx": 1,
                "end_idx": 5,
                "description": "A pedestrian is walking.",
                "tracks": [
                    {"frame_idx": 1, "bbox": [100, 120, 180, 320]},
                    {"frame_idx": 2, "bbox": Bbox.from_xyxy([100, 120, 180, 320])},
                    {"frame_idx": 3, "bbox": Bbox.from_xyxy([100, 120, 180, 320])},
                    {"frame_idx": 4, "bbox": Bbox.from_xyxy([100, 120, 180, 320])},
                    {"frame_idx": 5, "bbox": Bbox.from_xyxy([100, 120, 180, 320])},
                ],
            },
            {
                "category_id": 5,
                "track_id": 2,
                "start_idx": 1,
                "end_idx": 5,
                "description": "A pedestrian is walking.",
                "tracks": [
                    {"frame_idx": 1, "bbox": Bbox.from_xyxy([100, 120, 180, 320])},
                    {"frame_idx": 2, "bbox": Bbox.from_xyxy([100, 120, 180, 320])},
                    {"frame_idx": 3, "bbox": Bbox.from_xyxy([100, 120, 180, 320])},
                    {"frame_idx": 4, "bbox": Bbox.from_xyxy([100, 120, 180, 320])},
                    {"frame_idx": 5, "bbox": Bbox.from_xyxy([100, 120, 180, 320])},
                ],
            },
        ],
    )
    writer.save()

    data, _, rows = _read_annotations(writer, expected_count=2)
    assert "task_type" not in rows[0]
    assert "schema_version" not in rows[0]
    assert rows[0]["actions"][0]["tracks"][0]["bbox"] == [
        520.0,
        300.0,
        620.0,
        380.0,
    ]
    assert rows[1]["actions"][0]["category_id"] == 4
    assert isinstance(data[0], ActionAnnotation)
    assert data[0].actions[0].tracks[0].bbox.to_list() == [
        520.0,
        300.0,
        620.0,
        380.0,
    ]
    assert data[1].actions[0].category_id == 4
    assert len(data[1].actions) == 2


def test_reader_validate_returns_false_on_invalid_data(tmp_path):
    writer = AnnotationWriter(
        TaskType.DETECTION,
        label={1: "person"},
        task_dir=tmp_path / "detection",
    )
    writer.append(
        filename="images/ok.jpg",
        width=640,
        height=480,
        instances=[{"id": 0, "category_id": 1, "bbox": [10, 20, 100, 200]}],
    )
    writer.save()

    # Corrupt JSONL with unknown category_id
    data_path = writer.data_path
    data_path.write_text(
        '{"filename":"images/bad.jpg","width":640,"height":480,'
        '"instances":[{"id":0,"category_id":99,"bbox":[10,20,100,200]}]}\n',
        encoding="utf-8",
    )
    reader = AnnotationReader(TaskType.DETECTION, writer.save_dir())
    assert reader.validate() is False


def test_sequence_annotation_format():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        writer = AnnotationWriter(
            TaskType.SEQUENCE,
            label=ALPHANUMERIC_VOCAB,
            task_dir=tmp_path / "sequence",
        )
        writer.append(
            filename="batch1_crop_plate/27993412_car0_inst0.jpg",
            width=224,
            height=128,
            sequences=["B", "1", "0", "7", "7", "P", "D", "V"],
        )
        writer.append(
            filename="batch1_crop_plate/empty_seq.jpg",
            width=224,
            height=128,
            sequences=[],
        )
        writer.save()

        assert writer.meta_path.is_file()
        assert writer.meta_path.name == ANNOTATION_VOCAB_FILENAME
        assert writer.meta_path.read_text(encoding="utf-8") == ALPHANUMERIC_VOCAB.read_text(
            encoding="utf-8"
        )

        data, label, rows = _read_annotations(writer, expected_count=2)
        assert label == {"vocab": ANNOTATION_VOCAB_FILENAME}
        assert rows[0]["sequences"] == ["B", "1", "0", "7", "7", "P", "D", "V"]
        assert rows[1]["sequences"] == []
        assert isinstance(data[0], SequenceAnnotation)
        assert data[0].sequences == ["B", "1", "0", "7", "7", "P", "D", "V"]
        assert data[1].sequences == []


def test_sequence_custom_vocab_filename():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        writer = AnnotationWriter(
            TaskType.SEQUENCE,
            label=ALPHANUMERIC_VOCAB,
            task_dir=tmp_path / "sequence",
            task_meta_filename="custom_vocab.txt",
        )
        writer.append(
            filename="sample.jpg",
            width=224,
            height=128,
            sequences=["B", "1", "0"],
        )
        writer.save()

        assert writer.meta_path.name == "custom_vocab.txt"
        assert writer.meta_path.is_file()
        assert not (writer.save_dir() / ANNOTATION_VOCAB_FILENAME).is_file()

        data, label, _rows = _read_annotations(writer, expected_count=1)
        assert label == {"vocab": "custom_vocab.txt"}
        assert data[0].sequences == ["B", "1", "0"]


def test_sequence_unknown_token_raises():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        writer = AnnotationWriter(
            TaskType.SEQUENCE,
            label=ALPHANUMERIC_VOCAB,
            task_dir=Path(tmp) / "sequence",
        )
        try:
            writer.append(
                filename="bad.jpg",
                width=224,
                height=128,
                sequences=["B", "z"],
            )
            raised = False
        except AnnotationFormatError:
            raised = True
        assert raised


def test_sequence_vocab_file_format():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        multi_token = tmp_path / "bad_multi.txt"
        multi_token.write_text("A B\n", encoding="utf-8")
        try:
            AnnotationWriter(
                TaskType.SEQUENCE,
                label=multi_token,
                task_dir=tmp_path / "sequence",
            )
            raised = False
        except AnnotationFormatError:
            raised = True
        assert raised

        duplicate = tmp_path / "bad_dup.txt"
        duplicate.write_text("A\nA\n", encoding="utf-8")
        try:
            AnnotationWriter(
                TaskType.SEQUENCE,
                label=duplicate,
                task_dir=tmp_path / "sequence",
            )
            raised = False
        except AnnotationFormatError:
            raised = True
        assert raised

        trailing_space = tmp_path / "bad_space.txt"
        trailing_space.write_text("A \n", encoding="utf-8")
        try:
            AnnotationWriter(
                TaskType.SEQUENCE,
                label=trailing_space,
                task_dir=tmp_path / "sequence",
            )
            raised = False
        except AnnotationFormatError:
            raised = True
        assert raised

        empty = tmp_path / "bad_empty.txt"
        empty.write_text("\n\n", encoding="utf-8")
        try:
            AnnotationWriter(
                TaskType.SEQUENCE,
                label=empty,
                task_dir=tmp_path / "sequence",
            )
            raised = False
        except AnnotationFormatError:
            raised = True
        assert raised


def test_sequence_reader_rejects_unknown_token():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        writer = AnnotationWriter(
            TaskType.SEQUENCE,
            label=ASSETS_DIR / "digits_vocab.txt",
            task_dir=tmp_path / "sequence",
        )
        writer.append(
            filename="ok.jpg",
            width=224,
            height=128,
            sequences=["1", "2", "3"],
        )
        writer.save()

        writer.data_path.write_text(
            writer.data_path.read_text(encoding="utf-8")
            + '{"filename":"bad.jpg","width":224,"height":128,"sequences":["1","X"]}\n',
            encoding="utf-8",
        )
        reader = AnnotationReader(TaskType.SEQUENCE, writer.save_dir())
        assert reader.validate() is False


def run_all():
    from pathlib import Path
    import tempfile

    test_detection_annotation_format()
    test_keypoint_annotation_format()
    test_segmentation_annotation_format()
    test_classification_annotation_format()
    test_relationship_annotation_format()
    test_vlm_annotation_format()
    test_conversation_annotation_format()
    test_sequence_annotation_format()
    test_sequence_custom_vocab_filename()
    test_sequence_unknown_token_raises()
    test_sequence_vocab_file_format()
    test_sequence_reader_rejects_unknown_token()
    test_action_annotation_format()
    with tempfile.TemporaryDirectory() as tmp:
        test_reader_validate_returns_false_on_invalid_data(Path(tmp))


if __name__ == "__main__":
    run_all()
    print("all annotation format tests passed")
