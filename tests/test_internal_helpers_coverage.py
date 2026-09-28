"""Cover internal helpers below the per-file 90% CI gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vdschema import (
    ActionAnnotation,
    AnnotationFormatError,
    AnnotationWriter,
    ClassificationAnnotation,
    Name,
    RelationshipAnnotation,
    TaskType,
)
from vdschema._annotation_components import (
    ActionEvent,
    Bbox,
    ClassificationHead,
    ConversationRole,
    ConversationTurn,
    Instance,
    Keypoint,
    Relationship,
    SegmentationRLE,
    TrackItem,
    _action_event,
    _bbox,
    _content_part,
    _conversation_turn,
    _ensure_positive_int,
    _track_item,
    parse_keypoints,
)
from vdschema.annotation_dict import (
    ActionLabelDict,
    ClassificationLabelDict,
    DetectionLabelDict,
    KeypointLabelDict,
    LabelMap,
    RelationshipLabelDict,
    SegmentationLabelDict,
    SequenceLabelDict,
    TaskLabelDict,
    _coerce_name,
    _load_vocab_tokens,
    build_label_dict,
    load_reader_label_dict,
    meta_path_for,
)


def test_ensure_positive_int_rejects_zero() -> None:
    with pytest.raises(AnnotationFormatError, match=">= 1"):
        _ensure_positive_int(0, "height")


def test_bbox_from_xywh_and_cxcywh_length() -> None:
    with pytest.raises(AnnotationFormatError, match="exactly 4"):
        Bbox.from_xywh([1, 2, 3])
    with pytest.raises(AnnotationFormatError, match="exactly 4"):
        Bbox.from_cxcywh([1, 2, 3])


def test_segmentation_rle_validate_bad_size_or_counts() -> None:
    bad = SegmentationRLE._create(0, 1, "x")
    with pytest.raises(AnnotationFormatError, match="positive"):
        bad.validate()
    empty = SegmentationRLE._create(1, 1, "")
    with pytest.raises(AnnotationFormatError, match="counts"):
        empty.validate()


def test_keypoint_validation_and_parse_variants() -> None:
    with pytest.raises(AnnotationFormatError, match="\\[x,y,v\\]"):
        Keypoint.from_list([1, 2])
    bad_vis = Keypoint(0.0, 0.0, 5)
    with pytest.raises(AnnotationFormatError, match="visibility"):
        bad_vis.validate()

    assert parse_keypoints([]) == []
    triplets = parse_keypoints([[1, 2, 2], [3, 4, 0]])
    assert triplets and triplets[0].x == 1.0
    flat = parse_keypoints([1, 2, 2, 3, 4, 1])
    assert len(flat) == 2

    with pytest.raises(AnnotationFormatError, match="not an object"):
        parse_keypoints({"nose": [1, 2, 2]})
    with pytest.raises(AnnotationFormatError, match="must be a list"):
        parse_keypoints("bad")
    with pytest.raises(AnnotationFormatError, match="multiple of 3"):
        parse_keypoints([1, 2])
    with pytest.raises(AnnotationFormatError, match="COCO flat"):
        parse_keypoints(["bad"])


def test_instance_and_heads_validation() -> None:
    inst = Instance.from_dict({"id": -1, "category_id": 1, "bbox": [0, 0, 1, 1]})
    with pytest.raises(AnnotationFormatError, match="category_id"):
        inst.validate()

    head = ClassificationHead(category_attr="", category_ids=[1])
    with pytest.raises(AnnotationFormatError, match="category_attr"):
        head.validate()
    head2 = ClassificationHead(category_attr="a", category_ids=[])
    with pytest.raises(AnnotationFormatError, match="category_ids"):
        head2.validate()
    head3 = ClassificationHead(category_attr="a", category_ids=[1, 1])
    with pytest.raises(AnnotationFormatError, match="unique"):
        head3.validate()

    rel = Relationship(subject_id=0, object_id=0, relation_type="")
    with pytest.raises(AnnotationFormatError, match="relation_type"):
        rel.validate()


def test_action_track_validation() -> None:
    track = TrackItem(frame_idx=-1, bbox=Bbox.from_list([0, 0, 1, 1]))
    with pytest.raises(AnnotationFormatError, match="frame_idx"):
        track.validate()

    event = ActionEvent(
        category_id=1,
        track_id=1,
        start_idx=2,
        end_idx=1,
        tracks=[TrackItem(frame_idx=0, bbox=Bbox.from_list([0, 0, 1, 1]))],
    )
    with pytest.raises(AnnotationFormatError, match="end_idx"):
        event.validate()
    event2 = ActionEvent(
        category_id=1,
        track_id=1,
        start_idx=0,
        end_idx=0,
        tracks=[],
    )
    with pytest.raises(AnnotationFormatError, match="tracks"):
        event2.validate()


def test_bbox_and_rle_coercion_helpers() -> None:
    box = Bbox.from_list([0, 0, 1, 1])
    assert _bbox(box) is box
    assert _bbox({"format": "xywh", "bbox": [0, 0, 2, 2]}).to_list()[2] == 2.0
    assert _bbox({"format": "cxcywh", "bbox": [1, 1, 2, 2]}).x1 == 0.0
    assert _bbox({"format": "xyxy", "x1": 0, "y1": 0, "x2": 3, "y2": 3}).x2 == 3.0

    mask = np.ones((2, 2), dtype=np.uint8)
    from vdschema._annotation_components import _rle_mask

    parsed = _rle_mask({"mask": mask})
    assert parsed is not None
    assert _rle_mask(parsed) is parsed


def test_geometry_type_errors() -> None:
    with pytest.raises(AnnotationFormatError, match="polyline"):
        Instance.from_dict({"id": 0, "category_id": 1, "polyline": "x"})
    with pytest.raises(AnnotationFormatError, match="point"):
        Instance.from_dict({"id": 0, "category_id": 1, "point": "x"})


def test_conversation_and_action_wrappers() -> None:
    with pytest.raises(AnnotationFormatError, match="Unknown content part"):
        _content_part({"type": "audio"})

    turn = _conversation_turn(
        {
            "role": "user",
            "content": [
                {"type": "image", "filename": "a.jpg"},
                {"type": "text", "text": "hi"},
            ],
        }
    )
    assert turn.content[0].filename == "a.jpg"

    legacy = _conversation_turn(
        ConversationTurn(role=ConversationRole.USER, image="a.jpg", text="hi")
    )
    assert legacy.content

    tr = _track_item(
        {
            "frame_idx": 0,
            "bbox": [0, 0, 1, 1],
            "rle_mask": {"size": [2, 2], "counts": "6320"},
        }
    )
    assert tr.frame_idx == 0
    ev = ActionEvent(
        category_id=1,
        track_id=1,
        start_idx=0,
        end_idx=0,
        tracks=[tr],
    )
    assert _action_event(ev) is ev


def test_name_and_label_map_errors() -> None:
    with pytest.raises(AnnotationFormatError, match="name must not be empty"):
        Name("")
    with pytest.raises(AnnotationFormatError, match="alias"):
        Name("a", alias=[""])
    with pytest.raises(AnnotationFormatError, match="prompt"):
        Name("a", prompt=[""])

    with pytest.raises(AnnotationFormatError, match="label values must be"):
        _coerce_name(123)

    with pytest.raises(AnnotationFormatError, match="duplicate category id"):

        class _DupMap:
            def items(self):
                return [(1, "a"), (1, "b")]

        LabelMap.from_id_map(_DupMap())  # type: ignore[arg-type]
    with pytest.raises(AnnotationFormatError, match="duplicate category name"):
        LabelMap.from_id_map({1: "a", 2: "a"})

    lm = LabelMap.from_id_map({1: Name("a", alias=("b",), prompt=("p",))})
    entries = lm.to_entries(
        id_key="id",
        name_key="name",
        alias_key="alias",
        prompt_key="prompt",
    )
    assert entries[0]["alias"] == ["b"]
    rel_entries = lm.to_relationship_entries()
    assert rel_entries[0]["relationship_name"] == "a"


def test_task_label_dict_validate_branches(tmp_path: Path) -> None:
    det = DetectionLabelDict({1: "cat"})
    det.validate_annotation(
        ActionAnnotation(filename="v.mp4", width=1, height=1, actions=[])
    )  # type: ignore[arg-type]

    writer = AnnotationWriter(
        TaskType.KEYPOINT,
        label={1: "person"},
        task_dir=tmp_path / "kp",
    )
    writer.save()
    meta = meta_path_for(writer.save_dir())
    raw = json.loads(meta.read_text(encoding="utf-8"))
    raw.pop("keypoint")
    raw["detection"] = [{"category_id": 1, "category_name": "person"}]
    meta.write_text(json.dumps(raw), encoding="utf-8")
    loaded = KeypointLabelDict.load(meta)
    assert loaded.vocab.to_label()[1].name == "person"

    writer2 = AnnotationWriter(
        TaskType.SEGMENTATION,
        label={1: "obj"},
        task_dir=tmp_path / "seg",
    )
    writer2.save()
    meta2 = meta_path_for(writer2.save_dir())
    raw2 = json.loads(meta2.read_text(encoding="utf-8"))
    raw2.pop("segmentation")
    raw2["detection"] = [{"category_id": 1, "category_name": "obj"}]
    meta2.write_text(json.dumps(raw2), encoding="utf-8")
    assert SegmentationLabelDict.load(meta2).vocab.to_label()[1].name == "obj"

    action_ld = ActionLabelDict({1: "wave"})
    with pytest.raises(AnnotationFormatError, match="unknown category_id"):
        action_ld.validate_annotation(
            ActionAnnotation(
                filename="v.mp4",
                width=10,
                height=10,
                actions=[
                    {
                        "category_id": 99,
                        "track_id": 1,
                        "start_idx": 0,
                        "end_idx": 0,
                        "tracks": [{"frame_idx": 0, "bbox": [0, 0, 1, 1]}],
                    }
                ],
            )
        )

    cls_ld = ClassificationLabelDict({"color": {1: "red"}})
    with pytest.raises(AnnotationFormatError, match="unknown category_attr"):
        cls_ld.validate_annotation(
            ClassificationAnnotation(
                filename="a.jpg",
                width=1,
                height=1,
                categories=[{"category_attr": "size", "category_ids": [1]}],
            )
        )

    rel_ld = RelationshipLabelDict({1: "a"}, {0: "near"})
    with pytest.raises(AnnotationFormatError, match="unknown relation_type"):
        rel_ld.validate_annotation(
            RelationshipAnnotation(
                filename="a.jpg",
                width=10,
                height=10,
                instances=[{"id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]}],
                relationships=[
                    {"subject_id": 0, "object_id": 0, "relation_type": "far"}
                ],
            )
        )


def test_relationship_and_classification_load_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"detection": []}', encoding="utf-8")
    with pytest.raises(
        AnnotationFormatError, match="requires detection and relationship"
    ):
        RelationshipLabelDict.load(bad)

    with pytest.raises(AnnotationFormatError, match="relationship label must be"):
        RelationshipLabelDict.from_writer_label([])  # type: ignore[arg-type]
    with pytest.raises(AnnotationFormatError, match="requires 'detection'"):
        RelationshipLabelDict.from_writer_label({"detection": {1: "a"}})

    cls_meta = tmp_path / "cls.json"
    cls_meta.write_text(
        json.dumps(
            {
                "annotation_schema_ref": "x",
                "classification": [
                    {
                        "category_attr": "color",
                        "category_label": [{"category_id": 1, "category_name": "red"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert ClassificationLabelDict.load(cls_meta).to_label()["color"][1].name == "red"


def test_sequence_vocab_and_build_label_dict(tmp_path: Path) -> None:
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\n# comment\nb\n", encoding="utf-8")
    assert _load_vocab_tokens(vocab) == ["a", "b"]

    bad_ws = tmp_path / "ws.txt"
    bad_ws.write_text(" a\n", encoding="utf-8")
    with pytest.raises(AnnotationFormatError, match="whitespace"):
        _load_vocab_tokens(bad_ws)

    dup = tmp_path / "dup.txt"
    dup.write_text("a\na\n", encoding="utf-8")
    with pytest.raises(AnnotationFormatError, match="duplicate"):
        _load_vocab_tokens(dup)

    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(AnnotationFormatError, match="empty"):
        _load_vocab_tokens(empty)

    seq = SequenceLabelDict(vocab_path=vocab)
    with pytest.raises(AnnotationFormatError, match="external file"):
        seq.to_data()

    out_vocab = tmp_path / "out" / "annotation_vocab.txt"
    seq.save(out_vocab)
    assert out_vocab.is_file()

    with pytest.raises(AnnotationFormatError, match="requires label"):
        build_label_dict(TaskType.DETECTION, None)

    loaded, payload = load_reader_label_dict(
        TaskType.DETECTION, tmp_path / "missing.json"
    )
    assert payload is None
    assert loaded.to_label() == {}


def test_task_label_dict_not_implemented() -> None:
    base = TaskLabelDict()
    with pytest.raises(NotImplementedError):
        base.to_label()
    with pytest.raises(NotImplementedError):
        TaskLabelDict.load(Path("x.json"))
