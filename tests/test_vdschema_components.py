"""Bbox, RLE, and lazy package exports."""

from __future__ import annotations

import numpy as np
import pytest

import vdschema
from vdschema import AnnotationFormatError, Bbox, SegmentationRLE


def test_bbox_formats() -> None:
    assert Bbox.from_xywh([0, 0, 10, 20]).to_list() == [0, 0, 10, 20]
    cx = Bbox.from_cxcywh([5, 5, 10, 10]).to_list()
    assert cx[0] == 0 and cx[2] == 10


def test_bbox_invalid() -> None:
    with pytest.raises(AnnotationFormatError):
        Bbox.from_list([0, 0, 0, 1])
    with pytest.raises(AnnotationFormatError):
        Bbox.from_list([1, 2, 3])


def test_segmentation_rle_direct_construct_forbidden() -> None:
    with pytest.raises(AnnotationFormatError):
        SegmentationRLE(1, 1, "x")  # type: ignore[call-arg]


def test_segmentation_rle_from_mask_and_roundtrip() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    rle = SegmentationRLE.from_mask(mask)
    assert rle.to_mask().shape == (4, 4)


def test_vdschema_lazy_switch_and_source() -> None:
    switch_fn = vdschema.switch
    source = vdschema.Source
    assert switch_fn is not None
    assert source.MONOLITH.value == "monolith"


def test_bbox_unknown_format() -> None:
    from vdschema._annotation_components import _bbox

    with pytest.raises(AnnotationFormatError, match="Unknown bbox format"):
        _bbox({"format": "bad", "bbox": [0, 0, 1, 1]})


def test_segmentation_rle_from_dict_bytes_counts() -> None:
    rle = SegmentationRLE.from_dict({"size": [2, 2], "counts": b"6320"})
    assert rle.counts


def test_instance_rle_and_polygon_coexist() -> None:
    from vdschema.annotation_format import Instance

    inst = Instance.from_dict(
        {
            "id": 0,
            "category_id": 1,
            "bbox": [0, 0, 1, 1],
            "polygon": [0, 0, 1, 0, 1, 1],
            "rle_mask": {"size": [10, 10], "counts": "6320004"},
        }
    )
    d = inst.to_dict()
    assert "polygon" in d and "rle_mask" in d


def test_segmentation_rle_from_mask_wrong_ndim() -> None:
    with pytest.raises(AnnotationFormatError, match="2D"):
        SegmentationRLE.from_mask(np.zeros((2, 2, 2)))


def test_instance_polygon_must_be_list() -> None:
    from vdschema.annotation_format import Instance

    with pytest.raises(AnnotationFormatError, match="polygon"):
        Instance.from_dict({"id": 0, "category_id": 1, "polygon": "bad"})


def test_conversation_turn_legacy_fields() -> None:
    from vdschema import ConversationRole, ConversationTurn

    turn = ConversationTurn(role=ConversationRole.USER, image="a.jpg", text="hi")
    d = turn.to_dict()
    assert d["role"] == "user"


def test_kmot_track_empty_frame_range() -> None:
    from vdswitch.utils.kmot import KmotTrack

    track = KmotTrack(track_id=1)
    assert track.start_idx == 0
    assert track.end_idx == 0


def test_vdschema_lazy_invalid_attr() -> None:
    with pytest.raises(AttributeError):
        _ = vdschema.not_an_export  # type: ignore[attr-defined]
