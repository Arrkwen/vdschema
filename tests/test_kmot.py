"""Tests for kmot parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdswitch.utils.kmot import KmotTrack, filter_tracks_to_window, parse_kmot_file


def test_parse_kmot_track_absolute_indices(act_legacy_dir: Path) -> None:
    name_to_id = {"normal": 0, "package_tossing": 1}
    kmot_path = act_legacy_dir / "meta/kmot/sample.txt"
    tracks = parse_kmot_file(kmot_path, name_to_id=name_to_id)
    track = tracks[18]
    action = track.to_action_dict()
    action_legacy = track.to_action_dict(frame_index_offset=1)

    assert action["track_id"] == 18
    assert action["start_idx"] == 442
    assert action["end_idx"] == 445
    assert action["category_id"] == 1
    assert action["tracks"][0]["frame_idx"] == 442
    assert action_legacy["tracks"][0]["frame_idx"] == 443
    assert action_legacy["start_idx"] == 443
    assert action_legacy["end_idx"] == 446


def test_parse_kmot_records_all_track_ids(act_legacy_dir: Path) -> None:
    name_to_id = {"normal": 0, "package_tossing": 1}
    tracks = parse_kmot_file(
        act_legacy_dir / "meta/kmot/sample.txt",
        name_to_id=name_to_id,
    )
    assert len(tracks) == 1
    assert 18 in tracks


def test_parse_kmot_skips_invalid_lines(tmp_path: Path) -> None:
    kmot = tmp_path / "t.txt"
    kmot.write_text(
        "short\n0,1,0,0,0,0,0\n0,2,0,0,1,1,0,1\n",
        encoding="utf-8",
    )
    tracks = parse_kmot_file(kmot, name_to_id={"a": 1})
    assert 2 in tracks


def test_parse_kmot_unknown_category(tmp_path: Path) -> None:
    kmot = tmp_path / "t.txt"
    kmot.write_text("0,1,0,0,1,1,0,unknown_label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown action category"):
        parse_kmot_file(kmot, name_to_id={"a": 0})


def test_filter_tracks_to_window() -> None:
    from vdswitch.utils.kmot import KmotFrame

    track = KmotTrack(
        track_id=1,
        frames=[KmotFrame(5, [0, 0, 1, 1], "x")],
        category_id=0,
    )
    grouped = {1: track}
    filtered = filter_tracks_to_window(grouped, 4, 6, frame_index_offset=0)
    assert 1 in filtered
    assert filter_tracks_to_window(grouped, 10, 12) == {}
