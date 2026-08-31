"""Tests for kmot parsing."""

from __future__ import annotations

from pathlib import Path

from vdswitch.utils import parse_kmot_file


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
