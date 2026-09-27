"""Action converter edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdschema import Source, TaskType, switch
from vdswitch.converters.action import _load_act_label, _parse_legacy_meta_line


def test_load_act_label_action_list(tmp_path: Path) -> None:
    path = tmp_path / "label.json"
    path.write_text(
        '{"action":[{"category_id":1,"category_name":"walk"}]}',
        encoding="utf-8",
    )
    label, name_to_id = _load_act_label(path)
    assert label[1].name == "walk"
    assert name_to_id["walk"] == 1


def test_load_act_label_invalid(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid action label"):
        _load_act_label(path)

    empty_action = tmp_path / "empty_action.json"
    empty_action.write_text('{"action":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="action\\[\\] is empty"):
        _load_act_label(empty_action)

    empty_head = tmp_path / "empty_head.json"
    empty_head.write_text('{"head":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        _load_act_label(empty_head)


def test_parse_legacy_meta_line_skips_and_errors() -> None:
    with pytest.raises(ValueError, match="invalid act meta"):
        _parse_legacy_meta_line("only;three;parts")

    assert _parse_legacy_meta_line("v.avi;4;500;400;0;kmot/x.txt") is None
    assert _parse_legacy_meta_line("v.avi;4;443;444;0;kmot/x.txt") is None
    assert _parse_legacy_meta_line("v.avi;4;443;446;0;kmot/x.txt") == (
        "v.avi",
        "kmot/x.txt",
    )


def test_action_label_action_list_format(act_legacy_dir: Path, tmp_path: Path) -> None:
    label_path = act_legacy_dir / "meta" / "label_dict.json"
    label_path.write_text(
        json.dumps(
            {"action": [{"category_id": 1, "category_name": "package_tossing"}]}
        ),
        encoding="utf-8",
    )
    switch(
        task=TaskType.ACTION,
        source=Source.MONOLITH,
        input_data=act_legacy_dir / "meta" / "video_train.txt",
        input_label=label_path,
        input_root=act_legacy_dir,
        output=tmp_path / "out",
    )
