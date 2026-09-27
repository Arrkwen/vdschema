"""Classification converter error paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdswitch.converters.monolith import _load_cls_label


def test_load_cls_label_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid classification label"):
        _load_cls_label(bad)

    empty_head = tmp_path / "head.json"
    empty_head.write_text('{"gender":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        _load_cls_label(empty_head)
