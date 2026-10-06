"""vdswitch CLI and registry coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdschema import TaskType
from vdswitch.cli import (
    _terminal_bold,
    build_convert_parser,
    main,
    run_help,
)
from vdswitch.converters.registry import get_converter_class, parse_task
from vdswitch.converters.sources import Source


def test_main_help_subcommand() -> None:
    assert main(["help"]) == 0


def test_main_help_source_without_task(capsys) -> None:
    assert main(["help", "--source", "coco"]) == 2
    assert "--source requires --task" in capsys.readouterr().err


def test_run_help_invalid_source() -> None:
    with pytest.raises(SystemExit):
        run_help(["--task", "detection", "--source", "not_a_source"])


def test_run_help_value_error(monkeypatch) -> None:
    def boom(**kwargs):
        raise ValueError("bad help")

    monkeypatch.setattr("vdswitch.cli.format_help_text", boom)
    assert run_help(["--task", "detection"]) == 1


def test_terminal_bold_on_tty(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    class FakeStream:
        def isatty(self) -> bool:
            return True

    out = _terminal_bold("x", stream=FakeStream())
    assert out.startswith("\033[1m")


def test_main_convert_error(capsys, tmp_path: Path) -> None:
    code = main(
        [
            "--task",
            "detection",
            "--source",
            "monolith",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--option",
            f"category={tmp_path / 'missing.json'}",
            "--option",
            f"root={tmp_path}",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "vdswitch: error:" in capsys.readouterr().err


def test_parse_task_invalid() -> None:
    with pytest.raises(ValueError, match="unsupported task"):
        parse_task("not_a_task")


def test_get_converter_bad_source() -> None:
    with pytest.raises(ValueError, match="unsupported source"):
        get_converter_class(TaskType.DETECTION, Source.OCR)


def test_source_parse_legacy_up() -> None:
    assert Source.parse("up") is Source.MONOLITH


def test_source_parse_invalid() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        Source.parse("nope")


def test_terminal_bold_respects_no_color(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert _terminal_bold("hint") == "hint"


def test_terminal_bold_non_tty(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    class FakeStream:
        def isatty(self) -> bool:
            return False

    assert _terminal_bold("hint", stream=FakeStream()) == "hint"


def test_convert_parser_prints_help(capsys) -> None:
    build_convert_parser().print_help()
    assert "vdswitch help" in capsys.readouterr().out
