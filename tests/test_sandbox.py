"""Tests for sandbox runner output parsing (no Docker required)."""

from __future__ import annotations

import json

from dica.sandbox import _parse_runner_output


def test_parse_runner_output_last_json_line() -> None:
    payload = [
        {"tool": "ruff", "passed": True, "output": ""},
        {"tool": "mypy", "passed": False, "output": "error: x"},
    ]
    logs = "noise\nstarting\n" + json.dumps(payload) + "\n"
    checks = _parse_runner_output(logs)
    assert len(checks) == 2
    assert checks[0].tool == "ruff" and checks[0].passed is True
    assert checks[1].tool == "mypy" and checks[1].passed is False
    assert "error: x" in checks[1].output


def test_parse_runner_output_ignores_non_json() -> None:
    logs = "ruff said something\nmypy also\n"
    checks = _parse_runner_output(logs)
    assert len(checks) == 1
    assert checks[0].tool == "sandbox"
    assert checks[0].passed is False


def test_parse_runner_output_picks_last_valid_array() -> None:
    first = json.dumps([{"tool": "ruff", "passed": False, "output": "old"}])
    second = json.dumps([{"tool": "ruff", "passed": True, "output": "new"}])
    logs = first + "\n" + second + "\n"
    checks = _parse_runner_output(logs)
    assert checks[0].passed is True
    assert checks[0].output == "new"
