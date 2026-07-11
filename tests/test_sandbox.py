"""Tests for sandbox runner output parsing and local process hygiene."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from dica.sandbox import _kill_process, _local_verify, _parse_runner_output


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


@pytest.mark.asyncio
async def test_kill_process_reaps_sleeping_child() -> None:
    """M1 helper: kill + wait must reap a live subprocess."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.returncode is None
    await _kill_process(proc)
    assert proc.returncode is not None


@pytest.mark.asyncio
async def test_local_verify_timeout_kills_checkers(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the wall clock budget expires, checkers are killed and a failure is returned."""
    real_exec = asyncio.create_subprocess_exec

    async def _hanging_exec(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        # Ignore ruff/mypy argv; spawn a sleeper so the shared timeout always fires.
        return await real_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _hanging_exec)

    checks = await _local_verify("x = 1\n", timeout=0.3)
    assert len(checks) == 1
    assert checks[0].tool == "sandbox"
    assert checks[0].passed is False
    assert "killed" in checks[0].output.lower() or "exceeded" in checks[0].output.lower()
