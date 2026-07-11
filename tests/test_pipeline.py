"""Integration-style tests for the shared pipeline engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from dica.config import DispatchConfig
from dica.pipeline import (
    PipelineEngine,
    syntax_regression,
    unwrap_extraction,
)
from dica.extraction import extract_python_code
from tests.conftest import CORPUS_DIR, fenced_python


class _ScriptedClient:
    """Returns canned responses in order; raises if exhausted unexpectedly."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        if not self._responses:
            raise RuntimeError(f"unexpected complete() call #{self.calls}")
        return self._responses.pop(0)


def test_syntax_regression_helper() -> None:
    assert syntax_regression("x = 1") is None
    err = syntax_regression("def foo(")
    assert err is not None
    assert err.lineno is not None


def test_unwrap_requires_ok() -> None:
    bad = extract_python_code(fenced_python("def foo("))
    assert bad.ok is False
    assert unwrap_extraction(bad) is None
    good = extract_python_code(fenced_python("x = 1"))
    assert unwrap_extraction(good) == "x = 1"


@pytest.mark.asyncio
async def test_dry_run_finishes_without_client() -> None:
    engine = PipelineEngine(
        client=None,
        corpus=CORPUS_DIR,
        # Force lexical-only so dry-run does not need Ollama embeds.
        # (semantic build is skipped when weight is read from config;
        #  we still tolerate soft-fail if config has weight > 0)
    )
    events = []
    async for event in engine.run("async CRUD router", dry_run=True):
        events.append(event)
    assert events
    final = events[-1]
    assert final.finished is True
    assert final.exit_code == 0
    assert final.dry_run_text is not None
    assert "ACTIVE TARGET TASK" in final.dry_run_text
    assert final.schedule_lines


@pytest.mark.asyncio
async def test_empty_corpus_exits_one(tmp_path: Path) -> None:
    empty = tmp_path / "empty_corpus"
    empty.mkdir()
    engine = PipelineEngine(client=None, corpus=empty)
    events = [e async for e in engine.run("anything")]
    assert events[-1].finished is True
    assert events[-1].exit_code == 1
    assert events[-1].fatal is True


@pytest.mark.asyncio
async def test_scripted_pass0_abort_on_no_code() -> None:
    client = _ScriptedClient(
        [
            "I refuse to emit a code block.",
            "Still no fence here.",
            "Nope.",
        ]
    )
    engine = PipelineEngine(client=client, corpus=CORPUS_DIR, max_attempts=1)
    events = [e async for e in engine.run("build a tiny helper")]
    assert any(e.fatal or (e.finished and e.exit_code == 1) for e in events)
    # Format retries should have consumed multiple complete() calls
    assert client.calls >= 2


@pytest.mark.asyncio
async def test_scripted_valid_code_reaches_sandbox() -> None:
    """Happy-ish path: valid draft → refine or skip → verify (local sandbox).

    We do not assert verify PASS (mypy --strict is strict); we assert the
    pipeline reaches a terminal finished event with code present.
    """
    valid = fenced_python(
        "def greet(name: str) -> str:\n"
        '    """Return a greeting."""\n'
        '    return f"hi {name}"\n'
    )
    # Enough identical valid responses for pass 0 + up to 3 refinements
    # + possible format noise + corrections budget.
    client = _ScriptedClient([valid] * 20)
    engine = PipelineEngine(client=client, corpus=CORPUS_DIR, max_attempts=1)
    events = [e async for e in engine.run("write a greet helper")]
    final = events[-1]
    assert final.finished is True
    assert final.code is not None
    assert "def greet" in final.code
    assert final.exit_code in (0, 2)  # verified or exhausted corrections
    assert client.calls >= 1
