"""Tests for prompt orchestration, distillation, and corrections."""

from __future__ import annotations

from dica.context import ContextBudget
from dica.orchestrator import (
    PromptOrchestrator,
    PromptPayload,
    ReferenceSnippet,
    derive_constraints,
    distill_syntax,
)
from dica.vault import ChunkTags, CodeChunk, ChunkKind
from dica.dispatcher import DispatchResult
from dica.config import ContextConfig, OllamaConfig


def test_distill_syntax_strips_docstrings() -> None:
    src = '''\
def foo(x: int) -> int:
    """Return x."""
    return x
'''
    out = distill_syntax(src)
    assert "Return x" not in out
    assert "def foo" in out
    assert "return x" in out


def test_distill_syntax_docstring_only_body_becomes_pass() -> None:
    src = '''\
def bare():
    """Only docs."""
'''
    out = distill_syntax(src)
    assert "pass" in out


def test_distill_syntax_unparseable_passthrough() -> None:
    broken = "def foo("
    assert distill_syntax(broken) == broken


def test_derive_constraints_from_tags() -> None:
    chunk = CodeChunk(
        chunk_id="abc",
        name="Thing",
        kind=ChunkKind.CLASS,
        source="class Thing: pass",
        file_path="t.py",
        lineno=1,
        tags=ChunkTags(is_async=True, has_pydantic=True, is_class=True),
        keywords=frozenset({"thing"}),
    )
    result = DispatchResult(
        chunk=chunk, lexical_score=1.0, structural_score=0.5
    )
    constraints = derive_constraints([result])
    assert any("async" in c.lower() for c in constraints)
    assert any("pydantic" in c.lower() for c in constraints)


def test_build_correction_uses_diagnostics_field(
    roomy_budget: ContextBudget,
) -> None:
    orch = PromptOrchestrator()
    base = orch.build_draft_payload("build a router")
    corr = orch.build_correction(
        base,
        failed_code="def foo():\n    pass\n",
        diagnostics="error: incompatible types\n" * 20,
        budget=roomy_budget,
    )
    assert corr.diagnostics is not None
    assert "incompatible types" in corr.diagnostics
    # Raw diagnostics dump must not be stuffed into the task body
    assert "incompatible types\nincompatible types" not in corr.target_task
    assembled = corr.render_budgeted(roomy_budget)
    assert "[VERIFICATION DIAGNOSTICS]" in assembled.text
    assert "[ACTIVE TARGET TASK]" in assembled.text


def test_render_budgeted_drops_huge_reference(tight_budget: ContextBudget) -> None:
    payload = PromptPayload(
        references=(
            ReferenceSnippet(
                name="huge",
                origin="a.py",
                relevance=1.0,
                source="x = " + ("z" * 5000),
            ),
            ReferenceSnippet(
                name="tiny",
                origin="b.py",
                relevance=0.5,
                source="y = 1",
            ),
        ),
        target_task="do it",
        diagnostics="E" * 4000,
    )
    assembled = payload.render_budgeted(tight_budget)
    assert assembled.report.used_tokens <= assembled.report.budget_tokens + 100
    assert (
        "huge" in assembled.report.dropped_chunks
        or assembled.report.diagnostics_truncated
    )
