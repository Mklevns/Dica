"""Tests for prompt orchestration, distillation, and corrections."""

from __future__ import annotations

from dica.context import ContextBudget
from dica.orchestrator import (
    PromptOrchestrator,
    PromptPayload,
    ReferenceSnippet,
    build_selector_payload,
    derive_constraints,
    distill_syntax,
    parse_selection_name,
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


def test_distill_syntax_preserves_comments() -> None:
    """M7: CST distill must keep architectural # comments and type-ignores."""
    src = '''\
def foo(x: int) -> int:
    """Return x after the clamp — prose the model does not need."""
    # architectural why: clamp before fan-out to the worker pool
    if x < 0:
        x = 0
    return x  # type: ignore[no-any-return]
'''
    out = distill_syntax(src)
    assert "prose the model does not need" not in out
    assert "clamp before fan-out" in out
    assert "type: ignore[no-any-return]" in out
    assert "return x" in out


def test_distill_syntax_preserves_class_and_module_comments() -> None:
    src = '''\
"""Module docstring should go."""

# Keep this module-level rationale
class Gate:
    """Class docstring should go."""

    def check(self) -> bool:
        """Method docstring should go."""
        # Why: short-circuit on sentinel
        return True
'''
    out = distill_syntax(src)
    assert "Module docstring should go" not in out
    assert "Class docstring should go" not in out
    assert "Method docstring should go" not in out
    assert "Keep this module-level rationale" in out
    assert "short-circuit on sentinel" in out
    assert "class Gate" in out


def test_distill_syntax_docstring_only_body_becomes_pass() -> None:
    src = '''\
def bare():
    """Only docs."""
'''
    out = distill_syntax(src)
    assert "pass" in out
    assert "Only docs" not in out


def test_distill_syntax_docstring_plus_comment_footer_gets_pass() -> None:
    src = '''\
def bare():
    """Only docs."""
    # still need a real statement
'''
    out = distill_syntax(src)
    assert "pass" in out
    assert "still need a real statement" in out
    assert "Only docs" not in out


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


def _sample_chunk() -> CodeChunk:
    return CodeChunk(
        chunk_id="abc",
        name="Thing",
        kind=ChunkKind.CLASS,
        source='class Thing:\n    """A thing."""\n    pass\n',
        docstring="A thing.",
        file_path="t.py",
        lineno=1,
        tags=ChunkTags(is_async=True, has_pydantic=True, is_class=True),
        keywords=frozenset({"thing"}),
    )


def test_build_anchored_payload_single_reference() -> None:
    orch = PromptOrchestrator()
    chunk = _sample_chunk()
    payload = orch.build_anchored_payload(
        "refactor to async",
        "def old():\n    pass\n",
        chunk,
    )
    assert len(payload.references) == 1
    assert payload.references[0].name == "Thing"
    assert payload.references[0].relevance == 1.0
    assert "A thing." not in payload.references[0].source  # distilled
    assert any("async" in c.lower() for c in payload.dynamic_constraints)
    assert payload.target_task == "refactor to async"


def test_build_selector_payload_contains_sections() -> None:
    prompt = build_selector_payload(
        "def messy():\n    pass\n",
        "use pydantic",
        "- Pattern Name: Foo\n  Origin: a.py\n",
    )
    assert "AVAILABLE GOLD-STANDARD REFERENCE CATALOG" in prompt
    assert "REFACTORING INSTRUCTIONS" in prompt
    assert "def messy" in prompt
    assert "use pydantic" in prompt
    assert "Foo" in prompt


def test_parse_selection_name_first_line_and_noise() -> None:
    known = ["create_item", "AsyncUserRepository"]
    assert parse_selection_name("create_item\n", known) == "create_item"
    assert (
        parse_selection_name("Selection: AsyncUserRepository", known)
        == "AsyncUserRepository"
    )
    assert parse_selection_name("I pick create_item for this.", known) == "create_item"
    assert parse_selection_name("nope", known) is None


def test_build_correction_uses_diagnostics_field(
    roomy_budget: ContextBudget,
) -> None:
    orch = PromptOrchestrator()
    chunk = _sample_chunk()
    base = orch.build_anchored_payload("build a router", "", chunk)
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
