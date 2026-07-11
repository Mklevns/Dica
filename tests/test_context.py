"""Tests for context budget assembly and middle-out truncation."""

from __future__ import annotations

from dica.config import ContextConfig, OllamaConfig
from dica.context import BudgetSection, ContextBudget, estimate_tokens


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd", chars_per_token=2.0) == 2


def test_truncate_middle_preserves_head_and_tail() -> None:
    budget = ContextBudget(
        OllamaConfig(num_ctx=2048),
        ContextConfig(max_diagnostic_tokens=100),
    )
    lines = [f"error line {i}: problem with type Foo" for i in range(100)]
    text = "\n".join(lines)
    out, truncated = budget.truncate_middle(text, max_tokens=40)
    assert truncated is True
    assert "truncated by DICA" in out
    assert "error line 0" in out
    assert "error line 99" in out


def test_truncate_middle_noop_when_small() -> None:
    budget = ContextBudget(OllamaConfig(num_ctx=2048), ContextConfig())
    out, truncated = budget.truncate_middle("short", max_tokens=100)
    assert truncated is False
    assert out == "short"


def test_assemble_recency_drops_large_refs(tight_budget: ContextBudget) -> None:
    assembled = tight_budget.assemble_recency(
        system="You are a helpful assistant.",
        fixed_sections=[
            BudgetSection(name="constraints", text="[CONSTRAINTS]\n- use types\n"),
        ],
        references=[
            BudgetSection(name="big", text="x = " + ("a" * 4000)),
            BudgetSection(name="small", text="y = 1"),
        ],
        diagnostics="E" * 3000,
        task="[TASK]\ndo the thing\n[/TASK]",
    )
    assert assembled.report.used_tokens <= assembled.report.budget_tokens + 100
    assert "big" in assembled.report.dropped_chunks or assembled.report.diagnostics_truncated
    assert assembled.text.rstrip().endswith("[/TASK]") or "[/TASK]" in assembled.text
    # Task is last major section
    assert assembled.text.rfind("[TASK]") > assembled.text.find("You are a helpful")


def test_assemble_drops_fixed_when_no_room() -> None:
    # OllamaConfig enforces num_ctx >= 512; override the budget window lower
    # via ContextBudget(num_ctx=...) to force fixed-section pressure.
    budget = ContextBudget(
        OllamaConfig(num_ctx=512),
        ContextConfig(reserve_output_tokens=100, max_diagnostic_tokens=64),
        num_ctx=450,
    )
    huge_system = "SYS " * 200
    assembled = budget.assemble_recency(
        system=huge_system,
        fixed_sections=[
            BudgetSection(name="target_script", text="TARGET " * 500),
        ],
        references=[],
        task="task goes here",
    )
    assert "target_script" in assembled.report.dropped_chunks or len(
        assembled.text
    ) < len(huge_system) + 500
