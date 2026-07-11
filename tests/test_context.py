"""Tests for context budget assembly and middle-out truncation."""

from __future__ import annotations

import base64
import math
import os

from dica.config import ContextConfig, OllamaConfig
from dica.context import BudgetSection, ContextBudget, estimate_tokens


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_heuristic_path() -> None:
    assert estimate_tokens("abcd", chars_per_token=2.0, encoding_name=None) == 2


def test_estimate_tokens_bpe_counts_dense_text_heavier_than_heuristic() -> None:
    """Base64 is BPE-hostile: chars/3.2 under-counts and risks silent overflow."""
    dense = base64.b64encode(os.urandom(400)).decode("ascii")
    bpe = estimate_tokens(dense, encoding_name="cl100k_base")
    heuristic = math.ceil(len(dense) / 3.2)
    assert bpe > heuristic
    assert bpe > len(dense) / 4  # denser than a generous 4 chars/token


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
    assert budget._tokens(out) <= 40


def test_truncate_middle_noop_when_small() -> None:
    budget = ContextBudget(OllamaConfig(num_ctx=2048), ContextConfig())
    out, truncated = budget.truncate_middle("short", max_tokens=100)
    assert truncated is False
    assert out == "short"


def test_assemble_recency_drops_large_refs(tight_budget: ContextBudget) -> None:
    # High-entropy payload so BPE cannot collapse the reference the way
    # repeated ASCII letters would — mirrors real base64 / binary blobs.
    dense = base64.b64encode(os.urandom(2000)).decode("ascii")
    assembled = tight_budget.assemble_recency(
        system="You are a helpful assistant.",
        fixed_sections=[
            BudgetSection(name="constraints", text="[CONSTRAINTS]\n- use types\n"),
        ],
        references=[
            BudgetSection(name="big", text="x = " + dense),
            BudgetSection(name="small", text="y = 1"),
        ],
        diagnostics="E" * 3000,
        task="[TASK]\ndo the thing\n[/TASK]",
    )
    assert assembled.report.used_tokens <= assembled.report.budget_tokens
    assert "big" in assembled.report.dropped_chunks or assembled.report.diagnostics_truncated
    assert assembled.text.rstrip().endswith("[/TASK]") or "[/TASK]" in assembled.text
    # Task is last major section
    assert assembled.text.rfind("[TASK]") > assembled.text.find("You are a helpful")
    assert assembled.report.tokenizer.startswith("tiktoken")


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


def test_assemble_never_exceeds_budget_on_dense_payload() -> None:
    """Regression: dense target+refs must not silently pack over num_ctx."""
    budget = ContextBudget(
        OllamaConfig(num_ctx=1024),
        ContextConfig(
            reserve_output_tokens=128,
            max_diagnostic_tokens=64,
            min_chunk_tokens=16,
            token_safety_margin=0.05,
        ),
        num_ctx=1024,
    )
    dense = base64.b64encode(os.urandom(3000)).decode("ascii")
    assembled = budget.assemble_recency(
        system="SYSTEM RULES " * 20,
        fixed_sections=[
            BudgetSection(name="target_script", text="TARGET\n" + dense),
        ],
        references=[
            BudgetSection(name="ref1", text="REF\n" + dense[:800]),
        ],
        diagnostics=dense[:500],
        task="TASK: rewrite the target carefully\n" + ("detail " * 50),
    )
    assert assembled.report.used_tokens <= assembled.report.budget_tokens
    # System prompt must survive packing (would be lost under left-truncation).
    assert assembled.text.startswith("SYSTEM RULES")
