"""Module 5: Context Budget Manager.

Keeps every payload sent to the local LLM inside ``num_ctx`` using a cheap,
deterministic token heuristic — no tokenizer dependency, no network call.

Priority model
--------------
Sections are admitted in strict priority order:

1. **System prompt + task** — inviolable. If these alone overflow (they
   shouldn't), the task text is tail-truncated as a last resort.
2. **Diagnostics** — capped at ``max_diagnostic_tokens`` with *middle-out*
   truncation: ruff/mypy front-load per-line errors and Python tracebacks
   end with the actual exception, so head and tail both carry signal while
   the middle is usually repetitive frame noise.
3. **Reference chunks** — admitted best-rank-first until the budget is
   exhausted. A chunk either fits whole or is dropped; truncated reference
   code is worse than absent reference code, because a half-pattern teaches
   the model a broken pattern.

The assembler returns a :class:`BudgetReport` so the orchestrator can log
exactly what was shed on each self-correction iteration.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from dica.config import ContextConfig, OllamaConfig

logger = logging.getLogger(__name__)

_ELISION_MARKER = "\n[... diagnostics truncated by DICA context manager ...]\n"

# Telemetry tripwire: the 3.2 chars/token heuristic is conservative for
# *average* Python, but dense BPE-hostile snippets (long identifiers,
# unicode, minified one-liners) can tokenize heavier than estimated. When
# estimated headroom falls below this floor, the *actual* payload may be
# brushing num_ctx — warn loudly, never block.
_HEADROOM_WARN_TOKENS = 150


def estimate_tokens(text: str, *, chars_per_token: float = 3.2) -> int:
    """Heuristic token count: ``ceil(len / chars_per_token)``.

    3.2 chars/token is deliberately conservative for dense Python source
    (BPE tokenizers average ~3.5–4 chars/token on code); over-estimating
    means we under-fill the window, which is the safe direction.
    """
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


class BudgetSection(BaseModel):
    """A named, pre-rendered payload section awaiting admission."""

    model_config = ConfigDict(frozen=True)

    name: str
    text: str


class BudgetReport(BaseModel):
    """What the assembler kept, cut, and dropped for one payload."""

    model_config = ConfigDict(frozen=True)

    budget_tokens: int
    used_tokens: int
    diagnostics_truncated: bool
    dropped_chunks: tuple[str, ...]

    @property
    def headroom(self) -> int:
        return self.budget_tokens - self.used_tokens


class AssembledPayload(BaseModel):
    """Final prompt text plus its accounting."""

    model_config = ConfigDict(frozen=True)

    text: str
    report: BudgetReport


class ContextBudget:
    """Assembles LLM payloads that provably fit inside ``num_ctx``."""

    def __init__(
        self,
        ollama: OllamaConfig,
        context: ContextConfig,
        *,
        num_ctx: int | None = None,
    ) -> None:
        self._cpt = context.chars_per_token
        self._max_diag = context.max_diagnostic_tokens
        self._min_chunk = context.min_chunk_tokens
        window = num_ctx if num_ctx is not None else ollama.num_ctx
        self._budget = max(256, window - context.reserve_output_tokens)

    @property
    def budget_tokens(self) -> int:
        """Total token budget available for the prompt (excludes reserved output)."""
        return self._budget

    @property
    def max_diagnostic_tokens(self) -> int:
        """Configured cap for sandbox / extraction diagnostics."""
        return self._max_diag

    def _tokens(self, text: str) -> int:
        return estimate_tokens(text, chars_per_token=self._cpt)

    def truncate_middle(self, text: str, max_tokens: int) -> tuple[str, bool]:
        """Middle-out truncation preserving head and tail.

        Returns ``(text, was_truncated)``. Head gets ~60% of the allowance
        (the per-line error list up front is the densest signal), tail ~40%
        (the terminal exception / summary line).
        """
        if self._tokens(text) <= max_tokens:
            return text, False
        max_chars = int(max_tokens * self._cpt)
        head_chars = int(max_chars * 0.6)
        tail_chars = max_chars - head_chars
        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars > 0 else ""
        # Snap to line boundaries so we never hand the model a torn line.
        head = head.rsplit("\n", 1)[0]
        tail = tail.split("\n", 1)[-1]
        return f"{head}{_ELISION_MARKER}{tail}", True

    def assemble(
        self,
        *,
        system: str,
        task: str,
        diagnostics: str | None = None,
        chunks: Sequence[BudgetSection] = (),
        separator: str = "\n\n",
    ) -> AssembledPayload:
        """Build the largest payload that fits the token budget.

        ``chunks`` must arrive in descending rank order — the assembler
        admits greedily from the front and drops from the back.
        """
        sep_tokens = self._tokens(separator)
        parts: list[str] = [system, task]
        used = self._tokens(system) + self._tokens(task) + sep_tokens

        # Last-resort guard: fixed sections alone exceed the budget.
        if used > self._budget:
            overflow = used - self._budget
            keep_chars = max(0, len(task) - int(overflow * self._cpt))
            parts[1] = task[:keep_chars]
            logger.warning(
                "System+task alone overflow the context budget; "
                "tail-truncated task by ~%d tokens.",
                overflow,
            )
            used = self._budget

        diagnostics_truncated = False
        if diagnostics:
            allowance = min(self._max_diag, max(0, self._budget - used))
            diag_text, diagnostics_truncated = self.truncate_middle(
                diagnostics, allowance
            )
            diag_tokens = self._tokens(diag_text) + sep_tokens
            if used + diag_tokens <= self._budget:
                parts.append(diag_text)
                used += diag_tokens
            else:  # pragma: no cover — allowance math prevents this
                diagnostics_truncated = True

        dropped: list[str] = []
        for section in chunks:
            cost = self._tokens(section.text) + sep_tokens
            if used + cost <= self._budget:
                parts.append(section.text)
                used += cost
            else:
                dropped.append(section.name)

        if dropped:
            logger.info(
                "Context budget dropped %d reference chunk(s): %s",
                len(dropped),
                ", ".join(dropped),
            )

        report = BudgetReport(
            budget_tokens=self._budget,
            used_tokens=used,
            diagnostics_truncated=diagnostics_truncated,
            dropped_chunks=tuple(dropped),
        )
        if report.headroom < _HEADROOM_WARN_TOKENS:
            logger.warning(
                "Context headroom critically low: %d tokens remaining of a "
                "%d-token budget (used=%d, chars_per_token=%.2f). The "
                "estimation heuristic may be under-counting on this "
                "payload; the true token count could exceed num_ctx.",
                report.headroom,
                report.budget_tokens,
                report.used_tokens,
                self._cpt,
            )
        return AssembledPayload(text=separator.join(parts), report=report)

    def assemble_recency(
        self,
        *,
        system: str,
        fixed_sections: Sequence[BudgetSection] = (),
        references: Sequence[BudgetSection] = (),
        diagnostics: str | None = None,
        task: str,
        separator: str = "\n\n",
    ) -> AssembledPayload:
        """Assemble a recency-anchored prompt that fits the token budget.

        Layout (matches the orchestrator grammar)::

            <system>
            <fixed sections that fit — target script, constraints, ...>
            <diagnostics, middle-out truncated when present>
            <reference chunks that fit, best-rank first>
            <task>   ← always last (highest recency)

        Priority:
        1. **System + task** — reserved first; task is tail-truncated only if
           the pair alone overflows the budget.
        2. **Fixed sections** — admitted in order while space remains
           (whole section or drop; never partial code/constraints).
        3. **Diagnostics** — capped at ``max_diagnostic_tokens`` with
           middle-out truncation.
        4. **References** — greedy best-rank-first; dropped wholly when
           they do not fit (or fall below ``min_chunk_tokens`` remaining).
        """
        sep_tokens = self._tokens(separator)

        # Reserve system + task (task last in the final string, first in budget).
        system_tokens = self._tokens(system)
        task_text = task
        task_tokens = self._tokens(task_text)
        reserved = system_tokens + task_tokens + sep_tokens
        if reserved > self._budget:
            overflow = reserved - self._budget
            keep_chars = max(0, len(task_text) - int(overflow * self._cpt))
            task_text = task_text[:keep_chars]
            task_tokens = self._tokens(task_text)
            logger.warning(
                "System+task alone overflow the context budget; "
                "tail-truncated task by ~%d tokens.",
                overflow,
            )
            reserved = min(self._budget, system_tokens + task_tokens + sep_tokens)

        used_middle = 0
        middle: list[str] = []
        dropped: list[str] = []

        def remaining() -> int:
            return self._budget - reserved - used_middle

        for section in fixed_sections:
            cost = self._tokens(section.text) + sep_tokens
            if cost <= remaining():
                middle.append(section.text)
                used_middle += cost
            else:
                dropped.append(section.name)
                logger.info(
                    "Context budget dropped fixed section %r (~%d tokens).",
                    section.name,
                    cost,
                )

        diagnostics_truncated = False
        if diagnostics:
            allowance = min(self._max_diag, max(0, remaining() - sep_tokens))
            if allowance > 0:
                diag_body, diagnostics_truncated = self.truncate_middle(
                    diagnostics, allowance
                )
                diag_block = (
                    "[VERIFICATION DIAGNOSTICS]\n"
                    f"{diag_body}\n"
                    "[END VERIFICATION DIAGNOSTICS]"
                )
                cost = self._tokens(diag_block) + sep_tokens
                if cost <= remaining():
                    middle.append(diag_block)
                    used_middle += cost
                else:
                    diagnostics_truncated = True
            else:
                diagnostics_truncated = True

        for section in references:
            cost = self._tokens(section.text) + sep_tokens
            # Skip crumbs that would burn the last tokens for little signal.
            if cost > remaining() or remaining() < self._min_chunk:
                dropped.append(section.name)
            else:
                middle.append(section.text)
                used_middle += cost

        if dropped:
            logger.info(
                "Context budget dropped %d section(s): %s",
                len(dropped),
                ", ".join(dropped),
            )

        parts = [system, *middle, task_text]
        text = separator.join(parts) + "\n"
        used = self._tokens(text)

        report = BudgetReport(
            budget_tokens=self._budget,
            used_tokens=used,
            diagnostics_truncated=diagnostics_truncated,
            dropped_chunks=tuple(dropped),
        )
        if report.headroom < _HEADROOM_WARN_TOKENS:
            logger.warning(
                "Context headroom critically low: %d tokens remaining of a "
                "%d-token budget (used=%d, chars_per_token=%.2f).",
                report.headroom,
                report.budget_tokens,
                report.used_tokens,
                self._cpt,
            )
        return AssembledPayload(text=text, report=report)
