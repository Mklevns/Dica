"""Module 5: Context Budget Manager.

Keeps every payload sent to the local LLM inside ``num_ctx`` using a
**BPE token counter** (:mod:`tiktoken`) so dense, tokenizer-hostile text
(base64, long identifiers, non-ASCII) is not under-counted the way a static
``chars_per_token`` heuristic is.

When :mod:`tiktoken` is unavailable, the manager falls back to the
configured character heuristic and logs a warning — better than refusing to
assemble prompts.

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

Recency layout puts system / target *first* and the task *last*. Under-counting
tokens used to let the true payload exceed ``num_ctx``; local models then
left-truncate and silently drop system instructions and the target script.
Accurate BPE counts close that hole.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from dica.config import ContextConfig, OllamaConfig

logger = logging.getLogger(__name__)

_ELISION_MARKER = "\n[... diagnostics truncated by DICA context manager ...]\n"

# Telemetry tripwire when assembled headroom is razor-thin after packing.
_HEADROOM_WARN_TOKENS = 150

# Default OpenAI-compatible BPE; close enough for packing local chat models
# and far more accurate than chars/token on high-entropy text.
_DEFAULT_ENCODING = "cl100k_base"

TokenizerBackend = Literal["tiktoken", "heuristic"]


class _Encoding(Protocol):
    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]: ...
    def decode(self, tokens: list[int], *args: Any, **kwargs: Any) -> str: ...


@lru_cache(maxsize=4)
def _load_tiktoken_encoding(name: str) -> _Encoding | None:
    """Load a named tiktoken encoding once per process; ``None`` on failure."""
    try:
        import tiktoken
    except ImportError:
        logger.warning(
            "tiktoken is not installed; context budgeting falls back to the "
            "chars_per_token heuristic (may under-count dense/base64 text). "
            "Install with: pip install tiktoken"
        )
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # encoding name typo / package skew
        logger.warning(
            "tiktoken encoding %r unavailable (%s); using chars_per_token "
            "heuristic for token estimates.",
            name,
            exc,
        )
        return None


def estimate_tokens(
    text: str,
    *,
    chars_per_token: float = 3.2,
    encoding_name: str | None = _DEFAULT_ENCODING,
) -> int:
    """Count tokens in ``text`` via BPE when possible.

    Args:
        text: Source string to measure.
        chars_per_token: Fallback density when BPE is unavailable
            (``ceil(len / chars_per_token)``).
        encoding_name: tiktoken encoding to use, or ``None`` to force the
            character heuristic.

    Returns:
        Non-negative token estimate. Empty string → 0.
    """
    if not text:
        return 0
    if encoding_name:
        enc = _load_tiktoken_encoding(encoding_name)
        if enc is not None:
            # disallowed_special=() treats special tokens as ordinary text so
            # user code containing e.g. ``<|endoftext|>`` never raises.
            return len(enc.encode(text, disallowed_special=()))
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
    tokenizer: str = "tiktoken"

    @property
    def headroom(self) -> int:
        return self.budget_tokens - self.used_tokens


class AssembledPayload(BaseModel):
    """Final prompt text plus its accounting."""

    model_config = ConfigDict(frozen=True)

    text: str
    report: BudgetReport


class ContextBudget:
    """Assembles LLM payloads that fit inside ``num_ctx`` under BPE accounting."""

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
        self._encoding_name = (
            None
            if context.tokenizer == "heuristic"
            else context.tiktoken_encoding
        )
        self._enc: _Encoding | None = None
        if self._encoding_name is not None:
            self._enc = _load_tiktoken_encoding(self._encoding_name)
            if self._enc is None:
                # Permanent heuristic fallback for this instance after load fail.
                self._encoding_name = None
        self._tokenizer_label = (
            f"tiktoken:{context.tiktoken_encoding}"
            if self._enc is not None
            else f"heuristic:{self._cpt:g}cpt"
        )
        window = num_ctx if num_ctx is not None else ollama.num_ctx
        # Apply a small safety margin so model-specific tokenizers that run
        # denser than cl100k_base still leave room for the reserved reply.
        margin = max(0.0, context.token_safety_margin)
        effective_window = int(window * (1.0 - margin)) if margin else window
        self._budget = max(256, effective_window - context.reserve_output_tokens)
        self._num_ctx = window
        logger.debug(
            "ContextBudget ready: num_ctx=%d budget=%d tokenizer=%s "
            "safety_margin=%.3f",
            self._num_ctx,
            self._budget,
            self._tokenizer_label,
            margin,
        )

    @property
    def budget_tokens(self) -> int:
        """Total token budget available for the prompt (excludes reserved output)."""
        return self._budget

    @property
    def max_diagnostic_tokens(self) -> int:
        """Configured cap for sandbox / extraction diagnostics."""
        return self._max_diag

    @property
    def tokenizer_label(self) -> str:
        """Human-readable counter identity for logs / reports."""
        return self._tokenizer_label

    def _tokens(self, text: str) -> int:
        return estimate_tokens(
            text,
            chars_per_token=self._cpt,
            encoding_name=self._encoding_name,
        )

    def _truncate_prefix(self, text: str, max_tokens: int) -> str:
        """Keep a leading prefix of ``text`` that fits in ``max_tokens``."""
        if max_tokens <= 0:
            return ""
        if self._tokens(text) <= max_tokens:
            return text
        if self._enc is not None:
            ids = self._enc.encode(text, disallowed_special=())
            return self._enc.decode(ids[:max_tokens])
        keep_chars = max(0, int(max_tokens * self._cpt))
        return text[:keep_chars]

    def _truncate_suffix(self, text: str, max_tokens: int) -> str:
        """Keep a trailing suffix of ``text`` that fits in ``max_tokens``."""
        if max_tokens <= 0:
            return ""
        if self._tokens(text) <= max_tokens:
            return text
        if self._enc is not None:
            ids = self._enc.encode(text, disallowed_special=())
            return self._enc.decode(ids[-max_tokens:])
        keep_chars = max(0, int(max_tokens * self._cpt))
        return text[-keep_chars:] if keep_chars else ""

    def truncate_middle(self, text: str, max_tokens: int) -> tuple[str, bool]:
        """Middle-out truncation preserving head and tail.

        Returns ``(text, was_truncated)``. Head gets ~60% of the allowance
        (the per-line error list up front is the densest signal), tail ~40%
        (the terminal exception / summary line). Token accounting uses BPE
        when available so the result actually fits ``max_tokens``.
        """
        if max_tokens <= 0:
            return "", True
        if self._tokens(text) <= max_tokens:
            return text, False

        marker_tokens = self._tokens(_ELISION_MARKER)
        body_budget = max(1, max_tokens - marker_tokens)
        head_budget = max(1, int(body_budget * 0.6))
        tail_budget = max(1, body_budget - head_budget)

        if self._enc is not None:
            ids = self._enc.encode(text, disallowed_special=())
            head = self._enc.decode(ids[:head_budget])
            tail = self._enc.decode(ids[-tail_budget:]) if tail_budget else ""
        else:
            max_chars = int(body_budget * self._cpt)
            head_chars = int(max_chars * 0.6)
            tail_chars = max_chars - head_chars
            head = text[:head_chars]
            tail = text[-tail_chars:] if tail_chars > 0 else ""

        # Snap to line boundaries so we never hand the model a torn line.
        if "\n" in head:
            head = head.rsplit("\n", 1)[0]
        if "\n" in tail:
            tail = tail.split("\n", 1)[-1]
        out = f"{head}{_ELISION_MARKER}{tail}"

        # If line-snapping or marker overhead still overflows, hard-cap by BPE.
        if self._tokens(out) > max_tokens:
            out = self._truncate_prefix(out, max_tokens)
        return out, True

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
        task_text = task
        parts: list[str] = [system, task_text]
        used = self._tokens(system) + self._tokens(task_text) + sep_tokens

        # Last-resort guard: fixed sections alone exceed the budget.
        if used > self._budget:
            overflow = used - self._budget
            system_tokens = self._tokens(system)
            # Keep system intact; shrink task to whatever remains.
            task_budget = max(0, self._budget - system_tokens - sep_tokens)
            task_text = self._truncate_prefix(task, task_budget)
            parts[1] = task_text
            logger.warning(
                "System+task alone overflow the context budget; "
                "tail-truncated task by ~%d tokens (tokenizer=%s).",
                overflow,
                self._tokenizer_label,
            )
            used = self._tokens(system) + self._tokens(task_text) + sep_tokens

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

        text = separator.join(parts)
        used = self._tokens(text)
        report = self._finalize_report(
            text=text,
            used=used,
            diagnostics_truncated=diagnostics_truncated,
            dropped=tuple(dropped),
        )
        return AssembledPayload(text=text, report=report)

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
            task_budget = max(0, self._budget - system_tokens - sep_tokens)
            task_text = self._truncate_prefix(task, task_budget)
            task_tokens = self._tokens(task_text)
            logger.warning(
                "System+task alone overflow the context budget; "
                "tail-truncated task by ~%d tokens (tokenizer=%s).",
                overflow,
                self._tokenizer_label,
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

        # Hard guard: if join overhead or tokenizer skew pushed us over the
        # budget, drop trailing middle sections until we fit (never drop
        # system; shrink task only if middle is already empty).
        if used > self._budget and middle:
            while middle and self._tokens(
                separator.join([system, *middle, task_text]) + "\n"
            ) > self._budget:
                middle.pop()
                dropped.append("(overflow-shed)")
                logger.warning(
                    "Context budget overflow after join; dropped a middle "
                    "section to protect system+task (tokenizer=%s).",
                    self._tokenizer_label,
                )
            text = separator.join([system, *middle, task_text]) + "\n"
            used = self._tokens(text)

        if used > self._budget:
            # System + task alone still too large: shrink task by BPE prefix.
            overflow = used - self._budget
            task_budget = max(
                0,
                self._tokens(task_text) - overflow - sep_tokens,
            )
            task_text = self._truncate_prefix(task_text, task_budget)
            text = separator.join([system, *middle, task_text]) + "\n"
            used = self._tokens(text)
            logger.error(
                "Context payload still over budget after shedding middle "
                "sections (used=%d budget=%d tokenizer=%s). Task was "
                "force-truncated to protect the system prompt from left-"
                "truncation by the model runtime.",
                used,
                self._budget,
                self._tokenizer_label,
            )

        report = self._finalize_report(
            text=text,
            used=used,
            diagnostics_truncated=diagnostics_truncated,
            dropped=tuple(dropped),
        )
        return AssembledPayload(text=text, report=report)

    def _finalize_report(
        self,
        *,
        text: str,
        used: int,
        diagnostics_truncated: bool,
        dropped: tuple[str, ...],
    ) -> BudgetReport:
        report = BudgetReport(
            budget_tokens=self._budget,
            used_tokens=used,
            diagnostics_truncated=diagnostics_truncated,
            dropped_chunks=dropped,
            tokenizer=self._tokenizer_label,
        )
        if report.headroom < _HEADROOM_WARN_TOKENS:
            logger.warning(
                "Context headroom critically low: %d tokens remaining of a "
                "%d-token budget (used=%d, tokenizer=%s, num_ctx=%d).",
                report.headroom,
                report.budget_tokens,
                report.used_tokens,
                self._tokenizer_label,
                self._num_ctx,
            )
        if used > self._budget:
            # Should be unreachable after overflow guards; loud if not.
            logger.error(
                "Context budget accounting failure: used=%d > budget=%d "
                "(tokenizer=%s). Prompt may be left-truncated by Ollama.",
                used,
                self._budget,
                self._tokenizer_label,
            )
        return report
