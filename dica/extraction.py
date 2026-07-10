"""Module 4a: Hardened Code Extraction.

Pulls Python source out of an LLM's markdown response and enforces the
``ast.parse`` fail-fast gate *at the extraction boundary*, so no caller can
forget it.

Failure modes this module survives (all observed in the wild with small
local models):

* **Unclosed final fence** — the model hit its token limit mid-block, so the
  closing ``` never arrived. The old regex silently returned ``None``.
* **Fence info-string variants** — ```` ```python3 ````, ```` ``` py ````,
  trailing junk on the fence line.
* **Prose contamination** — conversational lines leaking into the fence
  ("Here is the corrected version:" *inside* the block, or a trailing
  "Let me know if..."). A bounded salvage scan trims leading/trailing
  non-code lines until the block parses.
* **Multiple blocks** — a tiny usage example alongside the real answer; the
  largest *parseable* block wins (the old code took the largest block
  whether or not it parsed).

Interior prose (chat text sandwiched between two valid statements) is *not*
repaired — that is unrecoverable without guessing, and the honest move is to
fail the attempt and let the self-correction loop feed the syntax error back
to the model.
"""

from __future__ import annotations

import ast
import logging
import re

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Closed fences: ```python / ```py / ```python3 / bare ``` — tolerant of
# trailing whitespace or junk on the opening fence line.
_CLOSED_FENCE_RE = re.compile(
    r"```[ \t]*(?:python3?|py)?[^\n]*\n(.*?)^[ \t]*```",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)

# Unclosed final fence: an opening fence with no closing fence before EOF.
_OPEN_FENCE_RE = re.compile(
    r"```[ \t]*(?:python3?|py)?[^\n]*\n(.*)\Z",
    re.DOTALL | re.IGNORECASE,
)

# Max leading/trailing lines the salvage scan may shave. Bounded so a
# pathological response costs at most _SALVAGE_SPAN**2 fast ast.parse calls.
_SALVAGE_SPAN = 15


class ExtractionResult(BaseModel):
    """Outcome of one extraction attempt, gate verdict included."""

    model_config = ConfigDict(frozen=True)

    code: str | None
    ok: bool
    error: str | None = None

    @property
    def failure_prompt(self) -> str:
        """Ready-to-embed description of the failure for a correction turn."""
        return self.error or "No Python code block was found in the response."


def _candidate_blocks(llm_response: str) -> list[str]:
    """All fenced blocks, largest first; falls back to an unclosed fence."""
    blocks = [m.group(1) for m in _CLOSED_FENCE_RE.finditer(llm_response)]
    if not blocks:
        # No closed fence anywhere — check for a truncated final block.
        tail = _OPEN_FENCE_RE.search(llm_response)
        if tail:
            blocks = [tail.group(1)]
    return sorted((b.strip() for b in blocks if b.strip()), key=len, reverse=True)


def _try_parse(code: str) -> str | None:
    """Return the SyntaxError message, or ``None`` when the code parses."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    return None


def _salvage(code: str) -> str | None:
    """Trim leading/trailing prose lines until the block parses.

    Scans every (start, end) window within ``_SALVAGE_SPAN`` lines of each
    edge, preferring the largest surviving window. Interior contamination is
    deliberately out of scope — see module docstring.
    """
    lines = code.splitlines()
    n = len(lines)
    max_start = min(_SALVAGE_SPAN, n)
    min_end = max(n - _SALVAGE_SPAN, 1)
    for start in range(max_start):
        for end in range(n, max(min_end, start), -1):
            candidate = "\n".join(lines[start:end]).strip()
            if candidate and _try_parse(candidate) is None:
                if start or end != n:
                    logger.info(
                        "Salvaged code block by trimming %d leading / %d "
                        "trailing line(s).",
                        start,
                        n - end,
                    )
                return candidate
    return None


def extract_python_code(llm_response: str) -> ExtractionResult:
    """Extract the best parseable Python block from a model response.

    The ``ast.parse`` gate lives here: ``result.ok`` is True *iff* the
    returned code compiles to an AST. Callers must treat ``ok=False`` as a
    failed generation attempt and feed ``result.failure_prompt`` back into
    the correction loop.
    """
    blocks = _candidate_blocks(llm_response)
    if not blocks:
        return ExtractionResult(
            code=None,
            ok=False,
            error=(
                "The response contained no fenced code block. Reply with "
                "the complete solution inside a single ```python fence."
            ),
        )

    first_error: str | None = None
    for block in blocks:  # largest first
        parse_error = _try_parse(block)
        if parse_error is None:
            return ExtractionResult(code=block, ok=True)
        first_error = first_error or parse_error
        salvaged = _salvage(block)
        if salvaged is not None:
            return ExtractionResult(code=salvaged, ok=True)

    return ExtractionResult(
        code=blocks[0],
        ok=False,
        error=(
            "The code block failed to parse as Python "
            f"(SyntaxError at {first_error}). Emit only valid Python inside "
            "the fence — no prose, no partial statements."
        ),
    )
