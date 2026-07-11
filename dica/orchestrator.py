"""Module 3: Prompt Orchestrator (context-window optimized).

Assembles the rigidly-delimited context payload sent to the local LLM for
**agentic selection** and **anchored generation**, with three compression /
alignment optimizations:

1. **CST docstring distillation** — reference snippets are rewritten with
   :mod:`libcst` so module/class/function docstrings are removed
   (:func:`distill_syntax`) while **inline and full-line comments stay**.
   Docstrings are the least token-dense part of a gold snippet; comments often
   carry the architectural "why" the model is told to reproduce as patterns.
2. **Dynamic Rule Extraction** — the boolean :class:`~dica.vault.ChunkTags`
   on the selected (or fallback-dispatched) chunk are aggregated into explicit
   natural-language constraints ("You must use async/await ..."), so structural
   intent is *stated*, not merely implied by example.
3. **Recency Anchoring** — the payload is reordered so the target script sits
   at the *top* (stable context), while constraints, references, and finally
   the active task sit progressively closer to the generation point. Small
   local models weight recent tokens most heavily; the task therefore renders
   last, immediately before the model begins emitting.

The generation payload remains a frozen Pydantic model: every prompt that
hits the model for code emission is a validated, serializable, loggable
artifact. Selector prompts are plain strings (short-answer name only).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import libcst as cst
import libcst.matchers as m
from pydantic import BaseModel, ConfigDict, Field

from dica.context import AssembledPayload, BudgetSection, ContextBudget
from dica.dispatcher import DispatchResult
from dica.vault import ChunkTags, CodeChunk

logger = logging.getLogger(__name__)

DIAGNOSTICS_OPEN = "[VERIFICATION DIAGNOSTICS]"
DIAGNOSTICS_CLOSE = "[END VERIFICATION DIAGNOSTICS]"

# ---------------------------------------------------------------------- #
# Sentinel delimiters — single source of truth for the prompt grammar.
# ---------------------------------------------------------------------- #
TARGET_OPEN = "[TARGET SCRIPT TO REFACTOR]"
TARGET_CLOSE = "[END TARGET SCRIPT]"
CONSTRAINTS_OPEN = "[REQUIRED ARCHITECTURAL CONSTRAINTS]"
CONSTRAINTS_CLOSE = "[END REQUIRED ARCHITECTURAL CONSTRAINTS]"
REF_OPEN = "[GOLD STANDARD REFERENCE CODE]"
REF_CLOSE = "[END GOLD STANDARD REFERENCE CODE]"
TASK_OPEN = "[ACTIVE TARGET TASK]"
TASK_CLOSE = "[END ACTIVE TARGET TASK]"

DEFAULT_SYSTEM_INSTRUCTIONS = """\
You are a senior Python engineer operating under strict alignment scaffolding.

Rules of engagement:
1. If a target-script section is present, the task is a REFACTORING order:
   rewrite that script to satisfy the task while preserving its observable
   behavior. Emit the COMPLETE refactored file, never a diff or fragment.
2. The architectural-constraints section is MANDATORY. Every listed
   constraint must hold in your output.
3. The reference code defines the REQUIRED style: naming, typing discipline,
   error handling, and async patterns. Docstrings have been stripped from the
   references for brevity, but inline comments are preserved as architectural
   guidance — your OUTPUT must still include Google-style docstrings on every
   public function and class.
4. Solve ONLY the task between the task delimiters. It is the final section
   of this prompt.
5. Emit exactly ONE Python code block (```python ... ```). No prose before
   or after the block unless explicitly asked.
6. All code must use Python 3.11+ syntax, full type annotations, and
   Pydantic v2 idioms where data models are needed.
7. Never import from the reference code; reproduce patterns, not symbols."""

SELECTOR_SYSTEM_INSTRUCTIONS = """\
You are a principal software architect. Your task is to analyze a messy target \
script, review a set of instructions, and select the ONE gold-standard reference \
pattern from the catalog that provides the best architectural blueprint for the \
refactor.

Respond with exactly ONE line containing only the Pattern Name you selected. Do not \
emit prose, reasoning, or markdown block formatting."""

CATALOG_OPEN = "[AVAILABLE GOLD-STANDARD REFERENCE CATALOG]"
CATALOG_CLOSE = "[END CATALOG]"
INSTRUCTIONS_OPEN = "[REFACTORING INSTRUCTIONS]"
INSTRUCTIONS_CLOSE = "[END REFACTORING INSTRUCTIONS]"


def build_selector_payload(
    target_code: str, instructions: str, catalog: str
) -> str:
    """Build the short-answer prompt used for agentic pattern selection.

    The model must reply with a single pattern name from ``catalog``. Full
    gold sources are not included — only the token-dense catalog menu.
    """
    target_body = target_code.strip() if target_code else "(no target script provided)"
    return (
        f"{SELECTOR_SYSTEM_INSTRUCTIONS}\n\n"
        f"{TARGET_OPEN}\n"
        f"{target_body}\n"
        f"{TARGET_CLOSE}\n\n"
        f"{INSTRUCTIONS_OPEN}\n"
        f"{instructions.strip()}\n"
        f"{INSTRUCTIONS_CLOSE}\n\n"
        f"{CATALOG_OPEN}\n"
        f"{catalog.strip()}\n"
        f"{CATALOG_CLOSE}\n\n"
        "Analyze the target script and instructions. Which pattern name is the "
        "single best match?\n"
        "Selection:"
    )


def parse_selection_name(raw: str, known_names: Sequence[str]) -> str | None:
    """Extract a vault pattern name from a noisy model selection response.

    Tries, in order: exact case-insensitive match on the first non-empty line;
    then the first known name that appears as a whole token in the response.
    """
    text = raw.strip().strip("`\"'")
    if not text:
        return None

    known_by_lower = {n.lower(): n for n in known_names}
    first_line = next(
        (ln.strip().strip("`\"'") for ln in text.splitlines() if ln.strip()),
        "",
    )
    if first_line:
        # Drop common prefixes models sometimes emit.
        for prefix in ("selection:", "pattern name:", "pattern:", "name:"):
            lower = first_line.lower()
            if lower.startswith(prefix):
                first_line = first_line[len(prefix) :].strip().strip("`\"'")
                break
        if first_line.lower() in known_by_lower:
            return known_by_lower[first_line.lower()]

    # Substring fallback: longest known name wins to avoid partial collisions.
    lowered = text.lower()
    matches = [
        canonical
        for key, canonical in known_by_lower.items()
        if key in lowered
    ]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]

# Tag predicate -> emitted constraint. Ordered, typed predicates (rather than
# getattr on a string) keep this table mypy-strict clean and refactor-safe.
_TAG_CONSTRAINTS: tuple[tuple[Callable[[ChunkTags], bool], str], ...] = (
    (
        lambda t: t.is_async,
        "You must use async/await for all I/O-bound operations; never block "
        "the event loop with synchronous I/O.",
    ),
    (
        lambda t: t.has_pydantic,
        "You must model all structured data with strict Pydantic v2 "
        "BaseModels (ConfigDict, field_validator — never v1 idioms).",
    ),
    (
        lambda t: t.is_class,
        "You must organize stateful behavior into classes mirroring the "
        "reference architecture; avoid loose module-level state.",
    ),
    (
        lambda t: t.has_decorators,
        "You must preserve and correctly apply the decorator patterns shown "
        "in the references (routing, validation, registration).",
    ),
    (
        lambda t: t.uses_typing,
        "Every function signature must carry complete type annotations, "
        "including return types; the output must pass mypy --strict.",
    ),
)

_FALLBACK_CONSTRAINT = (
    "Match the style conventions of the reference code exactly."
)


# Leading string expression that Python treats as a docstring (PEP 257).
_DOCSTRING_EXPR = m.Expr(
    value=m.OneOf(m.SimpleString(), m.ConcatenatedString())
)


def _is_docstring_statement(stmt: cst.BaseStatement) -> bool:
    """True when ``stmt`` is a standalone string-literal expression statement."""
    return m.matches(
        stmt,
        m.SimpleStatementLine(body=[_DOCSTRING_EXPR]),
    )


def _is_docstring_small_stmt(stmt: cst.BaseSmallStatement) -> bool:
    return m.matches(stmt, _DOCSTRING_EXPR)


def _has_executable_body(statements: Sequence[cst.BaseStatement]) -> bool:
    """True if any non-decorative statement remains (EmptyLine does not count)."""
    return any(not isinstance(stmt, cst.EmptyLine) for stmt in statements)


class _DocstringStripper(cst.CSTTransformer):
    """Remove module/class/function docstrings without rewriting other syntax.

    Unlike :func:`ast.unparse`, a CST transform keeps comments, ``# type:
    ignore`` annotations, and original formatting that carries gold-standard
    architectural intent.
    """

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        return updated_node.with_changes(
            body=self._strip_statements(
                updated_node.body, ensure_nonempty=False
            )
        )

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        return updated_node.with_changes(body=self._strip_suite(updated_node.body))

    def leave_AsyncFunctionDef(
        self,
        original_node: cst.AsyncFunctionDef,
        updated_node: cst.AsyncFunctionDef,
    ) -> cst.AsyncFunctionDef:
        return updated_node.with_changes(body=self._strip_suite(updated_node.body))

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        return updated_node.with_changes(body=self._strip_suite(updated_node.body))

    def _strip_suite(self, suite: cst.BaseSuite) -> cst.BaseSuite:
        if isinstance(suite, cst.SimpleStatementSuite):
            return self._strip_simple_suite(suite)
        if isinstance(suite, cst.IndentedBlock):
            new_body = self._strip_statements(
                suite.body, ensure_nonempty=True
            )
            return suite.with_changes(body=new_body)
        return suite

    def _strip_simple_suite(
        self, suite: cst.SimpleStatementSuite
    ) -> cst.SimpleStatementSuite:
        """Handle one-liner bodies like ``def f(): \"\"\"doc\"\"\"``."""
        body = list(suite.body)
        if body and _is_docstring_small_stmt(body[0]):
            body = body[1:]
        if not body:
            body = [cst.Pass()]
        return suite.with_changes(body=tuple(body))

    def _strip_statements(
        self,
        statements: Sequence[cst.BaseStatement],
        *,
        ensure_nonempty: bool,
    ) -> Sequence[cst.BaseStatement]:
        body = list(statements)
        if body and _is_docstring_statement(body[0]):
            body = body[1:]
        if ensure_nonempty and not _has_executable_body(body):
            # Docstring-only (or docstring + comment footer) suites are not
            # valid without at least one real statement.
            body.append(cst.SimpleStatementLine(body=[cst.Pass()]))
        return tuple(body)


def distill_syntax(source_code: str) -> str:
    """Strip docstrings from ``source_code`` while preserving comments.

    Uses :mod:`libcst` so token-heavy docstrings are removed without the
    destructive ``ast.parse`` / ``ast.unparse`` round-trip that drops ``#``
    comments and ``type: ignore`` pragmas.

    A suite whose body is *only* a docstring (optionally with leftover
    comment lines in the block footer) gets a ``pass`` so the result remains
    valid Python.

    Args:
        source_code: Python source to compress.

    Returns:
        The docstring-free source, or the original text verbatim if it does
        not parse (a broken reference must never take down payload assembly).
    """
    try:
        module = cst.parse_module(source_code)
    except cst.ParserSyntaxError:
        logger.warning("distill_syntax: unparseable snippet left intact")
        return source_code

    return module.visit(_DocstringStripper()).code


def derive_constraints(results: list[DispatchResult]) -> list[str]:
    """Aggregate :class:`ChunkTags` across dispatch hits into explicit rules.

    A constraint is emitted once if *any* dispatched chunk carries the
    corresponding tag — the dispatcher already decided these chunks embody
    the user's structural intent, so their union defines the contract.

    Args:
        results: Ranked dispatcher hits whose tags should be inspected.

    Returns:
        Ordered, de-duplicated constraint strings (possibly empty).
    """
    tags = [r.chunk.tags for r in results]
    return [text for predicate, text in _TAG_CONSTRAINTS if any(map(predicate, tags))]


class ReferenceSnippet(BaseModel):
    """A single gold-standard snippet embedded in the payload."""

    model_config = ConfigDict(frozen=True)

    name: str
    origin: str = Field(description="Source file the snippet was mined from.")
    relevance: float = Field(description="Dispatcher score, kept for logging.")
    source: str = Field(
        description="Docstring-stripped source (comments preserved via CST)."
    )


class PromptPayload(BaseModel):
    """Validated, immutable schema for everything sent to the model.

    Attributes:
        system_instructions: The fixed rules-of-engagement preamble.
        references: Ordered gold-standard snippets (best match first),
            already docstring-distilled at build time.
        dynamic_constraints: Tag-derived architectural rules. Declared as a
            tuple for true immutability; ``build`` may pass a plain
            ``list[str]`` and Pydantic coerces it.
        target_code: Optional messy script to refactor. ``None`` selects
            pure generation mode; a string selects refactor mode and adds
            the delimited target section to :meth:`render`.
        target_task: The user's active instruction.
        diagnostics: Optional sandbox / extraction diagnostics. When set,
            :meth:`render_budgeted` places them in their own section with
            middle-out truncation — never stuff multi-kilobyte mypy dumps
            into the task text unchecked.
    """

    model_config = ConfigDict(frozen=True)

    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS
    references: tuple[ReferenceSnippet, ...] = Field(
        description="Ordered gold-standard snippets (best match first)."
    )
    dynamic_constraints: tuple[str, ...] = Field(
        default=(),
        description="Tag-derived architectural constraints.",
    )
    target_code: str | None = Field(
        default=None,
        description="Messy source to refactor; None = generation mode.",
    )
    target_task: str = Field(min_length=1)
    diagnostics: str | None = Field(
        default=None,
        description="Optional verification / extraction diagnostics section.",
    )

    def _constraints_section(self) -> str:
        constraint_lines = self.dynamic_constraints or (_FALLBACK_CONSTRAINT,)
        constraint_body = "\n".join(f"- {rule}" for rule in constraint_lines)
        return f"{CONSTRAINTS_OPEN}\n{constraint_body}\n{CONSTRAINTS_CLOSE}"

    def _reference_sections(self) -> list[BudgetSection]:
        """One budgeted section per reference (droppable independently)."""
        if not self.references:
            return [
                BudgetSection(
                    name="(no references)",
                    text=f"{REF_OPEN}\n# (no references matched)\n{REF_CLOSE}",
                )
            ]
        sections: list[BudgetSection] = []
        for i, ref in enumerate(self.references, start=1):
            body = f"# Reference {i}: {ref.name}  (from {ref.origin})\n{ref.source}"
            # Each reference is its own droppable unit; delimiters wrap the
            # admitted subset at assemble time via a single REF block when
            # we join — here each chunk is self-describing for ranking drops.
            sections.append(
                BudgetSection(
                    name=ref.name,
                    text=body,
                )
            )
        return sections

    def render(self) -> str:
        """Serialize the full prompt with no token budget (dry-run / debug).

        Prefer :meth:`render_budgeted` for any prompt that hits the model so
        diagnostics and references respect ``num_ctx``.
        """
        parts: list[str] = [self.system_instructions]
        if self.target_code is not None:
            parts.append(
                f"{TARGET_OPEN}\n{self.target_code.rstrip()}\n{TARGET_CLOSE}"
            )
        parts.append(self._constraints_section())
        if self.diagnostics:
            parts.append(
                f"{DIAGNOSTICS_OPEN}\n{self.diagnostics.strip()}\n"
                f"{DIAGNOSTICS_CLOSE}"
            )
        ref_sections = self._reference_sections()
        if len(ref_sections) == 1 and ref_sections[0].name == "(no references)":
            parts.append(ref_sections[0].text)
        else:
            body = "\n\n".join(s.text for s in ref_sections)
            parts.append(f"{REF_OPEN}\n{body}\n{REF_CLOSE}")
        parts.append(
            f"{TASK_OPEN}\n{self.target_task.strip()}\n{TASK_CLOSE}"
        )
        return "\n\n".join(parts) + "\n"

    def render_budgeted(self, budget: ContextBudget) -> AssembledPayload:
        """Serialize into the recency-anchored layout under ``budget``.

        Layout::

            <system instructions>
            [TARGET SCRIPT TO REFACTOR]   (if target_code set; whole or drop)
            [REQUIRED ARCHITECTURAL CONSTRAINTS]
            [VERIFICATION DIAGNOSTICS]    (if diagnostics set; middle-out)
            gold reference chunks         (per-chunk drop when over budget)
            [ACTIVE TARGET TASK]          (always last)

        Returns:
            Assembled text plus a :class:`BudgetReport` for logging.
        """
        fixed: list[BudgetSection] = []
        if self.target_code is not None:
            fixed.append(
                BudgetSection(
                    name="target_script",
                    text=(
                        f"{TARGET_OPEN}\n{self.target_code.rstrip()}\n"
                        f"{TARGET_CLOSE}"
                    ),
                )
            )
        fixed.append(
            BudgetSection(name="constraints", text=self._constraints_section())
        )

        assembled = budget.assemble_recency(
            system=self.system_instructions,
            fixed_sections=fixed,
            references=self._reference_sections(),
            diagnostics=self.diagnostics,
            task=f"{TASK_OPEN}\n{self.target_task.strip()}\n{TASK_CLOSE}",
        )
        report = assembled.report
        if report.dropped_chunks or report.diagnostics_truncated:
            logger.info(
                "Prompt budget: used=%d/%d headroom=%d dropped=%s "
                "diag_trunc=%s",
                report.used_tokens,
                report.budget_tokens,
                report.headroom,
                ", ".join(report.dropped_chunks) or "(none)",
                report.diagnostics_truncated,
            )
        else:
            logger.debug(
                "Prompt budget: used=%d/%d headroom=%d",
                report.used_tokens,
                report.budget_tokens,
                report.headroom,
            )
        return assembled


class PromptOrchestrator:
    """Builds generation payloads for agentic selection and anchored emission."""

    def __init__(self, system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS) -> None:
        """Bind the fixed system preamble used for every generation payload.

        Args:
            system_instructions: Rules-of-engagement text; override for
                experiments, defaults to the module constant.
        """
        self._system_instructions = system_instructions

    def build(
        self,
        task: str,
        results: list[DispatchResult],
        *,
        target_code: str | None = None,
    ) -> PromptPayload:
        """Convert ranked dispatch results + the user task into a payload.

        Primarily used when the hybrid dispatcher supplies the reference
        (fallback path). Prefer :meth:`build_anchored_payload` after agentic
        selection. Reference sources are docstring-distilled here so correction
        retries keep the compressed form while preserving comments.

        Args:
            task: The user's instruction (generation goal or refactoring
                orders).
            results: Ranked dispatcher hits to embed as references.
            target_code: Optional messy script; when provided the payload
                renders in refactor mode.

        Returns:
            An immutable, render-ready payload.
        """
        references = tuple(
            ReferenceSnippet(
                name=r.chunk.name,
                origin=r.chunk.file_path,
                relevance=round(r.total, 4),
                source=distill_syntax(r.chunk.source),
            )
            for r in results
        )
        constraints = derive_constraints(results)
        logger.debug(
            "Orchestrator: %d references, %d dynamic constraints",
            len(references),
            len(constraints),
        )
        return PromptPayload(
            system_instructions=self._system_instructions,
            references=references,
            dynamic_constraints=tuple(constraints),
            target_code=target_code,
            target_task=task,
        )

    def build_anchored_payload(
        self,
        task: str,
        target_code: str,
        reference_chunk: CodeChunk,
    ) -> PromptPayload:
        """Build a generation payload anchored to a single agent-selected reference.

        The selected chunk is treated as 100% relevant. Dynamic constraints are
        derived from its structural tags via a lightweight :class:`DispatchResult`
        wrapper so tag → natural-language rules stay centralized.
        """
        references = (
            ReferenceSnippet(
                name=reference_chunk.name,
                origin=reference_chunk.file_path,
                relevance=1.0,
                source=distill_syntax(reference_chunk.source),
            ),
        )
        mock_result = DispatchResult(
            chunk=reference_chunk,
            lexical_score=1.0,
            structural_score=1.0,
            semantic_score=0.0,
        )
        constraints = derive_constraints([mock_result])
        logger.debug(
            "Anchored payload: reference=%s constraints=%d",
            reference_chunk.name,
            len(constraints),
        )
        return PromptPayload(
            system_instructions=self._system_instructions,
            references=references,
            dynamic_constraints=tuple(constraints),
            target_code=target_code or None,
            target_task=task,
        )

    def build_correction(
        self,
        original: PromptPayload,
        failed_code: str,
        diagnostics: str,
        *,
        budget: ContextBudget | None = None,
    ) -> PromptPayload:
        """Wrap sandbox failures into a localized self-correction payload.

        The original references, dynamic constraints, AND target script are
        preserved — the style anchor, the architectural contract, and the
        refactoring subject must not drift across retries — while the task
        is rewritten around the failing code. Verbatim checker output is
        stored on :attr:`PromptPayload.diagnostics` so
        :meth:`PromptPayload.render_budgeted` can middle-out truncate it
        instead of letting multi-kilobyte mypy dumps blow ``num_ctx``.

        Args:
            original: The payload whose output failed verification.
            failed_code: The extracted code that failed the gate.
            diagnostics: Verbatim ruff/mypy (or extraction) failure output.
            budget: Optional budget used to cap an oversized failing-code
                snippet before it is embedded in the task text.

        Returns:
            A new immutable payload carrying the correction task + diagnostics.
        """
        code_snippet = failed_code.strip()
        if budget is not None:
            # Cap the failing file so the task itself cannot dominate the window;
            # diagnostics get a separate middle-out pass at render time.
            cap = max(256, budget.max_diagnostic_tokens)
            code_snippet, truncated = budget.truncate_middle(code_snippet, cap)
            if truncated:
                logger.info(
                    "Correction payload: truncated failing code to ~%d tokens.",
                    cap,
                )

        correction_task = (
            "Your previous attempt at the task below FAILED automated "
            "verification.\n\n"
            f"Original task:\n{original.target_task.strip()}\n\n"
            "Your failing code:\n"
            f"```python\n{code_snippet}\n```\n\n"
            "Verification diagnostics appear in the "
            f"{DIAGNOSTICS_OPEN} section (may be truncated). "
            "Fix EVERY reported issue and re-emit the complete corrected "
            "file as a single ```python block. Do not explain the changes."
        )
        return PromptPayload(
            system_instructions=original.system_instructions,
            references=original.references,
            dynamic_constraints=original.dynamic_constraints,
            target_code=original.target_code,
            target_task=correction_task,
            diagnostics=diagnostics.strip() or None,
        )