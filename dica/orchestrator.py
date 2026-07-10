"""Module 3: Prompt Orchestrator (context-window optimized).

Assembles the final, rigidly-delimited context payload sent to the local LLM,
with three compression / alignment optimizations over the v1 orchestrator:

1. **AST Pruning** — reference snippets are round-tripped through the ``ast``
   module with docstrings stripped (:func:`distill_syntax`). Docstrings are
   the least token-dense part of a gold snippet: the model needs the
   *structure* (signatures, decorators, typing discipline), not the prose.
2. **Dynamic Rule Extraction** — the boolean :class:`~dica.vault.ChunkTags`
   on the dispatched chunks are aggregated into explicit natural-language
   constraints ("You must use async/await ..."), so the structural intent the
   dispatcher detected is *stated*, not merely implied by example.
3. **Recency Anchoring** — the payload is reordered so the target script sits
   at the *top* (stable context), while constraints, references, and finally
   the active task sit progressively closer to the generation point. Small
   local models weight recent tokens most heavily; the task therefore renders
   last, immediately before the model begins emitting.

The payload remains a frozen Pydantic model: every prompt that ever hits the
model is a validated, serializable, loggable artifact.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from dica.dispatcher import DispatchResult
from dica.vault import ChunkTags

logger = logging.getLogger(__name__)

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
   references for brevity — your OUTPUT must still include Google-style
   docstrings on every public function and class.
4. Solve ONLY the task between the task delimiters. It is the final section
   of this prompt.
5. Emit exactly ONE Python code block (```python ... ```). No prose before
   or after the block unless explicitly asked.
6. All code must use Python 3.11+ syntax, full type annotations, and
   Pydantic v2 idioms where data models are needed.
7. Never import from the reference code; reproduce patterns, not symbols."""

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


def distill_syntax(source_code: str) -> str:
    """Strip docstrings from ``source_code`` to maximize token density.

    The source is parsed with the built-in :mod:`ast` module; the leading
    docstring expression is removed from every ``Module``, ``FunctionDef``,
    ``AsyncFunctionDef``, and ``ClassDef`` node, and the tree is re-serialized
    with :func:`ast.unparse` (which also normalizes formatting).

    A node whose body is *only* a docstring gets an ``ast.Pass`` substituted
    so the pruned tree still unparses to valid Python.

    Args:
        source_code: Python source to compress.

    Returns:
        The docstring-free source, or the original text verbatim if it does
        not parse (a broken reference must never take down payload assembly).
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        logger.warning("distill_syntax: unparseable snippet left intact")
        return source_code

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            continue
        body = node.body
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            if len(body) == 1:
                # Docstring-only body: removal would leave an empty (invalid)
                # block, so substitute a `pass` statement instead.
                body[0] = ast.Pass()
            else:
                body.pop(0)

    return ast.unparse(tree)


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
    source: str = Field(description="AST-pruned (docstring-free) source.")


class PromptPayload(BaseModel):
    """Validated, immutable schema for everything sent to the model.

    Attributes:
        system_instructions: The fixed rules-of-engagement preamble.
        references: Ordered gold-standard snippets (best match first),
            already AST-pruned at build time.
        dynamic_constraints: Tag-derived architectural rules. Declared as a
            tuple for true immutability; ``build`` may pass a plain
            ``list[str]`` and Pydantic coerces it.
        target_code: Optional messy script to refactor. ``None`` selects
            pure generation mode; a string selects refactor mode and adds
            the delimited target section to :meth:`render`.
        target_task: The user's active instruction.
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

    def render(self) -> str:
        """Serialize the payload into the recency-anchored prompt format.

        Layout (fixed order)::

            <system instructions>

            [TARGET SCRIPT TO REFACTOR]        <- only when target_code set
            <the code>
            [END TARGET SCRIPT]

            [REQUIRED ARCHITECTURAL CONSTRAINTS]
            - <rule 1>
            - <rule 2>
            [END REQUIRED ARCHITECTURAL CONSTRAINTS]

            [GOLD STANDARD REFERENCE CODE]
            # Reference 1: <name>  (from <origin>)
            <AST-pruned code>
            [END GOLD STANDARD REFERENCE CODE]

            [ACTIVE TARGET TASK]
            <task>
            [END ACTIVE TARGET TASK]

        The task renders last so it occupies the highest-recency position in
        the context window, immediately before generation begins.

        Returns:
            The complete single-string prompt.
        """
        parts: list[str] = [self.system_instructions]

        if self.target_code is not None:
            parts.append(
                f"{TARGET_OPEN}\n{self.target_code.rstrip()}\n{TARGET_CLOSE}"
            )

        constraint_lines = self.dynamic_constraints or (_FALLBACK_CONSTRAINT,)
        constraint_body = "\n".join(f"- {rule}" for rule in constraint_lines)
        parts.append(
            f"{CONSTRAINTS_OPEN}\n{constraint_body}\n{CONSTRAINTS_CLOSE}"
        )

        ref_sections: list[str] = [
            f"# Reference {i}: {ref.name}  (from {ref.origin})\n{ref.source}"
            for i, ref in enumerate(self.references, start=1)
        ]
        ref_body = (
            "\n\n".join(ref_sections)
            if ref_sections
            else "# (no references matched)"
        )
        parts.append(f"{REF_OPEN}\n{ref_body}\n{REF_CLOSE}")

        parts.append(f"{TASK_OPEN}\n{self.target_task.strip()}\n{TASK_CLOSE}")
        return "\n\n".join(parts) + "\n"


class PromptOrchestrator:
    """Builds :class:`PromptPayload` objects from dispatcher output."""

    def __init__(self, system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS) -> None:
        """Bind the fixed system preamble used for every payload.

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

        Reference sources are AST-pruned here (once, at build time) so every
        subsequent :meth:`PromptPayload.render` — including correction
        retries — pays the compressed token cost. Chunk tags are aggregated
        into explicit constraints via :func:`derive_constraints`.

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

    def build_refinement_payload(
        self,
        previous_code: str,
        result: DispatchResult,
        original_task: str,
        pass_index: int,
        total_passes: int,
    ) -> PromptPayload:
        """Constructs the payload for an iterative single-reference alignment pass.
        
        The code from the previous pass is injected as the new target script.
        Only a single gold-standard reference is included to keep the model focused,
        and the original task is rewritten to explicitly state the refinement goals.

        Args:
            previous_code: The parsed source code generated in the previous pass.
            result: The single dispatcher hit for this pass.
            original_task: The user's initial instructions.
            pass_index: Current pass number (1-indexed).
            total_passes: Total scheduled refinement passes.

        Returns:
            An immutable payload ready to be sent to the model.
        """
        refinement_task = (
            f"Pass {pass_index} of {total_passes}: Align the target script "
            "with the provided gold-standard reference code while satisfying "
            f"the original request.\n\nOriginal task: {original_task.strip()}"
        )

        references = (
            ReferenceSnippet(
                name=result.chunk.name,
                origin=result.chunk.file_path,
                relevance=round(result.total, 4),
                source=distill_syntax(result.chunk.source),
            ),
        )
        constraints = derive_constraints([result])

        return PromptPayload(
            system_instructions=self._system_instructions,
            references=references,
            dynamic_constraints=tuple(constraints),
            target_code=previous_code,
            target_task=refinement_task,
        )

    def build_correction(
        self, original: PromptPayload, failed_code: str, diagnostics: str
    ) -> PromptPayload:
        """Wrap sandbox failures into a localized self-correction payload.

        The original references, dynamic constraints, AND target script are
        preserved — the style anchor, the architectural contract, and the
        refactoring subject must not drift across retries — while the task
        is rewritten to include the failing code and the verbatim
        linter/type-checker output.

        Args:
            original: The payload whose output failed verification.
            failed_code: The extracted code that failed the gate.
            diagnostics: Verbatim ruff/mypy failure output.

        Returns:
            A new immutable payload carrying the correction task.
        """
        correction_task = (
            "Your previous attempt at the task below FAILED automated "
            "verification.\n\n"
            f"Original task:\n{original.target_task.strip()}\n\n"
            "Your failing code:\n"
            f"```python\n{failed_code.strip()}\n```\n\n"
            "Verification diagnostics (ruff + mypy):\n"
            f"```\n{diagnostics.strip()}\n```\n\n"
            "Fix EVERY reported issue and re-emit the complete corrected "
            "file as a single ```python block. Do not explain the changes."
        )
        return PromptPayload(
            system_instructions=original.system_instructions,
            references=original.references,
            dynamic_constraints=original.dynamic_constraints,
            target_code=original.target_code,
            target_task=correction_task,
        )

    def build_draft_payload(
        self, task: str, *, target_code: str | None = None
    ) -> PromptPayload:
        """Build the **Pass 0** payload: task + target, ZERO references.

        The initial draft is produced with an unpolluted context window so
        the model commits to a structure driven purely by the user's
        instructions; gold-standard patterns are layered on in later passes.
        """
        return PromptPayload(
            system_instructions=self._system_instructions,
            references=(),
            target_code=target_code,
            target_task=task,
        )