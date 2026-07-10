"""DICA — Dynamic In-Context Alignment pipeline entrypoint.

Wires the four modules into a **4-pass iterative refinement** lifecycle:

    ingest -> dispatch(top_k=3)
        -> Pass 0: initial draft   (task + target script, ZERO references)
        -> Pass 1: align to hit #1 (previous output becomes the target)
        -> Pass 2: align to hit #2 (hit #1 flushed; one reference at a time)
        -> Pass 3: align to hit #3
        -> cloud polish            (optional foundation-model pass; fail-soft)
        -> verify(final)           (ruff + mypy — the ONLY verification gate)
        -> (correct)*              (self-correction loop on the final output)

Fail-fast intermediate gate
---------------------------
Each refinement pass's output must survive a bare ``ast.parse`` before it is
allowed to become the next pass's target. This is deliberately NOT the full
ruff/mypy gate — that stays reserved for the final output — it is a
microsecond-cheap regression tripwire: a pass that emits syntactically
broken Python is discarded and the schedule rolls back to the previous
known-parseable state instead of feeding garbage forward and compounding
the damage across the remaining passes.

Cloud Polish stage
------------------
Between the final local refinement pass and the verification gate, the
pipeline offers the assembled draft to :func:`cloud_polish` — a stub for a
frontier-model "final polish" round-trip. The stage is strictly fail-soft:
any exception (unconfigured backend, network failure, rate limit) logs a
warning and the local draft proceeds to the sandbox untouched. A polished
result must also survive the same ``ast.parse`` tripwire before it is
allowed to replace the local draft.

Extraction contract
-------------------
``dica.sandbox.extract_python_code`` returns an :class:`ExtractionResult`
Pydantic model, NOT a raw string. All call sites must unwrap the string
payload via :func:`unwrap_extraction` before treating it as code.

Usage::

    python main.py "Build an async CRUD router for a User resource"
    python main.py --target messy.py "Refactor to Pydantic v2 + async I/O"
    python main.py --dry-run "..."          # no model; prints Pass 0 payload
    python main.py --model phi4 --max-attempts 3 "..."

Requires an Ollama server on localhost (``ollama serve``) with the target
model pulled (``ollama pull phi4``). Fully offline; the only network hop is
loopback to Ollama (the cloud-polish stub raises until a backend is wired).

Event-loop shape
----------------
Everything below ``run_pipeline`` is a single coroutine tree driven by
``asyncio.run``. Each pass is one awaited HTTP round-trip; the final
verification fans out into two concurrent subprocesses (see
``dica.sandbox``). Nothing in the hot path blocks the loop, so this scales
to N parallel pipelines with a plain ``asyncio.gather`` if you ever batch
queries.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import logging
import sys
from pathlib import Path
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict

from dica.dispatcher import DispatchResult, IntentDispatcher
from dica.orchestrator import PromptOrchestrator, PromptPayload
from dica.sandbox import (
    ExtractionResult,
    cleanup_orphaned_containers,
    extract_python_code,
    verify,
)
from dica.vault import CodeVault

logger = logging.getLogger("dica")

OLLAMA_URL: Final[str] = "http://localhost:11434/api/chat"
DEFAULT_CORPUS: Final[Path] = Path(__file__).parent / "reference_corpus"

REFINEMENT_PASSES: Final[int] = 3   # Passes 1..3, one reference chunk each
DISPATCH_TOP_K: Final[int] = 3      # one dispatcher hit per refinement pass
FORMAT_RETRIES: Final[int] = 2      # extra tries per pass when no code fence


class LLMConfig(BaseModel):
    """Local inference settings."""

    model_config = ConfigDict(frozen=True)

    model: str = "qwen3-coder:30b"
    temperature: float = 0.2  # low temp: we want compliance, not creativity
    num_ctx: int = 8192       # payload + references need headroom
    timeout: float = 300.0


class PipelineInferenceError(RuntimeError):
    """Raised when the model backend is unreachable or returns garbage.

    Distinguishes *infrastructure* failures (abort the pipeline) from
    *protocol* failures like a missing code fence (retry within the pass).
    """


class CloudPolishError(RuntimeError):
    """Raised when the cloud-polish backend is unavailable or misbehaves.

    Deliberately a distinct type from :class:`PipelineInferenceError`:
    a *local* inference failure aborts the pipeline, whereas a *cloud*
    polish failure is always survivable — the caller falls back to the
    local draft and continues to the verification gate.
    """


class LocalLLMClient:
    """Thin async client for Ollama's /api/chat endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    async def complete(self, prompt: str) -> str:
        """One non-streaming chat round-trip. Raises on transport failure."""
        request = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_ctx": self._config.num_ctx,
            },
        }
        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            response = await client.post(OLLAMA_URL, json=request)
            response.raise_for_status()
            data = response.json()
        content = data.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content:
            raise RuntimeError(f"Malformed Ollama response: {data!r}")
        return content


def unwrap_extraction(result: ExtractionResult | None) -> str | None:
    """Normalize an :class:`ExtractionResult` into a plain code string.

    The extraction module returns a Pydantic model so it can carry
    diagnostics alongside the payload; the pipeline, however, only ever
    threads *strings* through its passes, line counters, and the sandbox.
    This is the single unwrap point — no other code in this module (or in
    ``app.py``, which imports this helper) may touch the model directly.

    Args:
        result: The value returned by ``extract_python_code``, or ``None``
            (tolerated for forward/backward compatibility with either
            return-type convention).

    Returns:
        The extracted source string, or ``None`` when no usable fenced
        block was found (missing result or empty ``code`` field).
    """
    if result is None:
        return None
    code = result.code
    if not code:
        return None
    return code


async def cloud_polish(code: str, task: str) -> str:
    """**Stub** — final-polish pass through a frontier foundation model.

    Intended contract once a backend is wired in: send ``code`` plus the
    original ``task`` to a cloud model with a "polish only — do not change
    observable behavior" instruction, and return the complete polished
    file as a single string.

    The stub raises :class:`CloudPolishError` unconditionally, which the
    pipeline treats as "stage unavailable" and falls back to the local
    draft. Replace the body with a real client (Anthropic API, etc.)
    without changing the signature and the wiring below keeps working.

    Args:
        code: The final local-pass output to polish.
        task: The user's original instruction, for context.

    Returns:
        The polished complete source file.

    Raises:
        CloudPolishError: Always, until a backend is configured.
    """
    raise CloudPolishError(
        "cloud polish backend is not configured (stub implementation); "
        f"skipping polish of {code.count(chr(10)) + 1}-line draft for "
        f"task {task[:48]!r}..."
    )


def _syntax_regression(code: str) -> SyntaxError | None:
    """Cheap fail-fast gate: does ``code`` at least parse?

    Runs a bare ``ast.parse`` — no linting, no typing — so it costs
    microseconds and never blocks the event loop meaningfully. Returning the
    exception (instead of raising) keeps the caller's control flow linear:
    ``None`` means the code is structurally sound enough to feed forward.

    Args:
        code: Candidate source emitted by a refinement pass.

    Returns:
        The captured :class:`SyntaxError` on failure, else ``None``.
    """
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return exc
    return None


async def _generate_code(
    client: LocalLLMClient,
    orchestrator: PromptOrchestrator,
    payload: PromptPayload,
    *,
    label: str,
) -> tuple[str | None, PromptPayload]:
    """One pipeline pass: call the model and extract its code block.

    Protocol violations (no ```python fence) are handled *inside* the pass:
    the payload is rewrapped with a format-correction task and retried up to
    ``FORMAT_RETRIES`` extra times, so a single malformed response never
    derails the multi-pass schedule.

    The extraction result is a Pydantic :class:`ExtractionResult`; the raw
    string payload is unwrapped here via :func:`unwrap_extraction`, so
    callers always receive ``str | None`` — never the model object.

    Args:
        client: The live model client.
        orchestrator: Used to build format-correction payloads.
        payload: The payload to render and send.
        label: Human-readable pass name for logging.

    Returns:
        ``(code, last_payload)`` — ``code`` is ``None`` only if every retry
        emitted no fenced block; ``last_payload`` is whatever was sent last
        (the anchor for any subsequent :meth:`build_correction` call).

    Raises:
        PipelineInferenceError: On transport / backend failure.
    """
    for attempt in range(1, FORMAT_RETRIES + 2):
        try:
            response = await client.complete(payload.render())
        except (httpx.HTTPError, RuntimeError) as exc:
            raise PipelineInferenceError(str(exc)) from exc

        result: ExtractionResult | None = extract_python_code(response)
        code = unwrap_extraction(result)
        if code is not None:
            return code, payload

        logger.warning(
            "%s: attempt %d emitted no ```python block — issuing format "
            "correction.",
            label,
            attempt,
        )
        payload = orchestrator.build_correction(
            payload,
            failed_code="(no code block emitted)",
            diagnostics="Response contained no ```python fenced block.",
        )
    return None, payload


def _read_target(target: Path | None) -> str | None:
    """Load the optional messy script to refactor.

    Args:
        target: Path from ``--target``, or ``None`` for pure generation.

    Returns:
        File contents, or ``None`` when no target was given.

    Raises:
        OSError: If the path was given but cannot be read.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    if target is None:
        return None
    return target.read_text(encoding="utf-8")


async def run_pipeline(
    query: str,
    *,
    corpus: Path,
    llm: LLMConfig,
    target: Path | None = None,
    max_attempts: int = 3,
    dry_run: bool = False,
) -> int:
    """Execute the full 4-pass DICA lifecycle. Returns a process exit code."""
    # ---- Stage 1: ingest ------------------------------------------------ #
    vault = CodeVault()
    n = vault.ingest_path(corpus)
    logger.info("Stage 1 [vault]        indexed %d chunks from %s", n, corpus)
    if n == 0:
        logger.error("Empty vault — add .py files to %s", corpus)
        return 1

    # ---- Stage 2: dispatch (top_k=3 — one hit per refinement pass) ------- #
    dispatcher = IntentDispatcher(vault)
    hits: list[DispatchResult] = dispatcher.dispatch(query, top_k=DISPATCH_TOP_K)
    logger.info(
        "Stage 2 [dispatcher]   matched: %s",
        ", ".join(f"{h.chunk.name} ({h.total:.2f})" for h in hits) or "(none)",
    )
    schedule = hits[:REFINEMENT_PASSES]
    if len(schedule) < REFINEMENT_PASSES:
        logger.warning(
            "Only %d dispatcher hit(s) available — running %d refinement "
            "pass(es) instead of %d.",
            len(schedule),
            len(schedule),
            REFINEMENT_PASSES,
        )

    # ---- Stage 3: orchestrate Pass 0 (draft: ZERO references) ------------ #
    try:
        target_code = _read_target(target)
    except (OSError, UnicodeDecodeError) as exc:
        logger.error("Cannot read --target %s: %s", target, exc)
        return 1

    orchestrator = PromptOrchestrator()
    draft_payload: PromptPayload = orchestrator.build_draft_payload(
        query, target_code=target_code
    )
    logger.info(
        "Stage 3 [orchestrator] Pass 0 draft payload rendered (%d chars, "
        "0 references)",
        len(draft_payload.render()),
    )

    if dry_run:
        banner = "=" * 72
        print(f"\n{banner}\nDRY RUN — Pass 0 draft payload:\n{banner}")
        print(draft_payload.render())
        print(f"{banner}\nPlanned refinement schedule:")
        for i, hit in enumerate(schedule, start=1):
            print(f"  Pass {i}: {hit.chunk.name} (score {hit.total:.2f})")
        return 0

    client = LocalLLMClient(llm)

    # ---- Stage 4a: Pass 0 — initial draft --------------------------------- #
    try:
        logger.info("Stage 4 [pass 0/%d]    drafting (no references) ...",
                    len(schedule))
        current_code, _ = await _generate_code(
            client, orchestrator, draft_payload, label="pass 0"
        )
    except PipelineInferenceError as exc:
        logger.error("Inference failed: %s (is `ollama serve` running?)", exc)
        return 1
    if current_code is None:
        logger.error("Pass 0 never produced a code block — aborting.")
        return 1
    final_payload: PromptPayload = draft_payload

    # ---- Stage 4b: Passes 1..3 — iterative single-reference alignment ----- #
    # Each pass FLUSHES the previous reference, injects exactly ONE new hit,
    # and feeds the previous pass's code back in as the target script. A
    # fail-fast `ast.parse` gate guards every hand-off: a pass whose output
    # does not even parse is a regression, and the pipeline rolls back to
    # the previous pass's code rather than compounding the breakage.
    for pass_no, hit in enumerate(schedule, start=1):
        payload = orchestrator.build_refinement_payload(
            previous_code=current_code,
            result=hit,
            original_task=query,
            pass_index=pass_no,
            total_passes=len(schedule),
        )
        logger.info(
            "Stage 4 [pass %d/%d]    aligning to '%s' (score %.2f) ...",
            pass_no,
            len(schedule),
            hit.chunk.name,
            hit.total,
        )
        try:
            code, payload = await _generate_code(
                client, orchestrator, payload, label=f"pass {pass_no}"
            )
        except PipelineInferenceError as exc:
            logger.error(
                "Inference failed: %s (is `ollama serve` running?)", exc
            )
            return 1
        if code is None:
            # Graceful degradation: this pass's alignment is skipped, but the
            # schedule advances with the previous pass's output intact.
            logger.warning(
                "Pass %d yielded no usable code after retries — carrying "
                "pass %d output forward.",
                pass_no,
                pass_no - 1,
            )
            continue

        # ---- fail-fast intermediate verification (ast.parse gate) -------- #
        syntax_error = _syntax_regression(code)
        if syntax_error is not None:
            logger.warning(
                "Pass %d introduced a syntax regression (line %s: %s) — "
                "rolling back to pass %d output and continuing the schedule.",
                pass_no,
                syntax_error.lineno,
                syntax_error.msg,
                pass_no - 1,
            )
            continue

        current_code = code
        final_payload = payload

    # ---- Stage 4b': Cloud Polish (foundation-model pass; fail-soft) ------- #
    # Runs AFTER the final local refinement pass and BEFORE the verification
    # gate. Strictly optional: any failure logs a warning and the local
    # draft proceeds unchanged. Polished output must survive the same
    # ast.parse tripwire the refinement passes are held to before it may
    # replace the local draft.
    logger.info("Stage 4 [cloud]        offering draft for cloud polish ...")
    try:
        polished = await cloud_polish(current_code, query)
    except Exception as exc:
        logger.warning(
            "Cloud polish unavailable (%s) — falling back to the local "
            "draft for verification.",
            exc,
        )
    else:
        polish_regression = _syntax_regression(polished)
        if polish_regression is None:
            current_code = polished
            logger.info(
                "Stage 4 [cloud]        polish accepted (%d lines).",
                current_code.count("\n") + 1,
            )
        else:
            logger.warning(
                "Cloud polish output failed ast.parse (line %s: %s) — "
                "discarding and keeping the local draft.",
                polish_regression.lineno,
                polish_regression.msg,
            )

    # ---- Stage 4c: verification gate (final output ONLY) ------------------ #
    n_lines = current_code.count("\n") + 1
    logger.info("Stage 4 [sandbox]      verifying final output (%d lines) ...",
                n_lines)
    report = await verify(current_code)

    # ---- Stage 4d: self-correction loop on the final output --------------- #
    correction = 0
    while not report.passed and correction < max_attempts:
        correction += 1
        logger.warning(
            "Verification failed (correction %d/%d). Diagnostics:\n%s",
            correction,
            max_attempts,
            report.diagnostics,
        )
        # Localized self-correction: same reference and target anchor, task
        # rewritten around the failing code + verbatim checker output.
        final_payload = orchestrator.build_correction(
            final_payload, current_code, report.diagnostics
        )
        try:
            code, final_payload = await _generate_code(
                client, orchestrator, final_payload,
                label=f"correction {correction}",
            )
        except PipelineInferenceError as exc:
            logger.error(
                "Inference failed: %s (is `ollama serve` running?)", exc
            )
            return 1
        if code is None:
            logger.error(
                "Correction %d produced no code block — stopping.", correction
            )
            break
        current_code = code
        report = await verify(current_code)

    banner = "=" * 72
    if report.passed:
        print(f"\n{banner}\n✓ VERIFIED OUTPUT (ruff + mypy clean)\n{banner}")
        print(current_code)
        return 0

    logger.error(
        "Exhausted the correction budget without passing the quality gate."
    )
    print(f"\n{banner}\n✗ LAST (UNVERIFIED) OUTPUT\n{banner}")
    print(current_code)
    return 2


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamic In-Context Alignment pipeline (4-pass refinement)"
    )
    parser.add_argument("query", help="Target task, e.g. 'Build an async CRUD router'")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--target", type=Path, default=None,
        help="Messy script to refactor; omit for pure generation from the task",
    )
    parser.add_argument("--model", default="phi4")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Render the Pass 0 payload and the refinement "
                             "schedule, then exit without calling the model")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    # Startup hygiene: reclaim sandbox containers orphaned by a previous
    # hard crash / SIGKILL. Runs BEFORE asyncio.run — no loop exists yet,
    # so the synchronous Docker SDK sweep cannot block one. Fail-soft: an
    # unreachable daemon logs and proceeds.
    cleanup_orphaned_containers()
    exit_code = asyncio.run(
        run_pipeline(
            args.query,
            corpus=args.corpus,
            llm=LLMConfig(model=args.model),
            target=args.target,
            max_attempts=args.max_attempts,
            dry_run=args.dry_run,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
