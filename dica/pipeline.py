"""Shared multi-pass DICA pipeline engine.

Single source of truth for the lifecycle used by the CLI (``main.py``) and
the Gradio UI (``app.py``):

    ingest -> dispatch
        -> Pass 0 draft (ZERO references; abort if ``ast.parse`` fails)
        -> Passes 1..N single-reference alignment (rollback on regression)
        -> cloud polish (fail-soft stub)
        -> verify (ruff + mypy)
        -> self-correction loop

Both adapters consume :meth:`PipelineEngine.run`, which is an async
generator of :class:`PipelineEvent` snapshots. The CLI turns events into
logs/exit codes; the UI turns them into live code/log panels.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from dica.config import OllamaConfig, get_config
from dica.context import ContextBudget
from dica.dispatcher import DispatchResult, IntentDispatcher
from dica.embeddings import OllamaEmbedder, SemanticIndex
from dica.orchestrator import PromptOrchestrator, PromptPayload
from dica.sandbox import ExtractionResult, extract_python_code, verify
from dica.vault import CodeVault

logger = logging.getLogger(__name__)

DEFAULT_CORPUS: Final[Path] = Path(__file__).resolve().parent.parent / "reference_corpus"
REFINEMENT_PASSES: Final[int] = 3
FORMAT_RETRIES: Final[int] = 2


class SupportsComplete(Protocol):
    """Structural type for any async LLM client the engine can drive."""

    async def complete(self, prompt: str) -> str:
        """Return the model's raw text response for ``prompt``."""
        ...


class LLMConfig(BaseModel):
    """Local inference settings (normally derived from :class:`OllamaConfig`)."""

    model_config = ConfigDict(frozen=True)

    model: str = "qwen3-coder:30b"
    temperature: float = 0.2
    num_ctx: int = 8192
    timeout: float = 300.0
    host: str = "http://localhost:11434"

    @classmethod
    def from_ollama(
        cls, ollama: OllamaConfig, *, model: str | None = None
    ) -> LLMConfig:
        """Build from unified config; optional ``model`` overrides the TOML value."""
        return cls(
            model=ollama.model if model is None else model,
            temperature=ollama.temperature,
            num_ctx=ollama.num_ctx,
            timeout=ollama.request_timeout_seconds,
            host=ollama.host,
        )


class PipelineInferenceError(RuntimeError):
    """Raised when the model backend is unreachable or returns garbage."""


class CloudPolishError(RuntimeError):
    """Raised when the cloud-polish backend is unavailable or misbehaves."""


class LocalLLMClient:
    """Thin async client for Ollama's /api/chat endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._url = f"{config.host.rstrip('/')}/api/chat"

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
            response = await client.post(self._url, json=request)
            response.raise_for_status()
            data = response.json()
        content = data.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content:
            raise RuntimeError(f"Malformed Ollama response: {data!r}")
        return content


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """One progress snapshot from the shared pipeline.

    Adapters map these into logs (CLI) or UI panels (Gradio). Terminal
    outcomes set ``finished=True`` and an ``exit_code`` (0 verified, 1 infra
    failure, 2 exhausted corrections / unverified).
    """

    message: str
    code: str | None = None
    level: str = "info"  # info | warning | error
    fatal: bool = False
    finished: bool = False
    verified: bool | None = None
    exit_code: int | None = None
    dry_run_text: str | None = None
    schedule_lines: tuple[str, ...] = ()


def unwrap_extraction(result: ExtractionResult | None) -> str | None:
    """Accept only gate-passing extractions (``ok=True`` + non-empty code)."""
    if result is None or not result.ok:
        return None
    code = result.code
    if not code:
        return None
    return code


async def cloud_polish(code: str, task: str) -> str:
    """**Stub** — final-polish pass through a frontier foundation model.

    Raises :class:`CloudPolishError` until a backend is configured; the
    pipeline treats that as fail-soft and keeps the local draft.
    """
    raise CloudPolishError(
        "cloud polish backend is not configured (stub implementation); "
        f"skipping polish of {code.count(chr(10)) + 1}-line draft for "
        f"task {task[:48]!r}..."
    )


def syntax_regression(code: str) -> SyntaxError | None:
    """Cheap fail-fast gate: return SyntaxError if ``code`` does not parse."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return exc
    return None


def count_lines(code: str) -> int:
    """Line count for log messages."""
    return code.count("\n") + 1


@dataclass(slots=True)
class _GenResult:
    """Mutable holder for one generation pass (async-gen cannot return values)."""

    payload: PromptPayload
    code: str | None = None
    fatal: bool = False


class PipelineEngine:
    """Owns vault/dispatcher/orchestrator and runs the full multi-pass lifecycle.

    Construct once (UI) or per invocation (CLI). Inject any
    :class:`SupportsComplete` client for tests or alternate backends.
    """

    def __init__(
        self,
        client: SupportsComplete | None = None,
        corpus: Path = DEFAULT_CORPUS,
        *,
        max_attempts: int | None = None,
        num_ctx: int | None = None,
    ) -> None:
        cfg = get_config()
        self._client = client
        self._corpus = corpus
        self._vault = CodeVault()
        self._chunk_count = self._vault.ingest_path(corpus)

        # Optional semantic side-table (fail-soft). Built only when weight > 0;
        # if Ollama/embed model is unavailable the index reports available=False
        # and the dispatcher falls back to lexical + structural scoring.
        semantic_index: SemanticIndex | None = None
        semantic_ready = False
        if cfg.dispatch.semantic_weight > 0.0 and self._chunk_count > 0:
            semantic_index = SemanticIndex(OllamaEmbedder(cfg.ollama))
            try:
                embedded = semantic_index.build_sync(self._vault)
                semantic_ready = semantic_index.available
                logger.info(
                    "Semantic index: embedded %d/%d chunks (available=%s, "
                    "weight=%.2f).",
                    embedded,
                    self._chunk_count,
                    semantic_ready,
                    cfg.dispatch.semantic_weight,
                )
            except Exception as exc:  # never let index build kill the pipeline
                logger.warning(
                    "Semantic index build failed (%s); using lexical dispatch.",
                    exc,
                )
                semantic_index = None

        self._dispatcher = IntentDispatcher(
            self._vault,
            config=cfg.dispatch,
            semantic_index=semantic_index,
        )
        self._orchestrator = PromptOrchestrator()
        self._max_attempts = (
            cfg.engine.max_retries if max_attempts is None else max_attempts
        )
        # Prefer the live LLM context window when the concrete client carries
        # one; otherwise fall back to config.toml / OllamaConfig defaults.
        ctx_window = num_ctx
        if ctx_window is None and isinstance(client, LocalLLMClient):
            ctx_window = client._config.num_ctx
        self._budget = ContextBudget(
            cfg.ollama, cfg.context, num_ctx=ctx_window
        )
        logger.info(
            "PipelineEngine ready: %d chunks from %s "
            "(top_k=%d, max_retries=%d, prompt_budget=%d tokens, "
            "semantic=%s)",
            self._chunk_count,
            corpus,
            cfg.dispatch.top_k,
            self._max_attempts,
            self._budget.budget_tokens,
            "on" if semantic_ready else "off",
        )

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    async def _generate(
        self,
        result: _GenResult,
        *,
        label: str,
        code_so_far: str | None,
    ) -> AsyncIterator[PipelineEvent]:
        """One model call + format retries; updates ``result`` in place."""
        if self._client is None:
            result.fatal = True
            yield PipelineEvent(
                message="No LLM client configured (cannot generate).",
                code=code_so_far,
                level="error",
                fatal=True,
                finished=True,
                exit_code=1,
            )
            return

        payload = result.payload
        for attempt in range(1, FORMAT_RETRIES + 2):
            yield PipelineEvent(
                message=(
                    f"{label}: attempt {attempt}/{FORMAT_RETRIES + 1} — "
                    "calling local model ..."
                ),
                code=code_so_far,
            )
            assembled = payload.render_budgeted(self._budget)
            yield PipelineEvent(
                message=(
                    f"{label}: prompt budget used={assembled.report.used_tokens}/"
                    f"{assembled.report.budget_tokens} "
                    f"(dropped={len(assembled.report.dropped_chunks)}, "
                    f"diag_trunc={assembled.report.diagnostics_truncated})"
                ),
                code=code_so_far,
            )
            try:
                response = await self._client.complete(assembled.text)
            except Exception as exc:
                logger.exception("Inference failed")
                result.payload = payload
                result.fatal = True
                yield PipelineEvent(
                    message=(
                        f"ERROR: inference failed: {exc} "
                        "(is `ollama serve` running?)"
                    ),
                    code=code_so_far,
                    level="error",
                    fatal=True,
                    finished=True,
                    exit_code=1,
                )
                return

            # extract_python_code may run up to ~_SALVAGE_SPAN**2 ast.parse
            # calls on a dirty fence; keep that CPU work off the event loop.
            extraction: ExtractionResult | None = await asyncio.to_thread(
                extract_python_code, response
            )
            code = unwrap_extraction(extraction)
            if code is not None:
                result.payload = payload
                result.code = code
                return

            failure = (
                extraction.failure_prompt
                if extraction is not None
                else "No Python code block was found in the response."
            )
            failed_snippet = (
                extraction.code
                if extraction is not None and extraction.code
                else "(no code block emitted)"
            )
            yield PipelineEvent(
                message=(
                    f"{label}: extraction failed — issuing format correction "
                    f"({failure})"
                ),
                code=code_so_far,
                level="warning",
            )
            payload = self._orchestrator.build_correction(
                payload,
                failed_code=failed_snippet,
                diagnostics=failure,
                budget=self._budget,
            )

        result.payload = payload
        result.code = None

    async def run(
        self,
        task: str,
        *,
        target_code: str | None = None,
        dry_run: bool = False,
    ) -> AsyncIterator[PipelineEvent]:
        """Execute the full lifecycle, yielding progress events.

        Args:
            task: User instruction (generation or refactor orders).
            target_code: Optional messy script contents for refactor mode.
            dry_run: If True, render Pass 0 payload + schedule and finish
                without calling the model.
        """
        if self._chunk_count == 0:
            yield PipelineEvent(
                message=f"Empty vault — add .py files to {self._corpus}",
                level="error",
                fatal=True,
                finished=True,
                exit_code=1,
            )
            return

        yield PipelineEvent(
            message=f"vault: {self._chunk_count} gold chunks indexed"
        )

        hits: list[DispatchResult] = await self._dispatcher.dispatch(task)
        matched = (
            ", ".join(f"{h.chunk.name} ({h.total:.2f})" for h in hits)
            or "(none)"
        )
        yield PipelineEvent(message=f"dispatcher: matched {matched}")

        schedule = hits[:REFINEMENT_PASSES]
        if len(schedule) < REFINEMENT_PASSES:
            yield PipelineEvent(
                message=(
                    f"dispatcher: only {len(schedule)} hit(s) available — "
                    f"running {len(schedule)} refinement pass(es) "
                    f"instead of {REFINEMENT_PASSES}"
                ),
                level="warning",
            )

        draft_payload = self._orchestrator.build_draft_payload(
            task, target_code=target_code
        )
        draft_assembled = draft_payload.render_budgeted(self._budget)
        yield PipelineEvent(
            message=(
                f"orchestrator: pass 0 draft payload rendered "
                f"({len(draft_assembled.text)} chars, "
                f"~{draft_assembled.report.used_tokens} tokens, "
                f"0 references planned)"
            )
        )

        if dry_run:
            schedule_lines = tuple(
                f"  Pass {i}: {hit.chunk.name} (score {hit.total:.2f})"
                for i, hit in enumerate(schedule, start=1)
            )
            yield PipelineEvent(
                message="dry-run: Pass 0 payload and schedule ready",
                dry_run_text=draft_assembled.text,
                schedule_lines=schedule_lines,
                finished=True,
                exit_code=0,
            )
            return

        # ---- Pass 0 ------------------------------------------------------ #
        yield PipelineEvent(
            message=f"pass 0/{len(schedule)}: drafting (no references) ..."
        )
        gen = _GenResult(payload=draft_payload)
        async for event in self._generate(gen, label="pass 0", code_so_far=None):
            yield event
            if gen.fatal:
                return

        if gen.code is None:
            yield PipelineEvent(
                message="pass 0 never produced a code block — aborting.",
                level="error",
                fatal=True,
                finished=True,
                exit_code=1,
            )
            return

        syntax_error = syntax_regression(gen.code)
        if syntax_error is not None:
            yield PipelineEvent(
                message=(
                    f"pass 0: draft failed ast.parse "
                    f"(line {syntax_error.lineno}: {syntax_error.msg}) — aborting."
                ),
                level="error",
                fatal=True,
                finished=True,
                exit_code=1,
            )
            return

        current_code = gen.code
        final_payload = gen.payload
        yield PipelineEvent(
            message=f"pass 0: draft captured ({count_lines(current_code)} lines)",
            code=current_code,
        )

        # ---- Passes 1..N ------------------------------------------------- #
        for pass_no, hit in enumerate(schedule, start=1):
            payload = self._orchestrator.build_refinement_payload(
                previous_code=current_code,
                result=hit,
                original_task=task,
                pass_index=pass_no,
                total_passes=len(schedule),
            )
            yield PipelineEvent(
                message=(
                    f"pass {pass_no}/{len(schedule)}: aligning to "
                    f"'{hit.chunk.name}' (score {hit.total:.2f})"
                ),
                code=current_code,
            )
            gen = _GenResult(payload=payload)
            async for event in self._generate(
                gen, label=f"pass {pass_no}", code_so_far=current_code
            ):
                yield event
                if gen.fatal:
                    return

            if gen.code is None:
                yield PipelineEvent(
                    message=(
                        f"pass {pass_no}: no usable code after retries — "
                        f"carrying pass {pass_no - 1} output forward"
                    ),
                    code=current_code,
                    level="warning",
                )
                continue

            syntax_error = syntax_regression(gen.code)
            if syntax_error is not None:
                yield PipelineEvent(
                    message=(
                        f"pass {pass_no}: syntax regression "
                        f"(line {syntax_error.lineno}: {syntax_error.msg}) — "
                        f"rolling back to pass {pass_no - 1} output"
                    ),
                    code=current_code,
                    level="warning",
                )
                continue

            current_code = gen.code
            final_payload = gen.payload
            yield PipelineEvent(
                message=(
                    f"pass {pass_no}: aligned output captured "
                    f"({count_lines(current_code)} lines)"
                ),
                code=current_code,
            )

        # ---- Cloud polish (fail-soft) ------------------------------------ #
        yield PipelineEvent(
            message="cloud: offering draft for cloud polish ...",
            code=current_code,
        )
        try:
            polished = await cloud_polish(current_code, task)
        except Exception as exc:
            logger.warning("Cloud polish unavailable: %s", exc)
            yield PipelineEvent(
                message=(
                    f"cloud: polish unavailable ({exc}) — "
                    "falling back to the local draft"
                ),
                code=current_code,
                level="warning",
            )
        else:
            polish_err = syntax_regression(polished)
            if polish_err is None:
                current_code = polished
                yield PipelineEvent(
                    message=(
                        f"cloud: polish accepted "
                        f"({count_lines(current_code)} lines)"
                    ),
                    code=current_code,
                )
            else:
                yield PipelineEvent(
                    message=(
                        f"cloud: polish output failed ast.parse "
                        f"(line {polish_err.lineno}: {polish_err.msg}) — "
                        "discarding, keeping the local draft"
                    ),
                    code=current_code,
                    level="warning",
                )

        # ---- Verify + self-correction ------------------------------------ #
        yield PipelineEvent(
            message=(
                f"sandbox: verifying final output "
                f"({count_lines(current_code)} lines, ruff + mypy) ..."
            ),
            code=current_code,
        )
        report = await verify(current_code)

        correction = 0
        max_attempts = self._max_attempts
        while not report.passed and correction < max_attempts:
            correction += 1
            yield PipelineEvent(
                message=(
                    f"sandbox: FAIL — correction {correction}/{max_attempts}\n"
                    f"{report.diagnostics}"
                ),
                code=current_code,
                level="warning",
            )
            final_payload = self._orchestrator.build_correction(
                final_payload,
                current_code,
                report.diagnostics,
                budget=self._budget,
            )
            yield PipelineEvent(
                message="orchestrator: correction payload built — retrying",
                code=current_code,
            )
            gen = _GenResult(payload=final_payload)
            async for event in self._generate(
                gen,
                label=f"correction {correction}",
                code_so_far=current_code,
            ):
                yield event
                if gen.fatal:
                    return

            if gen.code is None:
                yield PipelineEvent(
                    message=(
                        f"correction {correction}: no code block emitted — stopping."
                    ),
                    code=current_code,
                    level="error",
                )
                break

            current_code = gen.code
            final_payload = gen.payload
            yield PipelineEvent(
                message=f"sandbox: re-verifying correction {correction} ...",
                code=current_code,
            )
            report = await verify(current_code)

        if report.passed:
            yield PipelineEvent(
                message="sandbox: PASS — output is verified ✓",
                code=current_code,
                finished=True,
                verified=True,
                exit_code=0,
            )
        else:
            yield PipelineEvent(
                message=(
                    f"EXHAUSTED {max_attempts} corrections — "
                    "showing last UNVERIFIED output ✗"
                ),
                code=current_code,
                level="error",
                finished=True,
                verified=False,
                exit_code=2,
            )
