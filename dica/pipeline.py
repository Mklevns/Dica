"""Shared search-first DICA pipeline engine.

Single source of truth for the lifecycle used by the CLI (``main.py``) and
the Gradio UI (``app.py``):

    ingest vault
        -> agentic catalog selection (local model, temperature 0)
        -> resolve pattern name (dispatcher top-1 fallback on hallucination)
        -> anchored generation (single gold reference)
        -> cloud polish (optional; fail-soft stub when unset)
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
from dica.dispatcher import IntentDispatcher
from dica.embeddings import OllamaEmbedder, SemanticIndex
from dica.orchestrator import (
    PromptOrchestrator,
    PromptPayload,
    build_selector_payload,
    parse_selection_name,
)
from dica.sandbox import ExtractionResult, extract_python_code, verify
from dica.vault import CodeChunk, CodeVault, get_vault_catalog

logger = logging.getLogger(__name__)

DEFAULT_CORPUS: Final[Path] = Path(__file__).resolve().parent.parent / "reference_corpus"


class SupportsComplete(Protocol):
    """Structural type for any async LLM client the engine can drive."""

    async def complete(
        self, prompt: str, *, temperature: float | None = None
    ) -> str:
        """Return the model's raw text response for ``prompt``.

        Optional ``temperature`` overrides the client's default sampling
        temperature for this call (used for deterministic selection).
        """
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

    async def complete(
        self, prompt: str, *, temperature: float | None = None
    ) -> str:
        """One non-streaming chat round-trip. Raises on transport failure.

        When ``temperature`` is set it overrides :attr:`LLMConfig.temperature`
        for this request only (agentic selection uses ``0.0``).
        """
        temp = self._config.temperature if temperature is None else temperature
        request = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temp,
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
    """Owns vault/dispatcher/orchestrator and runs the search-first lifecycle.

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
        # Used only when agentic selection fails to resolve a valid pattern name.
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
        self._format_retries = cfg.engine.format_retries
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
            "(format_retries=%d, max_retries=%d, prompt_budget=%d tokens, "
            "semantic_fallback=%s)",
            self._chunk_count,
            corpus,
            self._format_retries,
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
        max_format_attempts = self._format_retries + 1
        for attempt in range(1, max_format_attempts + 1):
            yield PipelineEvent(
                message=(
                    f"{label}: attempt {attempt}/{max_format_attempts} — "
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

    async def _resolve_reference(
        self,
        task: str,
        *,
        target_code: str | None,
        dry_run: bool,
    ) -> AsyncIterator[PipelineEvent | CodeChunk]:
        """Agentic catalog selection with hybrid-dispatch fallback.

        Yields progress :class:`PipelineEvent`s, then yields the resolved
        :class:`CodeChunk` as the final item (or a fatal finished event and
        no chunk when resolution is impossible).
        """
        yield PipelineEvent(
            message="agent: building reference catalog for model selection..."
        )
        catalog_text = get_vault_catalog(self._vault)
        selector_prompt = build_selector_payload(
            target_code or "", task, catalog_text
        )
        known_names = [c.name for c in self._vault]

        selected_name: str | None = None
        raw_selection: str | None = None

        if not dry_run:
            if self._client is None:
                yield PipelineEvent(
                    message="No LLM client configured (cannot select pattern).",
                    level="error",
                    fatal=True,
                    finished=True,
                    exit_code=1,
                )
                return
            yield PipelineEvent(
                message="agent: querying local model for architectural blueprint..."
            )
            try:
                raw_selection = await self._client.complete(
                    selector_prompt, temperature=0.0
                )
                selected_name = parse_selection_name(raw_selection, known_names)
            except Exception as exc:
                logger.warning("Agentic search failed: %s", exc)
                yield PipelineEvent(
                    message=(
                        f"agent: selection query failed ({exc}) — "
                        "falling back to semantic dispatcher"
                    ),
                    level="warning",
                )

        target_chunk: CodeChunk | None = None
        if selected_name:
            target_chunk = self._vault.get_by_name(selected_name)

        if target_chunk is not None:
            yield PipelineEvent(
                message=(
                    f"agent: successfully locked onto blueprint "
                    f"'{target_chunk.name}'"
                )
            )
        else:
            if raw_selection is not None and selected_name is None:
                preview = raw_selection.strip().replace("\n", " ")[:80]
                yield PipelineEvent(
                    message=(
                        f"agent: selection {preview!r} invalid. "
                        "Falling back to semantic dispatcher."
                    ),
                    level="warning",
                )
            elif selected_name is not None:
                yield PipelineEvent(
                    message=(
                        f"agent: selection '{selected_name}' not in vault. "
                        "Falling back to semantic dispatcher."
                    ),
                    level="warning",
                )
            elif dry_run:
                yield PipelineEvent(
                    message=(
                        "agent: dry-run skips model selection — "
                        "resolving blueprint via dispatcher fallback"
                    )
                )

            hits = await self._dispatcher.dispatch(task, top_k=1)
            target_chunk = hits[0].chunk if hits else None
            if target_chunk is not None:
                yield PipelineEvent(
                    message=(
                        f"dispatcher: fallback blueprint "
                        f"'{target_chunk.name}' (score {hits[0].total:.2f})"
                    )
                )

        if target_chunk is None:
            yield PipelineEvent(
                message="Critical failure: No reference could be resolved.",
                level="error",
                fatal=True,
                finished=True,
                exit_code=1,
            )
            return

        yield target_chunk

    async def run(
        self,
        task: str,
        *,
        target_code: str | None = None,
        dry_run: bool = False,
    ) -> AsyncIterator[PipelineEvent]:
        """Execute the search-first lifecycle, yielding progress events.

        Args:
            task: User instruction (generation or refactor orders).
            target_code: Optional messy script contents for refactor mode.
            dry_run: If True, resolve a blueprint via dispatcher fallback,
                render the anchored generation payload, and finish without
                calling the model.
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

        # ---- 1–2. Agentic search + resolution / fallback ----------------- #
        target_chunk: CodeChunk | None = None
        async for item in self._resolve_reference(
            task, target_code=target_code, dry_run=dry_run
        ):
            if isinstance(item, CodeChunk):
                target_chunk = item
            else:
                yield item
                if item.fatal and item.finished:
                    return

        if target_chunk is None:
            # Fatal events already yielded by _resolve_reference.
            return

        # ---- Dry-run: show selector + anchored plan without generation --- #
        if dry_run:
            catalog_text = get_vault_catalog(self._vault)
            selector_prompt = build_selector_payload(
                target_code or "", task, catalog_text
            )
            payload = self._orchestrator.build_anchored_payload(
                task, target_code or "", target_chunk
            )
            assembled = payload.render_budgeted(self._budget)
            yield PipelineEvent(
                message=(
                    f"dry-run: scheduled to generate against {target_chunk.name}"
                ),
                dry_run_text=selector_prompt,
                schedule_lines=(
                    f"  Blueprint: {target_chunk.name} "
                    f"({target_chunk.file_path})",
                    f"  Anchored generation payload: "
                    f"~{assembled.report.used_tokens} tokens "
                    f"({len(assembled.text)} chars)",
                ),
                finished=True,
                exit_code=0,
            )
            return

        # ---- 3. Anchored generation -------------------------------------- #
        yield PipelineEvent(
            message=(
                f"generation: drafting code using {target_chunk.name} "
                "as primary constraint..."
            )
        )
        payload = self._orchestrator.build_anchored_payload(
            task, target_code or "", target_chunk
        )
        gen = _GenResult(payload=payload)
        async for event in self._generate(
            gen, label="anchored generation", code_so_far=None
        ):
            yield event
            if gen.fatal:
                return

        if gen.code is None:
            yield PipelineEvent(
                message="Generation failed to produce valid code.",
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
                    f"generation: output failed ast.parse "
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
            message=(
                f"generation: draft captured ({count_lines(current_code)} lines)"
            ),
            code=current_code,
        )

        # ---- 4. Cloud polish (optional; skipped when unset) -------------- #
        yield PipelineEvent(
            message="cloud: offering draft for cloud polish (optional) ...",
            code=current_code,
        )
        try:
            polished = await cloud_polish(current_code, task)
        except CloudPolishError as exc:
            logger.info("Cloud polish skipped (not configured): %s", exc)
            yield PipelineEvent(
                message=(
                    "cloud: polish not configured — skipping "
                    "(optional step; using local draft)"
                ),
                code=current_code,
            )
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

        # ---- 5. Verify + self-correction --------------------------------- #
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
