"""DICA Refactoring Copilot — asynchronous Gradio frontend.

Wires the existing pipeline (vault -> dispatcher -> orchestrator -> local
LLM -> sandbox) into a web UI for refactoring messy scripts against the
gold-standard corpus.

Architecture
------------
The UI layer is deliberately thin. All lifecycle logic lives in
:class:`RefactorEngine.run`, an **async generator** that yields a
``(code, log)`` snapshot after every stage. Gradio natively streams async
generators: each ``yield`` repaints the ``gr.Code`` output and the sandbox
log textbox, so the user watches dispatch, drafting, every refinement pass,
cloud polish, verification, and self-correction happen live instead of
staring at a spinner.

Multi-pass parity with ``main.py``
----------------------------------
The engine executes the exact schedule the CLI pipeline runs:

    dispatch(top_k=3)
        -> Pass 0: initial draft   (task + target script, ZERO references)
        -> Passes 1..N: iterative single-reference alignment
           (previous reference flushed; previous output becomes the target;
            fail-fast ``ast.parse`` gate with rollback on regression)
        -> cloud polish            (optional foundation-model pass; fail-soft)
        -> verify(final)           (ruff + mypy — the ONLY verification gate)
        -> (correct)*              (self-correction loop on the final output)

Because the engine is a generator, the code panel visibly *evolves* pass by
pass: the user sees the raw draft, then each alignment reshaping it, then
the verified (or last-unverified) result.

Extraction contract
-------------------
``dica.sandbox.extract_python_code`` returns an :class:`ExtractionResult`
Pydantic model, NOT a raw string. The engine unwraps the string payload via
:func:`main.unwrap_extraction` at the single extraction call site, so the
code panel, ``_count_lines``, and the sandbox only ever see plain ``str``.

The LLM client is injected behind a :class:`SupportsComplete` protocol so
the engine is unit-testable with a fake client — no Ollama required to
exercise the full multi-pass and correction machinery.

Run with::

    python app.py            # serves on http://127.0.0.1:7860
"""

from __future__ import annotations

import ast
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import gradio as gr

from dica.dispatcher import DispatchResult, IntentDispatcher
from dica.orchestrator import PromptOrchestrator, PromptPayload
from dica.sandbox import (
    ExtractionResult,
    cleanup_orphaned_containers,
    extract_python_code,
    verify,
)
from dica.vault import CodeVault
from main import LLMConfig, LocalLLMClient, cloud_polish, unwrap_extraction

logger = logging.getLogger("dica.app")

CORPUS_DIR = Path(__file__).parent / "reference_corpus"

REFINEMENT_PASSES = 3   # Passes 1..3, one reference chunk each (as in main.py)
DISPATCH_TOP_K = 3      # one dispatcher hit per refinement pass
FORMAT_RETRIES = 2      # extra tries per pass when no code fence is emitted
MAX_ATTEMPTS = 3        # sandbox-failure self-correction budget


class SupportsComplete(Protocol):
    """Structural type for any async LLM client the engine can drive.

    Pattern: depending on a protocol instead of the concrete Ollama client
    lets tests substitute a scripted fake and lets future backends
    (llama.cpp server, vLLM) drop in without touching the engine.
    """

    async def complete(self, prompt: str) -> str:
        """Return the model's raw text response for ``prompt``."""
        ...


@dataclass(slots=True)
class _GenerationOutcome:
    """Result channel for one :meth:`RefactorEngine._generate` invocation.

    Async generators can only *yield*, not *return* values, so the streaming
    generation helper communicates its verdict through this mutable holder
    while its yields keep repainting the UI.

    Attributes:
        payload: The payload most recently sent to the model (the anchor
            for any subsequent ``build_correction`` call). Always valid.
        code: The extracted code block (already unwrapped to a plain
            string), or ``None`` if every retry emitted no fenced block.
        fatal: ``True`` when the backend itself failed (transport error) —
            the engine must abort the whole run, not merely skip a pass.
    """

    payload: PromptPayload
    code: str | None = None
    fatal: bool = False


class RefactorEngine:
    """Drives the full multi-pass refactor lifecycle for one user request.

    The engine owns the long-lived pieces (vault, dispatcher, orchestrator,
    client); each call to :meth:`run` is an independent request.
    """

    def __init__(self, client: SupportsComplete, corpus: Path = CORPUS_DIR) -> None:
        """Ingest the corpus once and bind the pipeline components.

        Args:
            client: Any object satisfying :class:`SupportsComplete`.
            corpus: Directory of gold-standard ``.py`` files to index.
        """
        self._vault = CodeVault()
        self._chunk_count = self._vault.ingest_path(corpus)
        self._dispatcher = IntentDispatcher(self._vault)
        self._orchestrator = PromptOrchestrator()
        self._client = client
        logger.info("Engine ready: %d chunks indexed", self._chunk_count)

    async def _generate(
        self,
        outcome: _GenerationOutcome,
        *,
        label: str,
        log: _LogBuffer,
        code_panel: str,
    ) -> AsyncIterator[tuple[str, str]]:
        """One generation pass with in-pass format retries, streamed.

        Mirrors ``main._generate_code``: protocol violations (no ```python
        fence) are retried up to ``FORMAT_RETRIES`` extra times with a
        format-correction payload; transport failures mark the outcome as
        fatal. Every retry boundary yields a UI snapshot.

        The extraction result is a Pydantic :class:`ExtractionResult`; the
        raw string payload is unwrapped here via ``unwrap_extraction``, so
        ``outcome.code`` is always ``str | None`` — never the model object.

        Args:
            outcome: Mutable result holder, pre-seeded with the payload to
                send; updated in place with the verdict.
            label: Human-readable pass name for the log panel.
            log: The run's shared log buffer.
            code_panel: Current code-panel text to keep repainting.

        Yields:
            ``(code_panel, log_panel)`` tuples for live UI updates.
        """
        payload = outcome.payload
        for attempt in range(1, FORMAT_RETRIES + 2):
            log.add(
                f"{label}: attempt {attempt}/{FORMAT_RETRIES + 1} — "
                "calling local model ..."
            )
            yield code_panel, log.text

            try:
                response = await self._client.complete(payload.render())
            except Exception as exc:  # backend failure — surface, don't crash UI
                logger.exception("Inference failed")
                log.add(
                    f"ERROR: inference failed: {exc} (is `ollama serve` running?)"
                )
                outcome.payload = payload
                outcome.fatal = True
                yield code_panel, log.text
                return

            # Single extraction call site: unwrap the Pydantic model into a
            # plain string BEFORE it can reach the code panel or the line
            # counter. (Previously the raw ExtractionResult leaked into
            # `outcome.code`, crashing `_count_lines` with
            # AttributeError: 'ExtractionResult' object has no attribute 'count'.)
            result: ExtractionResult | None = extract_python_code(response)
            code = unwrap_extraction(result)
            if code is not None:
                outcome.payload = payload
                outcome.code = code
                return

            log.add(
                f"{label}: response contained no ```python block — "
                "issuing format correction"
            )
            payload = self._orchestrator.build_correction(
                payload,
                failed_code="(no code block emitted)",
                diagnostics="Response contained no ```python fenced block.",
            )
            yield code_panel, log.text

        outcome.payload = payload
        outcome.code = None

    async def run(
        self, target_code: str, instructions: str
    ) -> AsyncIterator[tuple[str, str]]:
        """Execute dispatch -> draft -> N alignment passes -> polish -> verify.

        Async-generator contract: yields ``(code_panel, log_panel)`` after
        every meaningful stage — including each intermediate refinement
        pass — so the UI streams the code evolving in real time. The final
        yield carries either verified code or the last unverified attempt
        with a failure banner in the log.

        Args:
            target_code: Raw contents of the uploaded messy script.
            instructions: The user's refactoring orders.

        Yields:
            Tuples of (current code-panel text, accumulated log text).
        """
        log = _LogBuffer()
        code_panel = "# awaiting model output ..."

        # ---- Stage 1: dispatch ------------------------------------------ #
        log.add(f"vault: {self._chunk_count} gold chunks indexed")
        hits: list[DispatchResult] = self._dispatcher.dispatch(
            instructions, top_k=DISPATCH_TOP_K
        )
        matched = ", ".join(
            f"{h.chunk.name} ({h.total:.2f})" for h in hits
        ) or "(none)"
        log.add(f"dispatcher: matched {matched}")
        schedule = hits[:REFINEMENT_PASSES]
        if len(schedule) < REFINEMENT_PASSES:
            log.add(
                f"dispatcher: only {len(schedule)} hit(s) available — "
                f"running {len(schedule)} refinement pass(es) "
                f"instead of {REFINEMENT_PASSES}"
            )
        yield code_panel, log.text

        # ---- Stage 2: Pass 0 — initial draft (ZERO references) ----------- #
        draft_payload: PromptPayload = self._orchestrator.build_draft_payload(
            instructions, target_code=target_code
        )
        log.add(
            f"orchestrator: pass 0 draft payload rendered "
            f"({len(draft_payload.render())} chars, 0 references)"
        )
        yield code_panel, log.text

        outcome = _GenerationOutcome(payload=draft_payload)
        async for snapshot in self._generate(
            outcome, label="pass 0", log=log, code_panel=code_panel
        ):
            yield snapshot
        if outcome.fatal:
            return
        if outcome.code is None:
            log.add("pass 0 never produced a code block — aborting.")
            yield code_panel, log.text
            return

        current_code = outcome.code
        final_payload = outcome.payload
        code_panel = current_code
        log.add(f"pass 0: draft captured ({_count_lines(current_code)} lines)")
        yield code_panel, log.text

        # ---- Stage 3: Passes 1..N — single-reference alignment ----------- #
        # Each pass FLUSHES the previous reference, injects exactly ONE new
        # hit, and feeds the previous pass's code back in as the target. A
        # fail-fast `ast.parse` gate guards every hand-off: a pass whose
        # output does not parse is a regression and is rolled back.
        for pass_no, hit in enumerate(schedule, start=1):
            payload = self._orchestrator.build_refinement_payload(
                previous_code=current_code,
                result=hit,
                original_task=instructions,
                pass_index=pass_no,
                total_passes=len(schedule),
            )
            log.add(
                f"pass {pass_no}/{len(schedule)}: aligning to "
                f"'{hit.chunk.name}' (score {hit.total:.2f})"
            )
            yield code_panel, log.text

            outcome = _GenerationOutcome(payload=payload)
            async for snapshot in self._generate(
                outcome, label=f"pass {pass_no}", log=log, code_panel=code_panel
            ):
                yield snapshot
            if outcome.fatal:
                return
            if outcome.code is None:
                log.add(
                    f"pass {pass_no}: no usable code after retries — "
                    f"carrying pass {pass_no - 1} output forward"
                )
                yield code_panel, log.text
                continue

            # ---- fail-fast intermediate verification (ast.parse gate) ---- #
            try:
                ast.parse(outcome.code)
            except SyntaxError as exc:
                log.add(
                    f"pass {pass_no}: syntax regression "
                    f"(line {exc.lineno}: {exc.msg}) — rolling back to "
                    f"pass {pass_no - 1} output"
                )
                yield code_panel, log.text
                continue

            current_code = outcome.code
            final_payload = outcome.payload
            code_panel = current_code
            log.add(
                f"pass {pass_no}: aligned output captured "
                f"({_count_lines(current_code)} lines)"
            )
            yield code_panel, log.text

        # ---- Stage 3.5: Cloud Polish (foundation-model pass; fail-soft) -- #
        # Runs AFTER the final local refinement pass and BEFORE the
        # verification gate. Strictly optional: any failure logs a warning
        # and the local draft proceeds to the sandbox unchanged. Polished
        # output must survive the same ast.parse tripwire the refinement
        # passes are held to before it may replace the local draft.
        log.add("cloud: offering draft for cloud polish ...")
        yield code_panel, log.text
        try:
            polished = await cloud_polish(current_code, instructions)
        except Exception as exc:
            logger.warning("Cloud polish unavailable: %s", exc)
            log.add(
                f"cloud: polish unavailable ({exc}) — "
                "falling back to the local draft"
            )
            yield code_panel, log.text
        else:
            try:
                ast.parse(polished)
            except SyntaxError as exc:
                log.add(
                    f"cloud: polish output failed ast.parse "
                    f"(line {exc.lineno}: {exc.msg}) — discarding, "
                    "keeping the local draft"
                )
                yield code_panel, log.text
            else:
                current_code = polished
                code_panel = current_code
                log.add(
                    f"cloud: polish accepted "
                    f"({_count_lines(current_code)} lines)"
                )
                yield code_panel, log.text

        # ---- Stage 4: verification gate (final output ONLY) -------------- #
        log.add(
            f"sandbox: verifying final output "
            f"({_count_lines(current_code)} lines, ruff + mypy) ..."
        )
        yield code_panel, log.text
        report = await verify(current_code)

        # ---- Stage 5: self-correction loop on the final output ----------- #
        correction = 0
        while not report.passed and correction < MAX_ATTEMPTS:
            correction += 1
            log.add(f"sandbox: FAIL — correction {correction}/{MAX_ATTEMPTS}")
            log.add(report.diagnostics)
            final_payload = self._orchestrator.build_correction(
                final_payload, current_code, report.diagnostics
            )
            log.add("orchestrator: correction payload built — retrying")
            yield code_panel, log.text

            outcome = _GenerationOutcome(payload=final_payload)
            async for snapshot in self._generate(
                outcome,
                label=f"correction {correction}",
                log=log,
                code_panel=code_panel,
            ):
                yield snapshot
            if outcome.fatal:
                return
            if outcome.code is None:
                log.add(
                    f"correction {correction}: no code block emitted — stopping."
                )
                yield code_panel, log.text
                break

            current_code = outcome.code
            final_payload = outcome.payload
            code_panel = current_code
            log.add(f"sandbox: re-verifying correction {correction} ...")
            yield code_panel, log.text
            report = await verify(current_code)

        if report.passed:
            log.add("sandbox: PASS — output is verified ✓")
        else:
            log.add(
                f"EXHAUSTED {MAX_ATTEMPTS} corrections — "
                "showing last UNVERIFIED output ✗"
            )
        yield code_panel, log.text


def _count_lines(code: str) -> int:
    """Number of lines in ``code`` (for log messages)."""
    return code.count("\n") + 1


class _LogBuffer:
    """Accumulates timestamped log lines for the UI footer panel."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def add(self, message: str) -> None:
        """Append one timestamped line.

        Args:
            message: Log text (may be multi-line, e.g. diagnostics).
        """
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._lines.append(f"[{stamp}] {message}")

    @property
    def text(self) -> str:
        """The full log as one newline-joined string."""
        return "\n".join(self._lines)


def _read_upload(file_path: str | None) -> str | None:
    """Read the uploaded script's contents, or ``None`` if unusable.

    Args:
        file_path: Temp-file path Gradio provides for the upload.

    Returns:
        File contents as text, or ``None`` on missing/undecodable input.
    """
    if file_path is None:
        return None
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.exception("Failed to read upload %s", file_path)
        return None


def build_app(engine: RefactorEngine) -> gr.Blocks:
    """Construct the Gradio Blocks UI around a ready engine.

    Args:
        engine: The pipeline engine handling refactor requests.

    Returns:
        The assembled (unlaunched) Blocks application.
    """

    async def on_refactor(
        file_path: str | None, instructions: str
    ) -> AsyncIterator[tuple[str, str]]:
        """Async click handler: validate inputs, then stream engine output.

        Args:
            file_path: Path of the uploaded script (``gr.File`` value).
            instructions: Refactoring orders from the textbox.

        Yields:
            ``(code_panel, log_panel)`` tuples for live UI updates.
        """
        target = _read_upload(file_path)
        if target is None:
            yield "# no output", "ERROR: upload a UTF-8 .py file before refactoring."
            return
        if not instructions.strip():
            yield "# no output", "ERROR: enter refactoring instructions."
            return
        async for code, log in engine.run(target, instructions):
            yield code, log

    app = gr.Blocks(title="DICA Refactoring Copilot")
    with app:
        gr.Markdown(
            "# DICA Refactoring Copilot\n"
            "Upload a messy script, state your orders, and watch the "
            "multi-pass pipeline draft, align, polish, and verify it against "
            "the gold-standard corpus (ruff + mypy gated). "
            "Local-first: Phi-4 via Ollama, with an optional cloud polish "
            "pass (fail-soft stub)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                script_file = gr.File(
                    label="Messy script (.py)",
                    file_types=[".py"],
                    type="filepath",
                )
                instructions_box = gr.Textbox(
                    label="Refactoring instructions",
                    placeholder=(
                        "e.g. Refactor this to use Pydantic models and async I/O"
                    ),
                    lines=4,
                )
                refactor_btn = gr.Button("Refactor", variant="primary")
            with gr.Column(scale=2):
                code_out = gr.Code(
                    label="Verified output",
                    language="python",
                    value="# output appears here",
                )
        sandbox_log = gr.Textbox(
            label="Sandbox verification log (ruff + mypy)",
            lines=14,
            interactive=False,
        )

        refactor_btn.click(
            fn=on_refactor,
            inputs=[script_file, instructions_box],
            outputs=[code_out, sandbox_log],
        )
    return app


def main() -> None:
    """Entrypoint: build the engine against Ollama and launch the UI."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s"
    )
    # Startup hygiene: sweep orphaned sandbox containers before the first
    # request can create new ones. Sync context — Gradio's loop doesn't
    # exist until launch() — so the sync sweep is the right variant here.
    cleanup_orphaned_containers()
    engine = RefactorEngine(client=LocalLLMClient(LLMConfig()))
    app = build_app(engine)
    app.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
