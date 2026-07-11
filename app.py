"""DICA Refactoring Copilot — thin Gradio adapter over PipelineEngine.

All multi-pass lifecycle logic lives in :mod:`dica.pipeline`. This module
only validates uploads, streams :class:`~dica.pipeline.PipelineEvent`
snapshots into Gradio panels, and launches the local UI.

Run with::

    python app.py            # serves on http://127.0.0.1:7860
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr

from dica.config import get_config
from dica.pipeline import (
    DEFAULT_CORPUS,
    LLMConfig,
    LocalLLMClient,
    PipelineEngine,
)
from dica.sandbox import cleanup_orphaned_containers

logger = logging.getLogger("dica.app")


class _LogBuffer:
    """Accumulates timestamped log lines for the UI footer panel."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def add(self, message: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._lines.append(f"[{stamp}] {message}")

    @property
    def text(self) -> str:
        return "\n".join(self._lines)


class RefactorEngine:
    """UI-facing wrapper: one shared :class:`PipelineEngine` per process."""

    def __init__(
        self,
        client: LocalLLMClient,
        corpus: Path = DEFAULT_CORPUS,
    ) -> None:
        self._engine = PipelineEngine(client=client, corpus=corpus)

    async def run(
        self, target_code: str, instructions: str
    ) -> AsyncIterator[tuple[str, str]]:
        """Stream ``(code_panel, log_panel)`` updates for Gradio."""
        log = _LogBuffer()
        code_panel = "# awaiting model output ..."

        async for event in self._engine.run(
            instructions, target_code=target_code
        ):
            log.add(event.message)
            if event.code is not None:
                code_panel = event.code
            yield code_panel, log.text
            if event.fatal or event.finished:
                return


def _read_upload(file_path: str | None) -> str | None:
    """Read the uploaded script's contents, or ``None`` if unusable."""
    if file_path is None:
        return None
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.exception("Failed to read upload %s", file_path)
        return None


def build_app(engine: RefactorEngine) -> gr.Blocks:
    """Construct the Gradio Blocks UI around a ready engine."""

    async def on_refactor(
        file_path: str | None, instructions: str
    ) -> AsyncIterator[tuple[str, str]]:
        target = _read_upload(file_path)
        if target is None:
            yield (
                "# no output",
                "ERROR: upload a UTF-8 .py file before refactoring.",
            )
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
            "Local-first via Ollama (model from config.toml), with an "
            "optional cloud polish pass (fail-soft stub)."
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
    cfg = get_config()
    cleanup_orphaned_containers(cfg.sandbox)
    llm = LLMConfig.from_ollama(cfg.ollama)
    engine = RefactorEngine(client=LocalLLMClient(llm))
    app = build_app(engine)
    app.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
