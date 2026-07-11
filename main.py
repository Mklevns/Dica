"""DICA CLI — thin adapter over :class:`dica.pipeline.PipelineEngine`.

Usage::

    python main.py "Build an async CRUD router for a User resource"
    python main.py --target messy.py "Refactor to Pydantic v2 + async I/O"
    python main.py --dry-run "..."
    python main.py --model phi4 --max-attempts 3 "..."

Exit codes: 0 = verified, 1 = infra failure, 2 = exhausted corrections.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dica.config import get_config
from dica.pipeline import (
    DEFAULT_CORPUS,
    LLMConfig,
    LocalLLMClient,
    PipelineEngine,
    PipelineEvent,
    cloud_polish,
    syntax_regression,
    unwrap_extraction,
)
from dica.sandbox import cleanup_orphaned_containers

# Re-exports for callers that historically imported from ``main``.
__all__ = [
    "LLMConfig",
    "LocalLLMClient",
    "cloud_polish",
    "main",
    "parse_args",
    "run_pipeline",
    "syntax_regression",
    "unwrap_extraction",
]

logger = logging.getLogger("dica")

# Back-compat alias used by older imports (``from main import _syntax_regression``).
_syntax_regression = syntax_regression


def _read_target(target: Path | None) -> str | None:
    """Load the optional messy script to refactor."""
    if target is None:
        return None
    return target.read_text(encoding="utf-8")


def _emit_event(event: PipelineEvent) -> None:
    """Map a pipeline event onto the standard logging hierarchy."""
    if event.level == "error":
        logger.error("%s", event.message)
    elif event.level == "warning":
        logger.warning("%s", event.message)
    else:
        logger.info("%s", event.message)


async def run_pipeline(
    query: str,
    *,
    corpus: Path,
    llm: LLMConfig,
    target: Path | None = None,
    max_attempts: int = 3,
    dry_run: bool = False,
) -> int:
    """Execute the shared pipeline; return a process exit code."""
    try:
        target_code = _read_target(target)
    except (OSError, UnicodeDecodeError) as exc:
        logger.error("Cannot read --target %s: %s", target, exc)
        return 1

    engine = PipelineEngine(
        client=None if dry_run else LocalLLMClient(llm),
        corpus=corpus,
        max_attempts=max_attempts,
        num_ctx=llm.num_ctx,
    )

    exit_code = 1
    final_code: str | None = None
    verified: bool | None = None

    async for event in engine.run(
        query, target_code=target_code, dry_run=dry_run
    ):
        _emit_event(event)

        if event.dry_run_text is not None:
            banner = "=" * 72
            print(f"\n{banner}\nDRY RUN — Pass 0 draft payload:\n{banner}")
            print(event.dry_run_text)
            print(f"{banner}\nPlanned refinement schedule:")
            for line in event.schedule_lines:
                print(line)

        if event.code is not None:
            final_code = event.code

        if event.finished:
            exit_code = event.exit_code if event.exit_code is not None else 1
            verified = event.verified
            if event.fatal and event.exit_code is not None:
                return event.exit_code

    if dry_run:
        return exit_code

    banner = "=" * 72
    if verified is True and final_code is not None:
        print(f"\n{banner}\n✓ VERIFIED OUTPUT (ruff + mypy clean)\n{banner}")
        print(final_code)
        return 0

    if final_code is not None and exit_code == 2:
        logger.error(
            "Exhausted the correction budget without passing the quality gate."
        )
        print(f"\n{banner}\n✗ LAST (UNVERIFIED) OUTPUT\n{banner}")
        print(final_code)
        return 2

    return exit_code


def parse_args(argv: list[str]) -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Dynamic In-Context Alignment pipeline (4-pass refinement)"
    )
    parser.add_argument(
        "query", help="Target task, e.g. 'Build an async CRUD router'"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Messy script to refactor; omit for pure generation from the task",
    )
    parser.add_argument(
        "--model",
        default=cfg.ollama.model,
        help=f"Ollama model name (default from config: {cfg.ollama.model})",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=cfg.engine.max_retries,
        help=(
            "Self-correction attempts after sandbox failure "
            f"(default from config: {cfg.engine.max_retries})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the Pass 0 payload and the refinement "
        "schedule, then exit without calling the model",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    cfg = get_config()
    cleanup_orphaned_containers(cfg.sandbox)
    exit_code = asyncio.run(
        run_pipeline(
            args.query,
            corpus=args.corpus,
            llm=LLMConfig.from_ollama(cfg.ollama, model=args.model),
            target=args.target,
            max_attempts=args.max_attempts,
            dry_run=args.dry_run,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
