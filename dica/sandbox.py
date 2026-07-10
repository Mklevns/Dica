"""Module 4: Verification Sandbox (containerized).

Automated quality gate for LLM output: run ``ruff check`` + ``mypy`` against
a candidate file, by default inside an ephemeral, network-less, resource-
capped Docker container.

Backends
--------
* ``docker`` — the hardened path. Per verification: one throwaway container
  from the pinned ``dica-sandbox`` image with ``network_disabled=True``,
  read-only rootfs, tmpfs ``/tmp``, memory / pid / CPU limits,
  ``no-new-privileges``, non-root user, candidate bind-mounted read-only.
  Both checkers run concurrently *inside* the container (one container
  startup, not two) via the baked-in runner script, which emits JSON.
* ``local`` — the original concurrent-subprocess path, retained for
  environments without a Docker daemon. Adequate while the gate is pure
  static analysis; the container backend is what makes it safe to ever
  *execute* generated code.
* ``auto`` — probe the daemon once, prefer ``docker``, fall back to
  ``local`` with a logged warning.

Event-loop notes
----------------
The Docker SDK is synchronous. Calling it inline would block the loop and
stall the RefactorEngine's async generator mid-stream, so the entire
container lifecycle runs in :func:`asyncio.to_thread`; ``await verify(...)``
keeps its original contract and the loop stays free while the container
grinds. The local backend keeps the original ``create_subprocess_exec`` +
``gather`` fan-out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from dica.config import SandboxConfig, get_config

# Re-export: extraction moved to its own module; keep the old import path
# alive. NOTE: the return type changed from `str | None` to
# `ExtractionResult` — see the integration notes.
from dica.extraction import ExtractionResult, extract_python_code

__all__ = [
    "CheckResult",
    "ExtractionResult",
    "VerificationReport",
    "cleanup_orphaned_containers",
    "cleanup_orphaned_containers_async",
    "extract_python_code",
    "verify",
]

logger = logging.getLogger(__name__)

# Container identity: every sandbox container carries this label AND a
# name with this prefix. Labels are the primary sweep key (immune to
# rename); the name prefix makes orphans obvious in `docker ps -a` and
# gives the sweep a second, human-auditable filter.
_SANDBOX_LABEL = "dica.sandbox"
_SANDBOX_NAME_PREFIX = "dica_sandbox_"

# Local-backend checker arguments (the container runner pins its own copy).
_RUFF_ARGS: tuple[str, ...] = (
    "check", "--select", "E,F,W,I", "--no-cache", "--fix",
    "--extend-ignore", "E501",
)
_MYPY_ARGS: tuple[str, ...] = (
    "--strict",
    "--no-color-output",
    "--no-error-summary",
    "--ignore-missing-imports",
    # NOTE: `--allow-any-return` / `--allow-any-expr` are NOT real mypy CLI
    # flags — `--no-warn-return-any` is the correct relaxation.
    "--no-warn-return-any",
)


class CheckResult(BaseModel):
    """Outcome of a single external checker run."""

    model_config = ConfigDict(frozen=True)

    tool: str
    passed: bool
    output: str


class VerificationReport(BaseModel):
    """Aggregate quality-gate verdict for one generation attempt."""

    model_config = ConfigDict(frozen=True)

    code: str
    checks: tuple[CheckResult, ...]
    backend: str = "local"

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def diagnostics(self) -> str:
        """Verbatim failure output, ready to embed in a correction prompt.

        Feed this through ``ContextBudget.truncate_middle`` (or let
        ``ContextBudget.assemble`` do it) before prompting — mypy under
        ``--strict`` can emit far more than the model's window.
        """
        return "\n\n".join(
            f"--- {c.tool} ---\n{c.output.strip() or '(no output)'}"
            for c in self.checks
            if not c.passed
        )


# --------------------------------------------------------------------- #
# Backend resolution
# --------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _docker_client() -> Any | None:
    """Probe the Docker daemon once per process. ``None`` == unavailable."""
    try:
        import docker  # noqa: PLC0415 — optional dependency, import at need
    except ImportError:
        logger.info("docker SDK not installed; sandbox will use local backend.")
        return None
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # docker raises a zoo of exception types here
        logger.warning("Docker daemon unreachable (%s); using local backend.", exc)
        return None
    return client


def _resolve_backend(config: SandboxConfig) -> str:
    if config.backend == "local":
        return "local"
    client = _docker_client()
    if client is not None:
        return "docker"
    if config.backend == "docker":
        raise RuntimeError(
            "Sandbox backend is 'docker' but the Docker daemon is "
            "unavailable. Start Docker or set sandbox.backend = 'local'."
        )
    return "local"


# --------------------------------------------------------------------- #
# Orphaned-container cleanup (startup hygiene sweep)
# --------------------------------------------------------------------- #
def cleanup_orphaned_containers(config: SandboxConfig | None = None) -> int:
    """Force-remove sandbox containers orphaned by a previous hard crash.

    ``finally: container.remove(force=True)`` covers every *soft* failure,
    but a ``SIGKILL`` / OOM-kill / power loss of the parent DICA process
    bypasses Python entirely and strands the container on the host. This
    sweep runs at application startup, before the first generation pass,
    and reclaims anything matching our identity markers:

    * label ``dica.sandbox`` (every container this module creates),
    * name prefix ``dica_sandbox_``,
    * ancestor image ``config.image`` — the belt-and-suspenders filter
      that also catches orphans created by *older* DICA builds that
      predate the label/name tagging.

    Results are deduplicated by container id before removal, so a
    container matching all three filters is removed exactly once.

    Fail-soft by contract: an unreachable daemon, a permission error, or a
    race with another process removing the same container are all logged
    and swallowed. Startup proceeds regardless — a stale container is a
    hygiene problem, not a correctness problem.

    Returns:
        The number of containers actually removed.
    """
    cfg = config if config is not None else get_config().sandbox
    client = _docker_client()
    if client is None:
        logger.info(
            "Docker unavailable; skipping orphaned-container sweep."
        )
        return 0

    candidates: dict[str, Any] = {}
    filter_sets = (
        {"label": _SANDBOX_LABEL},
        {"name": _SANDBOX_NAME_PREFIX},
        {"ancestor": cfg.image},
    )
    for filters in filter_sets:
        try:
            for container in client.containers.list(all=True, filters=filters):
                candidates[container.id] = container
        except Exception as exc:  # daemon hiccup mid-sweep: degrade, don't die
            logger.warning(
                "Orphan sweep query %r failed (%s); continuing with "
                "whatever was already found.",
                filters,
                exc,
            )

    removed = 0
    for container in candidates.values():
        try:
            container.remove(force=True)
            removed += 1
            logger.info(
                "Removed orphaned sandbox container %s (%s).",
                container.name,
                container.short_id,
            )
        except Exception as exc:  # already gone / racing remover / API error
            logger.warning(
                "Could not remove orphaned container %s: %s",
                container.short_id,
                exc,
            )

    if removed:
        logger.info(
            "Startup hygiene: removed %d orphaned sandbox container(s).",
            removed,
        )
    else:
        logger.debug("Startup hygiene: no orphaned sandbox containers found.")
    return removed


async def cleanup_orphaned_containers_async(
    config: SandboxConfig | None = None,
) -> int:
    """Loop-safe wrapper: the sync Docker SDK sweep runs on a worker thread.

    Use this variant when startup wiring lives *inside* the coroutine tree
    (e.g. at the top of ``run_pipeline``); use the sync function directly
    when wiring runs before ``asyncio.run`` (e.g. ``main()``), where no
    loop exists to block.
    """
    return await asyncio.to_thread(cleanup_orphaned_containers, config)


# --------------------------------------------------------------------- #
# Docker backend (sync core, executed via asyncio.to_thread)
# --------------------------------------------------------------------- #
def _docker_verify_sync(code: str, config: SandboxConfig) -> list[CheckResult]:
    import docker.errors  # noqa: PLC0415

    client = _docker_client()
    if client is None:  # pragma: no cover — _resolve_backend guards this
        raise RuntimeError("Docker client vanished after probe.")

    with tempfile.TemporaryDirectory(prefix="dica_sandbox_") as tmpdir:
        target = Path(tmpdir) / "candidate.py"
        target.write_text(code.rstrip("\n") + "\n", encoding="utf-8")

        container = client.containers.run(
            config.image,
            command=["python", "/opt/dica/runner.py", "/work/candidate.py"],
            name=f"{_SANDBOX_NAME_PREFIX}{uuid.uuid4().hex[:12]}",
            labels={_SANDBOX_LABEL: "1"},
            volumes={tmpdir: {"bind": "/work", "mode": "ro"}},
            network_disabled=True,
            read_only=True,
            tmpfs={"/tmp": "size=128m,mode=1777"},
            mem_limit=config.mem_limit,
            pids_limit=config.pids_limit,
            cpu_period=100_000,
            cpu_quota=config.cpu_quota_percent * 1_000,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            detach=True,
        )
        try:
            try:
                exit_info = container.wait(timeout=config.timeout_seconds)
            except Exception as exc:  # requests.Timeout / ConnectionError
                container.kill()
                return [
                    CheckResult(
                        tool="sandbox",
                        passed=False,
                        output=(
                            f"Verification container exceeded "
                            f"{config.timeout_seconds:.0f}s and was killed "
                            f"({exc})."
                        ),
                    )
                ]
            logs = container.logs(stdout=True, stderr=True).decode(
                errors="replace"
            )
        finally:
            try:
                container.remove(force=True)
            except docker.errors.APIError:  # pragma: no cover
                logger.warning("Failed to remove sandbox container %s.", container.id)

    if exit_info.get("StatusCode", 1) != 0:
        return [
            CheckResult(
                tool="sandbox",
                passed=False,
                output=f"Runner exited non-zero inside the container:\n{logs.strip()}",
            )
        ]
    return _parse_runner_output(logs)


def _parse_runner_output(logs: str) -> list[CheckResult]:
    """Extract the runner's JSON verdict — the last JSON-decodable line."""
    for line in reversed(logs.strip().splitlines()):
        line = line.strip()
        if not (line.startswith("[") and line.endswith("]")):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        return [CheckResult.model_validate(item) for item in raw]
    return [
        CheckResult(
            tool="sandbox",
            passed=False,
            output=f"Runner produced no parseable verdict. Raw logs:\n{logs.strip()}",
        )
    ]


# --------------------------------------------------------------------- #
# Local backend (original behaviour, retained as fallback)
# --------------------------------------------------------------------- #
async def _run_local_checker(
    tool: str, args: tuple[str, ...], target: Path
) -> CheckResult:
    """Spawn one checker as a non-blocking subprocess and capture its output.

    ``sys.executable -m <tool>`` guarantees we invoke the checker installed
    in the *current* interpreter's environment, not whatever is on PATH.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", tool, *args, str(target),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = (stdout.decode() + stderr.decode()).strip()
    passed = proc.returncode == 0
    logger.info("%s: %s", tool, "PASS" if passed else "FAIL")
    return CheckResult(tool=tool, passed=passed, output=output)


async def _local_verify(code: str, timeout: float) -> list[CheckResult]:
    with tempfile.TemporaryDirectory(prefix="dica_sandbox_") as tmpdir:
        target = Path(tmpdir) / "candidate.py"
        # Normalize the trailing newline: fence extraction strips it, and a
        # missing EOF newline (ruff W292) is an artifact of extraction, not
        # a defect in the model's code.
        target.write_text(code.rstrip("\n") + "\n", encoding="utf-8")
        checks = await asyncio.wait_for(
            asyncio.gather(
                _run_local_checker("ruff", _RUFF_ARGS, target),
                _run_local_checker("mypy", _MYPY_ARGS, target),
            ),
            timeout=timeout,
        )
    return list(checks)


# --------------------------------------------------------------------- #
# Public gate
# --------------------------------------------------------------------- #
async def verify(
    code: str,
    *,
    config: SandboxConfig | None = None,
    timeout: float | None = None,
) -> VerificationReport:
    """Run the full quality gate against ``code``.

    Signature-compatible with the previous implementation: existing
    ``await verify(code)`` call sites keep working; ``timeout`` overrides
    the configured value when given.
    """
    cfg = config if config is not None else get_config().sandbox
    effective_timeout = timeout if timeout is not None else cfg.timeout_seconds
    backend = _resolve_backend(cfg)

    if backend == "docker":
        checks = await asyncio.to_thread(_docker_verify_sync, code, cfg)
    else:
        checks = await _local_verify(code, effective_timeout)

    report = VerificationReport(code=code, checks=tuple(checks), backend=backend)
    logger.info(
        "Verification [%s backend]: %s",
        backend,
        "PASS" if report.passed else "FAIL",
    )
    return report
