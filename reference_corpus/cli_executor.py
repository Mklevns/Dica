"""Secure, non-blocking execution of system administration CLI tools.

This module is a reference implementation of safe async subprocess use:

* ``asyncio.create_subprocess_exec`` only — arguments are passed as a
  vector, so no shell ever interprets them (``shell=True`` is banned).
* Both pipes are drained concurrently with ``asyncio.gather`` to avoid
  the classic pipe-buffer deadlock, and the whole exchange is bounded
  by ``asyncio.wait_for``.
* Timed-out processes are killed and reaped; they are never leaked.
* Non-zero exit codes raise a structured ``CommandExecutionError``
  carrying argv, the exit code, and both captured streams.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "CommandExecutionError",
    "CommandResult",
    "CommandTimeoutError",
    "archive_directory",
    "run_command",
]

DEFAULT_TIMEOUT_SECONDS: float = 60.0
_READ_CHUNK_BYTES: int = 65_536


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of a successfully completed subprocess.

    Attributes:
        argv: The exact argument vector that was executed.
        returncode: Process exit code (always ``0`` on this type).
        stdout: Captured standard output, decoded as UTF-8 with
            undecodable bytes replaced.
        stderr: Captured standard error, decoded the same way.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandExecutionError(RuntimeError):
    """Raised when a subprocess exits with a non-zero return code.

    Attributes:
        argv: The argument vector that was executed.
        returncode: The non-zero exit code, or ``None`` if the process
            never produced one (e.g. it was killed on timeout).
        stdout: Whatever standard output was captured before failure.
        stderr: Whatever standard error was captured before failure.
    """

    def __init__(
        self,
        argv: tuple[str, ...],
        returncode: int | None,
        stdout: str,
        stderr: str,
        message: str | None = None,
    ) -> None:
        """Initializes the error with full diagnostic context.

        Args:
            argv: The argument vector that was executed.
            returncode: Exit code, or ``None`` if unavailable.
            stdout: Captured standard output.
            stderr: Captured standard error.
            message: Optional override for the exception message.
        """
        self.argv = argv
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        detail = message or (
            f"command {argv!r} failed with exit code {returncode}; "
            f"stderr: {stderr.strip()!r}"
        )
        super().__init__(detail)


class CommandTimeoutError(CommandExecutionError):
    """Raised when a subprocess exceeds its wall-clock timeout."""


async def _drain_stream(
    stream: asyncio.StreamReader | None,
    label: str,
    command_name: str,
) -> str:
    """Incrementally reads one pipe to exhaustion without blocking.

    Reading in fixed-size chunks (rather than lines) is immune to
    ``LimitOverrunError`` on pathologically long lines and keeps the
    event loop responsive while the child produces output.

    Args:
        stream: The pipe to drain, or ``None`` if it was not captured.
        label: Human-readable stream name (``"stdout"``/``"stderr"``)
            used for debug logging.
        command_name: Executable name, used for debug logging.

    Returns:
        The full decoded contents of the stream.
    """
    if stream is None:
        return ""
    chunks: list[bytes] = []
    while True:
        chunk = await stream.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        chunks.append(chunk)
        logger.debug(
            "drained %d bytes from %s of %s", len(chunk), label, command_name
        )
    return b"".join(chunks).decode("utf-8", errors="replace")


async def run_command(
    *argv: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cwd: str | None = None,
) -> CommandResult:
    """Executes a CLI tool safely without blocking the event loop.

    The command is executed directly via ``create_subprocess_exec``;
    arguments are never joined into a shell string, so shell metachar
    injection is structurally impossible.

    Args:
        *argv: The executable followed by its arguments, e.g.
            ``run_command("tar", "--create", "--file", dest)``.
        timeout: Wall-clock budget in seconds for the entire exchange
            (spawn, stream drain, and exit).
        cwd: Optional working directory for the child process.

    Returns:
        A ``CommandResult`` with the captured streams on success.

    Raises:
        ValueError: If ``argv`` is empty or ``timeout`` is not positive.
        CommandTimeoutError: If the process exceeds ``timeout``; the
            process is killed and reaped before this is raised.
        CommandExecutionError: If the process exits with a non-zero
            return code, or cannot be spawned at all.
    """
    if not argv:
        raise ValueError("argv must contain at least the executable name")
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    argv_t: tuple[str, ...] = tuple(argv)
    logger.debug("spawning %r (timeout=%.1fs, cwd=%r)", argv_t, timeout, cwd)

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv_t,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except (FileNotFoundError, PermissionError) as exc:
        raise CommandExecutionError(
            argv_t,
            None,
            "",
            "",
            message=f"failed to spawn {argv_t[0]!r}: {exc}",
        ) from exc

    async def _exchange() -> tuple[str, str]:
        """Drains both pipes concurrently, then reaps the child."""
        out, err = await asyncio.gather(
            _drain_stream(proc.stdout, "stdout", argv_t[0]),
            _drain_stream(proc.stderr, "stderr", argv_t[0]),
        )
        await proc.wait()
        return out, err

    try:
        stdout, stderr = await asyncio.wait_for(_exchange(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise CommandTimeoutError(
            argv_t,
            None,
            "",
            "",
            message=f"command {argv_t!r} timed out after {timeout:.1f}s",
        ) from None

    returncode = proc.returncode
    if returncode is None:
        raise CommandExecutionError(
            argv_t,
            None,
            stdout,
            stderr,
            message=f"command {argv_t!r} ended in an unknown process state",
        )

    if returncode != 0:
        raise CommandExecutionError(argv_t, returncode, stdout, stderr)

    logger.debug("command %r completed cleanly", argv_t)
    return CommandResult(
        argv=argv_t,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


async def archive_directory(
    source_dir: str,
    dest_tarball: str,
    timeout: float = 300.0,
) -> CommandResult:
    """Archives a directory (e.g. log rotation output) into a tarball.

    A typical maintenance task: paths arrive as discrete arguments, so
    a filename like ``"; rm -rf /"`` is just a filename, never code.

    Args:
        source_dir: Directory whose contents will be archived.
        dest_tarball: Path of the ``.tar.gz`` file to create.
        timeout: Wall-clock budget in seconds.

    Returns:
        The ``CommandResult`` from the underlying ``tar`` invocation.

    Raises:
        CommandTimeoutError: If archiving exceeds ``timeout``.
        CommandExecutionError: If ``tar`` exits non-zero.
    """
    return await run_command(
        "tar",
        "--create",
        "--gzip",
        f"--file={dest_tarball}",
        "--directory",
        source_dir,
        ".",
        timeout=timeout,
    )


if __name__ == "__main__":

    async def _demo() -> None:
        """Runs one succeeding and one failing command as a smoke test."""
        logging.basicConfig(level=logging.DEBUG)

        ok = await run_command("uname", "-a", timeout=10.0)
        logger.info("uname says: %s", ok.stdout.strip())

        try:
            await run_command("ls", "/definitely/not/a/real/path", timeout=10.0)
        except CommandExecutionError as exc:
            logger.info(
                "failure surfaced correctly (code=%s): %s",
                exc.returncode,
                exc.stderr.strip(),
            )

    asyncio.run(_demo())
