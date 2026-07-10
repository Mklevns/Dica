"""DICA sandbox runner — executes INSIDE the container.

Copies the read-only mounted candidate into container-local tmpfs (ruff
``--fix`` mutates its target, which would fail on the ``ro`` bind mount),
runs ruff and mypy concurrently, and emits a single JSON line on stdout:

    [{"tool": "ruff", "passed": true, "output": "..."}, ...]

The host parses the *last* JSON-decodable line, so incidental stderr noise
from the checkers can never corrupt the protocol.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

RUFF_ARGS: tuple[str, ...] = (
    "check", "--select", "E,F,W,I", "--no-cache", "--fix",
    "--extend-ignore", "E501",
)
MYPY_ARGS: tuple[str, ...] = (
    "--strict",
    "--no-color-output",
    "--no-error-summary",
    "--ignore-missing-imports",
    "--no-warn-return-any",
    "--cache-dir", "/tmp/mypy_cache",
)


async def run_checker(tool: str, args: tuple[str, ...], target: Path) -> dict[str, object]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", tool, *args, str(target),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return {
        "tool": tool,
        "passed": proc.returncode == 0,
        "output": (stdout.decode() + stderr.decode()).strip(),
    }


async def main() -> None:
    mounted = Path(sys.argv[1])
    target = Path("/tmp/candidate.py")
    shutil.copyfile(mounted, target)
    results = await asyncio.gather(
        run_checker("ruff", RUFF_ARGS, target),
        run_checker("mypy", MYPY_ARGS, target),
    )
    print(json.dumps(list(results)))


if __name__ == "__main__":
    asyncio.run(main())
