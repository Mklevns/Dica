"""Async retry with true exponential backoff and full jitter.

This module is a reference implementation of resilient async retries:

* A fully typed ``@retry`` decorator (``ParamSpec`` + ``TypeVar``) that
  preserves the wrapped coroutine's exact signature under
  ``mypy --strict``.
* True exponential backoff (``base * factor ** attempt``) capped at a
  ceiling, with *full jitter* (``uniform(0, ceiling)``) to decorrelate
  competing retriers and avoid thundering herds.
* A mandatory, explicit tuple of retryable exception types — a bare
  ``except Exception`` would silently swallow programming errors.
* A structured ``logging`` warning on every failed attempt, and a
  ``MaxRetriesExceededError`` that chains the final underlying failure.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

__all__ = [
    "MaxRetriesExceededError",
    "retry",
]

P = ParamSpec("P")
T = TypeVar("T")


class MaxRetriesExceededError(RuntimeError):
    """Raised when every retry attempt has been exhausted.

    Attributes:
        func_name: Qualified name of the function that kept failing.
        attempts: Total number of attempts that were made.
        last_exception: The exception raised by the final attempt; it
            is also chained as ``__cause__``.
    """

    def __init__(
        self,
        func_name: str,
        attempts: int,
        last_exception: BaseException,
    ) -> None:
        """Initializes the error with the final failure context.

        Args:
            func_name: Qualified name of the failing function.
            attempts: Total attempts made before giving up.
            last_exception: The exception from the final attempt.
        """
        self.func_name = func_name
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"{func_name!r} failed after {attempts} attempt(s); "
            f"last error: {last_exception!r}"
        )


def retry(
    exceptions: tuple[type[Exception], ...],
    *,
    max_attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Retries a coroutine on specific transient exceptions.

    The delay before attempt ``n`` (1-indexed) is drawn uniformly from
    ``[0, min(max_delay, base_delay * backoff_factor ** (n - 1))]`` —
    exponential growth with full jitter. Non-listed exceptions (and
    ``asyncio.CancelledError``, which is not an ``Exception``) always
    propagate immediately.

    Args:
        exceptions: The transient exception types to retry on, e.g.
            ``(ConnectionError, TimeoutError)``. Deliberately required
            so callers cannot lazily retry on everything.
        max_attempts: Total attempts including the first call.
        base_delay: Backoff ceiling in seconds before the first retry.
        max_delay: Absolute cap on the backoff ceiling in seconds.
        backoff_factor: Multiplier applied to the ceiling per attempt.

    Returns:
        A decorator that wraps an ``async`` callable, preserving its
        parameter and return types.

    Raises:
        ValueError: If any numeric parameter is out of range, or if
            ``exceptions`` is empty.
    """
    if not exceptions:
        raise ValueError("exceptions must name at least one exception type")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if base_delay <= 0.0:
        raise ValueError(f"base_delay must be > 0, got {base_delay}")
    if max_delay < base_delay:
        raise ValueError(
            f"max_delay ({max_delay}) must be >= base_delay ({base_delay})"
        )
    if backoff_factor < 1.0:
        raise ValueError(f"backoff_factor must be >= 1, got {backoff_factor}")

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        """Wraps ``func`` with the configured retry policy.

        Args:
            func: The coroutine function to protect.

        Returns:
            An async wrapper with an identical signature.
        """

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            """Invokes ``func``, sleeping with backoff between failures.

            Args:
                *args: Positional arguments forwarded to ``func``.
                **kwargs: Keyword arguments forwarded to ``func``.

            Returns:
                Whatever ``func`` returns on its first success.

            Raises:
                MaxRetriesExceededError: If all attempts fail with one
                    of the configured transient exceptions.
            """
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    ceiling = min(
                        max_delay,
                        base_delay * backoff_factor ** (attempt - 1),
                    )
                    delay = random.uniform(0.0, ceiling)
                    logger.warning(
                        "transient failure in %s "
                        "(attempt=%d/%d, retry_in=%.3fs, error_type=%s): %s",
                        func.__qualname__,
                        attempt,
                        max_attempts,
                        delay,
                        type(exc).__name__,
                        exc,
                        extra={
                            "retry_func": func.__qualname__,
                            "retry_attempt": attempt,
                            "retry_max_attempts": max_attempts,
                            "retry_delay_seconds": round(delay, 3),
                            "retry_error_type": type(exc).__name__,
                        },
                    )
                    await asyncio.sleep(delay)
            if last_exc is None:
                raise AssertionError(
                    "retry loop exited without a result or an exception"
                )
            raise MaxRetriesExceededError(
                func.__qualname__, max_attempts, last_exc
            ) from last_exc

        return wrapper

    return decorator


if __name__ == "__main__":

    _CALL_COUNT = 0

    @retry((ConnectionError,), max_attempts=4, base_delay=0.05, max_delay=0.5)
    async def _flaky_database_ping() -> str:
        """Simulates a connection that succeeds on the third attempt.

        Returns:
            A success marker once the simulated connection recovers.

        Raises:
            ConnectionError: On the first two simulated attempts.
        """
        global _CALL_COUNT
        _CALL_COUNT += 1
        if _CALL_COUNT < 3:
            raise ConnectionError(f"connection dropped (call {_CALL_COUNT})")
        return "pong"

    async def _demo() -> None:
        """Exercises both the recovery and the exhaustion paths."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

        result = await _flaky_database_ping()
        logger.info("recovered with result=%r after %d calls", result, _CALL_COUNT)

        @retry((TimeoutError,), max_attempts=3, base_delay=0.05, max_delay=0.2)
        async def _always_times_out() -> None:
            raise TimeoutError("lock is still held")

        try:
            await _always_times_out()
        except MaxRetriesExceededError as exc:
            logger.info("gave up as expected: %s", exc)

    asyncio.run(_demo())
