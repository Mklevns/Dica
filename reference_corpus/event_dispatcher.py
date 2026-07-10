"""Gold-standard pattern: async pub-sub event dispatcher on ``asyncio.Queue``.

Demonstrates the canonical in-process eventing shape: an immutable
Pydantic ``Event`` envelope, topic-keyed subscription of coroutine
handlers, a single queue-draining worker task, fan-out with per-handler
error isolation (one failing subscriber never starves the others), and a
drain-then-cancel graceful shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

EventHandler = Callable[["Event"], Awaitable[None]]
"""Type alias: a subscriber is any coroutine function taking one Event."""


class DispatcherError(Exception):
    """Base class for all dispatcher failures."""


class DispatcherClosedError(DispatcherError):
    """Raised when publishing to a dispatcher that has been stopped.

    Pattern: publishing after shutdown is a programming error and must
    fail loudly rather than silently dropping events.
    """

    def __init__(self) -> None:
        super().__init__("dispatcher is closed; publish rejected")


class DispatcherNotStartedError(DispatcherError):
    """Raised when stopping a dispatcher whose worker was never started."""

    def __init__(self) -> None:
        super().__init__("dispatcher worker was never started")


class Event(BaseModel):
    """Immutable event envelope routed by topic.

    Pattern: events are frozen value objects — handlers can never mutate
    a shared event, which makes concurrent fan-out safe by construction.

    Attributes:
        event_id: Unique id for tracing/deduplication.
        topic: Routing key that selects which subscribers run.
        payload: Arbitrary JSON-safe event data.
        published_at: Timezone-aware UTC publication timestamp.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    topic: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    published_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AsyncEventDispatcher:
    """Topic-based pub-sub dispatcher backed by one ``asyncio.Queue``.

    Pattern: publishers enqueue and return immediately (backpressure via
    ``maxsize``); a single worker coroutine drains the queue and fans
    each event out to its topic's handlers concurrently. Handler failures
    are logged and isolated, never propagated to the publisher.
    """

    def __init__(self, *, max_queue_size: int = 1024) -> None:
        """Initialize an idle dispatcher.

        Args:
            max_queue_size: Queue capacity; ``publish`` awaits (applies
                backpressure) once the queue is full.
        """
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._handlers: defaultdict[str, list[EventHandler]] = defaultdict(list)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Register a coroutine handler for a topic.

        Args:
            topic: Routing key the handler should receive events for.
            handler: Coroutine function invoked once per matching event.
        """
        self._handlers[topic].append(handler)
        logger.debug("Subscribed %r to topic %r", handler, topic)

    async def publish(self, event: Event) -> None:
        """Enqueue an event for asynchronous delivery.

        Args:
            event: The event to route.

        Raises:
            DispatcherClosedError: If the dispatcher has been stopped.
        """
        if self._closed:
            raise DispatcherClosedError()
        await self._queue.put(event)

    async def start(self) -> None:
        """Spawn the queue-draining worker task (idempotent)."""
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="event-dispatcher")
            logger.info("Dispatcher worker started")

    async def stop(self) -> None:
        """Gracefully shut down: drain, then cancel the worker.

        Pattern: refuse new events first, await ``queue.join()`` so every
        already-published event is fully handled, then cancel the worker
        and suppress its ``CancelledError``.

        Raises:
            DispatcherNotStartedError: If ``start`` was never called.
        """
        if self._worker is None:
            raise DispatcherNotStartedError()
        self._closed = True
        await self._queue.join()
        self._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        self._worker = None
        logger.info("Dispatcher worker stopped")

    async def _run(self) -> None:
        """Worker loop: pull one event at a time and fan it out.

        ``task_done`` is called in a ``finally`` so ``queue.join()`` in
        :meth:`stop` can never deadlock, even if dispatch itself fails.
        """
        while True:
            event = await self._queue.get()
            try:
                await self._dispatch(event)
            finally:
                self._queue.task_done()

    async def _dispatch(self, event: Event) -> None:
        """Fan one event out to all handlers for its topic, concurrently.

        Pattern: ``asyncio.gather(..., return_exceptions=True)`` gives
        per-handler error isolation — every handler always runs, and each
        failure is logged with full traceback against the offending
        handler.

        Args:
            event: The event to deliver.
        """
        handlers = self._handlers.get(event.topic, [])
        if not handlers:
            logger.debug(
                "No handlers for topic %r; dropping %s", event.topic, event.event_id
            )
            return
        results = await asyncio.gather(
            *(handler(event) for handler in handlers), return_exceptions=True
        )
        for handler, result in zip(handlers, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "Handler %r failed for event %s on topic %r",
                    handler,
                    event.event_id,
                    event.topic,
                    exc_info=result,
                )
