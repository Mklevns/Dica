"""Gold-standard reference: async CRUD patterns with Pydantic v2 models.

This file is *corpus*, not framework code — the vault mines it for snippets.
Everything here is deliberately exemplary: full typing, Pydantic v2 idioms,
structured error handling, and clean async discipline.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class UserSchema(BaseModel):
    """Canonical user resource model (Pydantic v2)."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID = Field(default_factory=uuid4)
    username: str = Field(min_length=3, max_length=32)
    email: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AsyncUserRepository:
    """In-memory async CRUD repository for :class:`UserSchema`.

    Demonstrates the canonical async repository shape: every public method
    is a coroutine, mutation is guarded by a single lock, and lookups raise
    ``KeyError`` rather than returning ``None`` so callers can't silently
    ignore missing resources.
    """

    def __init__(self) -> None:
        self._store: dict[UUID, UserSchema] = {}
        self._lock = asyncio.Lock()

    async def create(self, user: UserSchema) -> UserSchema:
        """Insert a new user; rejects duplicate ids."""
        async with self._lock:
            if user.user_id in self._store:
                raise ValueError(f"duplicate user_id: {user.user_id}")
            self._store[user.user_id] = user
            return user

    async def read(self, user_id: UUID) -> UserSchema:
        """Fetch a user or raise ``KeyError``."""
        try:
            return self._store[user_id]
        except KeyError as exc:
            raise KeyError(f"user not found: {user_id}") from exc

    async def update(self, user_id: UUID, **changes: object) -> UserSchema:
        """Apply a partial update via model_copy (models are frozen)."""
        async with self._lock:
            current = await self.read(user_id)
            updated = current.model_copy(update=dict(changes))
            self._store[user_id] = updated
            return updated

    async def delete(self, user_id: UUID) -> None:
        """Remove a user; raises ``KeyError`` if absent."""
        async with self._lock:
            if user_id not in self._store:
                raise KeyError(f"user not found: {user_id}")
            del self._store[user_id]

    async def list_all(self) -> list[UserSchema]:
        """Snapshot of all users, newest first."""
        return sorted(
            self._store.values(), key=lambda u: u.created_at, reverse=True
        )


async def gather_with_limit[T](
    coros: list[object], limit: int = 8
) -> list[T]:
    """Run coroutines concurrently under a semaphore-enforced concurrency cap.

    The canonical bounded-fan-out pattern: wrap each coroutine so it must
    acquire the semaphore before running, then gather the wrappers.
    """
    semaphore = asyncio.Semaphore(limit)

    async def bounded(coro: object) -> T:
        async with semaphore:
            return await coro  # type: ignore[misc]

    return await asyncio.gather(*(bounded(c) for c in coros))
