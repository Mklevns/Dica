"""Gold-standard pattern: generic async repository over SQLAlchemy 2.0.

Demonstrates the canonical data-access shape: fully typed ``Mapped`` ORM
models on a ``DeclarativeBase``, a ``Generic[ModelT]`` repository whose
methods are all coroutines, an ``asynccontextmanager`` session scope that
owns commit/rollback/close, and domain exceptions that wrap
``SQLAlchemyError`` so callers never depend on driver internals.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Generic, TypeVar
from uuid import uuid4

from sqlalchemy import DateTime, Integer, String, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Base class for all data-access failures.

    Pattern: every storage exception surfaces as (a subclass of) this
    type, so service code catches one stable exception family instead of
    driver-specific errors.
    """


class EntityNotFoundError(RepositoryError):
    """Raised when a primary-key lookup matches no row.

    Pattern: lookups raise instead of returning ``None`` so a missing
    entity can never be silently ignored by the caller.
    """

    def __init__(self, model_name: str, entity_id: str) -> None:
        self.model_name = model_name
        self.entity_id = entity_id
        super().__init__(f"{model_name} not found: {entity_id}")


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models in the application."""


class UserModel(Base):
    """Example ORM model with fully typed ``Mapped`` columns.

    Pattern: SQLAlchemy 2.0 style — every column is declared with
    ``Mapped[...]`` + ``mapped_column`` so mypy knows exact attribute
    types; string UUIDs keep the model portable across SQLite/Postgres.

    Attributes:
        user_id: String UUID primary key, generated server-side default.
        username: Unique login name, max 64 chars.
        email: Contact address, max 255 chars.
        created_at: Timezone-aware creation timestamp.
    """

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    login_count: Mapped[int] = mapped_column(Integer, default=0)


ModelT = TypeVar("ModelT", bound=Base)


class AsyncRepository(Generic[ModelT]):
    """Generic async CRUD repository parameterized over an ORM model.

    Pattern: one reusable class provides typed CRUD for any ``Base``
    subclass; concrete repositories are just
    ``AsyncRepository(session, UserModel)``. The repository borrows a
    session — transaction lifecycle belongs to :func:`session_scope`.
    """

    def __init__(self, session: AsyncSession, model: type[ModelT]) -> None:
        """Bind the repository to a live session and a model class.

        Args:
            session: An open ``AsyncSession``; the repository never
                commits or closes it.
            model: The ORM class this repository operates on.
        """
        self._session = session
        self._model = model

    async def add(self, entity: ModelT) -> ModelT:
        """Stage a new entity for insertion.

        Args:
            entity: A transient ORM instance.

        Returns:
            The same instance, flushed so defaults (e.g. generated ids)
            are populated.

        Raises:
            RepositoryError: If the flush fails at the database layer.
        """
        try:
            self._session.add(entity)
            await self._session.flush()
        except SQLAlchemyError as exc:
            logger.exception("Failed to add %s", self._model.__name__)
            raise RepositoryError(f"add failed for {self._model.__name__}") from exc
        return entity

    async def get(self, entity_id: str) -> ModelT:
        """Fetch one entity by primary key.

        Args:
            entity_id: Primary-key value.

        Returns:
            The matching ORM instance.

        Raises:
            EntityNotFoundError: If no row matches ``entity_id``.
            RepositoryError: If the query fails at the database layer.
        """
        try:
            entity = await self._session.get(self._model, entity_id)
        except SQLAlchemyError as exc:
            logger.exception("Failed to get %s", self._model.__name__)
            raise RepositoryError(f"get failed for {self._model.__name__}") from exc
        if entity is None:
            raise EntityNotFoundError(self._model.__name__, entity_id)
        return entity

    async def list_all(self, *, limit: int = 100, offset: int = 0) -> Sequence[ModelT]:
        """Fetch a page of entities.

        Args:
            limit: Maximum number of rows to return.
            offset: Number of rows to skip.

        Returns:
            A sequence of ORM instances (possibly empty).

        Raises:
            RepositoryError: If the query fails at the database layer.
        """
        try:
            result = await self._session.execute(
                select(self._model).limit(limit).offset(offset)
            )
        except SQLAlchemyError as exc:
            logger.exception("Failed to list %s", self._model.__name__)
            raise RepositoryError(f"list failed for {self._model.__name__}") from exc
        return result.scalars().all()

    async def delete(self, entity_id: str) -> None:
        """Delete one entity by primary key.

        Args:
            entity_id: Primary-key value.

        Raises:
            EntityNotFoundError: If no row matches ``entity_id``.
            RepositoryError: If the delete fails at the database layer.
        """
        entity = await self.get(entity_id)
        try:
            await self._session.delete(entity)
            await self._session.flush()
        except SQLAlchemyError as exc:
            logger.exception("Failed to delete %s", self._model.__name__)
            raise RepositoryError(f"delete failed for {self._model.__name__}") from exc


def create_engine_and_factory(
    database_url: str, *, echo: bool = False
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Build the process-wide engine and session factory.

    Pattern: exactly one engine per process; sessions are cheap and
    created per unit-of-work from the returned factory.

    Args:
        database_url: Async driver URL, e.g.
            ``sqlite+aiosqlite:///./app.db`` or
            ``postgresql+asyncpg://user:pass@host/db``.
        echo: Whether SQLAlchemy logs emitted SQL.

    Returns:
        The engine and a bound ``async_sessionmaker``.
    """
    engine = create_async_engine(database_url, echo=echo)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Provide a transactional scope around one unit of work.

    Pattern: commit on clean exit, rollback on any exception, close
    always. Database failures are wrapped in :class:`RepositoryError`;
    non-database exceptions still roll back but propagate unchanged.

    Args:
        factory: The application's session factory.

    Yields:
        An open session; the caller must not commit or close it.

    Raises:
        RepositoryError: If commit fails or a database error escapes the
            block.
    """
    session = factory()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        logger.exception("Transaction rolled back")
        raise RepositoryError("transaction failed") from exc
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def initialize_schema(engine: AsyncEngine) -> None:
    """Create all tables known to the declarative base.

    Args:
        engine: The application's async engine.
    """
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("Schema initialized")
