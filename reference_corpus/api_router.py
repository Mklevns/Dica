"""Gold-standard pattern: asynchronous FastAPI router with Pydantic v2.

Demonstrates the canonical shape of a production REST resource router:
strict request/response schemas via Pydantic v2 (``ConfigDict``,
``model_validate``), constructor-free dependency injection through
``Annotated[..., Depends(...)]``, domain exceptions translated to HTTP
errors at the boundary, and structured logging throughout.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ItemNotFoundError(Exception):
    """Raised by the service layer when an item id does not exist.

    Domain exceptions stay HTTP-agnostic: the router is the only layer
    allowed to translate them into ``HTTPException``.
    """

    def __init__(self, item_id: UUID) -> None:
        self.item_id = item_id
        super().__init__(f"item not found: {item_id}")


class DuplicateItemError(Exception):
    """Raised when a create request collides with an existing item name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"item name already exists: {name}")


class ItemCreateRequest(BaseModel):
    """Inbound payload for creating an item.

    Pattern: request models are strict (``extra='forbid'``) so unknown
    fields fail loudly at the edge instead of being silently dropped.

    Attributes:
        name: Human-readable unique item name, 1-128 chars, whitespace
            stripped by the model config.
        quantity: Non-negative stock count.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    quantity: int = Field(ge=0)


class ItemResponse(BaseModel):
    """Outbound representation of a stored item.

    Pattern: response models are frozen (immutable) and enable
    ``from_attributes`` so ``model_validate`` can lift ORM objects or any
    attribute-bearing domain object directly into the schema.

    Attributes:
        item_id: Server-assigned UUID primary key.
        name: Unique item name.
        quantity: Current stock count.
        created_at: Timezone-aware creation timestamp (UTC).
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    item_id: UUID
    name: str
    quantity: int
    created_at: datetime


class InMemoryItemService:
    """Async service layer backing the router with a dict store.

    Pattern: the router never touches storage directly; it depends on a
    service whose public surface is entirely coroutines, so a database
    implementation can be swapped in without changing a single route.
    """

    def __init__(self) -> None:
        self._items: dict[UUID, ItemResponse] = {}

    async def create(self, request: ItemCreateRequest) -> ItemResponse:
        """Create and store a new item.

        Args:
            request: Validated inbound creation payload.

        Returns:
            The stored item including its server-assigned id.

        Raises:
            DuplicateItemError: If an item with the same name exists.
        """
        if any(item.name == request.name for item in self._items.values()):
            raise DuplicateItemError(request.name)
        item = ItemResponse(
            item_id=uuid4(),
            name=request.name,
            quantity=request.quantity,
            created_at=datetime.now(UTC),
        )
        self._items[item.item_id] = item
        logger.info("Created item %s (%s)", item.item_id, item.name)
        return item

    async def get(self, item_id: UUID) -> ItemResponse:
        """Fetch a single item by id.

        Args:
            item_id: Primary key of the item.

        Returns:
            The matching item.

        Raises:
            ItemNotFoundError: If no item has ``item_id``.
        """
        try:
            return self._items[item_id]
        except KeyError as exc:
            raise ItemNotFoundError(item_id) from exc

    async def list_all(self) -> list[ItemResponse]:
        """Return all items, newest first.

        Returns:
            Items sorted by ``created_at`` descending.
        """
        return sorted(
            self._items.values(), key=lambda item: item.created_at, reverse=True
        )

    async def delete(self, item_id: UUID) -> None:
        """Remove an item by id.

        Args:
            item_id: Primary key of the item to remove.

        Raises:
            ItemNotFoundError: If no item has ``item_id``.
        """
        if item_id not in self._items:
            raise ItemNotFoundError(item_id)
        del self._items[item_id]
        logger.info("Deleted item %s", item_id)


_service_singleton = InMemoryItemService()


def get_item_service() -> InMemoryItemService:
    """FastAPI dependency provider for the item service.

    Pattern: providers are plain callables so tests can override them via
    ``app.dependency_overrides[get_item_service] = fake_factory``.

    Returns:
        The process-wide service instance.
    """
    return _service_singleton


ItemServiceDep = Annotated[InMemoryItemService, Depends(get_item_service)]

router = APIRouter(prefix="/items", tags=["items"])


@router.post("", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def create_item(
    payload: ItemCreateRequest, service: ItemServiceDep
) -> ItemResponse:
    """FastAPI POST endpoint: create a new item resource.

    Pattern: async route handler with dependency injection via
    ``Annotated[..., Depends]``; domain exceptions are translated to
    HTTP status codes only at this boundary.

    Args:
        payload: Validated creation request body.
        service: Injected item service.

    Returns:
        The created item with server-assigned fields.

    Raises:
        HTTPException: 409 if the item name already exists.
    """
    try:
        return await service.create(payload)
    except DuplicateItemError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.get("/{item_id}", response_model=ItemResponse)
async def read_item(item_id: UUID, service: ItemServiceDep) -> ItemResponse:
    """FastAPI GET endpoint: fetch one item resource by id.

    Pattern: typed path parameter, injected service dependency, 404
    translation at the route boundary.

    Args:
        item_id: UUID path parameter identifying the item.
        service: Injected item service.

    Returns:
        The matching item.

    Raises:
        HTTPException: 404 if the item does not exist.
    """
    try:
        return await service.get(item_id)
    except ItemNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.get("", response_model=list[ItemResponse])
async def list_items(service: ItemServiceDep) -> list[ItemResponse]:
    """FastAPI GET endpoint: list the item resource collection.

    Pattern: collection route returning a typed response model list via
    an injected async service dependency.

    Args:
        service: Injected item service.

    Returns:
        All stored items sorted by creation time descending.
    """
    return await service.list_all()


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: UUID, service: ItemServiceDep) -> None:
    """FastAPI DELETE endpoint: remove one item resource by id.

    Pattern: 204 No Content on success, injected service dependency,
    404 translation at the route boundary.

    Args:
        item_id: UUID path parameter identifying the item.
        service: Injected item service.

    Raises:
        HTTPException: 404 if the item does not exist.
    """
    try:
        await service.delete(item_id)
    except ItemNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
