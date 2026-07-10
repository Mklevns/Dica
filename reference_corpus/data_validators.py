"""Strict Pydantic V2 validation for untrusted analytics configuration payloads.

This module is a reference implementation of modern Pydantic V2 mechanics:

* ``ConfigDict(extra="forbid", str_strip_whitespace=True)`` on every model,
  so unknown keys are rejected and string inputs are normalized.
* ``@field_validator(mode="before")`` for input coercion of untrusted shapes.
* ``@model_validator(mode="after")`` for cross-field invariants.
* ``model_validate_json()`` guarded by ``except ValidationError`` at the
  trust boundary, re-raised as a domain-specific error.

Only V2 APIs are used. V1 constructs (``@validator``, ``parse_raw``,
``class Config``) are intentionally absent and must not be introduced.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyticsConfig",
    "ConfigurationError",
    "Environment",
    "RetentionPolicy",
    "SinkConfig",
    "load_analytics_config",
]


class ConfigurationError(RuntimeError):
    """Raised when an untrusted payload fails schema validation.

    Attributes:
        error_count: Number of individual validation errors reported
            by Pydantic for the rejected payload.
    """

    def __init__(self, message: str, error_count: int) -> None:
        """Initializes the error.

        Args:
            message: Human-readable summary of the failure.
            error_count: Number of individual validation errors.
        """
        self.error_count = error_count
        super().__init__(message)


class Environment(str, Enum):
    """Deployment environment for an analytics pipeline."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class RetentionPolicy(BaseModel):
    """Data retention rules for stored analytics events.

    Attributes:
        days: Retention window in days; bounded to a sane range.
        archive_on_expiry: Whether expired data is archived rather
            than deleted outright.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    days: int = Field(ge=1, le=3650)
    archive_on_expiry: bool = False


class SinkConfig(BaseModel):
    """A single downstream sink that receives analytics events.

    Attributes:
        name: Unique sink identifier within a configuration.
        endpoint: Destination URI; scheme is validated before parsing.
        batch_size: Events per flush; bounded to protect the sink.
        tags: Free-form routing tags; a comma-separated string is
            coerced into a list before validation.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: str = Field(min_length=1, max_length=64)
    endpoint: str = Field(min_length=1)
    batch_size: int = Field(default=500, ge=1, le=10_000)
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: object) -> object:
        """Coerces a comma-separated string into a list of tags.

        Runs *before* Pydantic's own parsing, so it must accept any
        untrusted shape and only normalize the cases it recognizes.

        Args:
            value: Raw, untrusted input for the ``tags`` field.

        Returns:
            A list of stripped, non-empty tags if ``value`` was a
            string; otherwise the value unchanged, so Pydantic's
            normal ``list[str]`` validation still applies.
        """
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("endpoint", mode="after")
    @classmethod
    def _check_endpoint_scheme(cls, value: str) -> str:
        """Rejects endpoints that do not use an allowed URI scheme.

        Args:
            value: The already-parsed, whitespace-stripped endpoint.

        Returns:
            The validated endpoint string, unchanged.

        Raises:
            ValueError: If the endpoint lacks an allowed scheme.
        """
        allowed = ("https://", "kafka://", "file://")
        if not value.startswith(allowed):
            raise ValueError(
                f"endpoint must start with one of {allowed}, got {value!r}"
            )
        return value


class AnalyticsConfig(BaseModel):
    """Top-level analytics pipeline configuration.

    Attributes:
        project: Project identifier.
        environment: Target deployment environment.
        created_at: Payload creation time; epoch seconds are coerced
            to timezone-aware UTC datetimes before parsing.
        sampling_rate: Fraction of events retained, in ``[0.0, 1.0]``.
        retention: Nested retention policy.
        sinks: One or more downstream sinks.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    project: str = Field(min_length=1, max_length=128)
    environment: Environment
    created_at: datetime
    sampling_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    retention: RetentionPolicy
    sinks: list[SinkConfig] = Field(min_length=1)

    @field_validator("created_at", mode="before")
    @classmethod
    def _coerce_epoch_seconds(cls, value: object) -> object:
        """Accepts Unix epoch seconds in addition to ISO-8601 strings.

        Args:
            value: Raw, untrusted input for the ``created_at`` field.

        Returns:
            A timezone-aware UTC ``datetime`` if ``value`` was numeric;
            otherwise the value unchanged for normal datetime parsing.
        """
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return value

    @model_validator(mode="after")
    def _enforce_cross_field_invariants(self) -> Self:
        """Enforces invariants that span multiple fields.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If sink names collide, or if a production
                configuration attempts to downsample events.
        """
        names = [sink.name for sink in self.sinks]
        if len(names) != len(set(names)):
            raise ValueError(f"sink names must be unique, got {names!r}")
        if (
            self.environment is Environment.PRODUCTION
            and self.sampling_rate < 1.0
        ):
            raise ValueError(
                "production configurations must not downsample: "
                f"sampling_rate={self.sampling_rate}"
            )
        return self


def load_analytics_config(raw_json: str | bytes) -> AnalyticsConfig:
    """Parses and validates an untrusted JSON payload at the trust boundary.

    ``model_validate_json`` parses and validates in a single pass (no
    intermediate ``json.loads``), which is both faster and stricter.

    Args:
        raw_json: Untrusted JSON document as text or UTF-8 bytes.

    Returns:
        A fully validated ``AnalyticsConfig`` instance.

    Raises:
        ConfigurationError: If the payload is malformed JSON or fails
            schema validation. The original ``ValidationError`` is
            preserved as ``__cause__``.
    """
    try:
        return AnalyticsConfig.model_validate_json(raw_json)
    except ValidationError as exc:
        errors = exc.errors()
        for err in errors:
            logger.error(
                "config rejected: loc=%s type=%s msg=%s",
                err["loc"],
                err["type"],
                err["msg"],
            )
        raise ConfigurationError(
            f"analytics config failed validation with {len(errors)} error(s)",
            error_count=len(errors),
        ) from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    _VALID_PAYLOAD = """
    {
        "project": "  edge-telemetry  ",
        "environment": "staging",
        "created_at": 1767225600,
        "sampling_rate": 0.25,
        "retention": {"days": 90, "archive_on_expiry": true},
        "sinks": [
            {
                "name": "warehouse",
                "endpoint": "https://ingest.example.internal/v1",
                "tags": "gpu, edge , telemetry"
            }
        ]
    }
    """

    config = load_analytics_config(_VALID_PAYLOAD)
    logger.info("accepted: %s", config.model_dump_json(indent=2))

    _INVALID_PAYLOAD = '{"project": "x", "environment": "prod", "junk": 1}'
    try:
        load_analytics_config(_INVALID_PAYLOAD)
    except ConfigurationError as boundary_exc:
        logger.info("rejected as expected: %s", boundary_exc)
