"""Module 0: Unified Configuration.

Single source of truth for every tunable in the pipeline. Values are read
from a TOML file (stdlib ``tomllib``, Python 3.11+) and validated through
Pydantic v2 models, so a typo'd key or an out-of-range value fails loudly at
startup instead of surfacing as a mystery three modules downstream.

Resolution order
----------------
1. ``$DICA_CONFIG`` environment variable, if set.
2. ``./config.toml`` in the current working directory.
3. Built-in defaults (every field below has one) — DICA runs config-less.

The loaded config is cached process-wide via :func:`get_config`; call
:func:`load_config` directly in tests to bypass the cache.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import tomllib
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_ENV_VAR = "DICA_CONFIG"
_DEFAULT_FILENAME = "config.toml"


class OllamaConfig(BaseModel):
    """Connection and generation parameters for the local Ollama server."""

    model_config = ConfigDict(frozen=True)

    host: str = "http://localhost:11434"
    model: str = "qwen3-coder:30b"
    embed_model: str = "nomic-embed-text"
    num_ctx: int = Field(default=8192, ge=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    request_timeout_seconds: float = Field(default=300.0, gt=0)
    embed_timeout_seconds: float = Field(default=10.0, gt=0)


class DispatchConfig(BaseModel):
    """Hybrid retrieval scoring weights."""

    model_config = ConfigDict(frozen=True)

    # Used when agentic selection fails; fallback path takes top_k=1.
    top_k: int = Field(default=3, ge=1)
    lexical_weight: float = Field(default=1.0, ge=0.0)
    semantic_weight: float = Field(default=0.75, ge=0.0)
    structural_boost: float = Field(default=0.35, ge=0.0)
    name_boost: float = Field(default=0.15, ge=0.0)


class ContextConfig(BaseModel):
    """Token-budget controls for payload assembly.

    Prefer ``tokenizer="tiktoken"`` (default): BPE counts catch dense base64 /
    non-ASCII / long-identifier text that a static chars-per-token heuristic
    under-counts — under-counting lets prompts exceed ``num_ctx`` and local
    models left-truncate system instructions + target script silently.
    """

    model_config = ConfigDict(frozen=True)

    tokenizer: Literal["tiktoken", "heuristic"] = "tiktoken"
    tiktoken_encoding: str = Field(
        default="cl100k_base",
        description="tiktoken encoding used for packing (proxy for local models).",
    )
    # Fallback density when tokenizer=heuristic or tiktoken fails to load.
    chars_per_token: float = Field(default=3.2, gt=1.0)
    # Shrink the usable window slightly so model-specific tokenizers that run
    # denser than cl100k_base still leave headroom for the reserved reply.
    token_safety_margin: float = Field(default=0.05, ge=0.0, le=0.25)
    reserve_output_tokens: int = Field(default=1536, ge=0)
    max_diagnostic_tokens: int = Field(default=1200, ge=64)
    min_chunk_tokens: int = Field(default=64, ge=1)


class SandboxConfig(BaseModel):
    """Verification sandbox backend and container resource limits."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["docker", "local", "auto"] = "auto"
    image: str = "dica-sandbox:latest"
    timeout_seconds: float = Field(default=90.0, gt=0)
    mem_limit: str = "512m"
    pids_limit: int = Field(default=64, ge=8)
    cpu_quota_percent: int = Field(default=100, ge=10, le=800)


class EngineConfig(BaseModel):
    """Anchored generation retries and self-correction loop parameters."""

    model_config = ConfigDict(frozen=True)

    format_retries: int = Field(
        default=2,
        ge=0,
        description="Extra model retries after a failed code-extraction gate.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Sandbox self-correction attempts after verify failures.",
    )


class DicaConfig(BaseModel):
    """Root configuration object handed to every module."""

    model_config = ConfigDict(frozen=True)

    ollama: OllamaConfig = OllamaConfig()
    dispatch: DispatchConfig = DispatchConfig()
    context: ContextConfig = ContextConfig()
    sandbox: SandboxConfig = SandboxConfig()
    engine: EngineConfig = EngineConfig()


def _resolve_path(explicit: Path | None) -> Path | None:
    """Locate the config file per the documented resolution order."""
    if explicit is not None:
        return explicit
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env)
    cwd_default = Path.cwd() / _DEFAULT_FILENAME
    if cwd_default.is_file():
        return cwd_default
    return None


def load_config(path: Path | None = None) -> DicaConfig:
    """Load and validate configuration; fall back to defaults gracefully.

    A *missing* file is fine (defaults apply). A file that exists but fails
    to parse or validate raises immediately — silently ignoring a broken
    config the operator clearly intended to use is worse than crashing.
    """
    resolved = _resolve_path(path)
    if resolved is None:
        logger.info("No config file found; using built-in defaults.")
        return DicaConfig()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("rb") as fh:
        raw = tomllib.load(fh)
    config = DicaConfig.model_validate(raw)
    logger.info("Loaded config from %s", resolved)
    return config


@lru_cache(maxsize=1)
def get_config() -> DicaConfig:
    """Process-wide cached accessor (resolution order applies once)."""
    return load_config()
