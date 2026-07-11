"""Shared fixtures for the DICA test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from dica.config import (
    ContextConfig,
    DispatchConfig,
    OllamaConfig,
    get_config,
)
from dica.context import ContextBudget
from dica.vault import CodeVault

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "reference_corpus"

FENCE = chr(96) * 3  # ``` without PowerShell eating backticks


@pytest.fixture(autouse=True)
def _clear_config_cache() -> None:
    """Ensure each test sees a fresh config cache."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def corpus_vault() -> CodeVault:
    vault = CodeVault()
    count = vault.ingest_path(CORPUS_DIR)
    assert count > 0, f"expected chunks from {CORPUS_DIR}"
    return vault


@pytest.fixture
def tight_budget() -> ContextBudget:
    """Small window to force drops / truncation in budget tests."""
    return ContextBudget(
        OllamaConfig(num_ctx=900),
        ContextConfig(
            reserve_output_tokens=200,
            max_diagnostic_tokens=64,
            min_chunk_tokens=10,
            chars_per_token=3.2,
        ),
        num_ctx=900,
    )


@pytest.fixture
def roomy_budget() -> ContextBudget:
    return ContextBudget(
        OllamaConfig(num_ctx=8192),
        ContextConfig(),
        num_ctx=8192,
    )


@pytest.fixture
def lexical_dispatch_config() -> DispatchConfig:
    return DispatchConfig(
        top_k=3,
        lexical_weight=1.0,
        semantic_weight=0.0,
        structural_boost=0.35,
        name_boost=0.15,
    )


def fenced_python(code: str) -> str:
    """Wrap ``code`` in a markdown python fence."""
    return f"{FENCE}python\n{code}\n{FENCE}\n"
