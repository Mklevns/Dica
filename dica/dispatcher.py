"""Module 2: Intent Dispatcher.

Turns a raw user prompt ("Build an async CRUD router") into the top-K most
*structurally relevant* :class:`~dica.vault.CodeChunk` records from the vault.

Two-stage relevance model
-------------------------
1. **Lexical overlap** — the prompt is tokenized with the same tokenizer the
   vault used at ingest time (crucial: query and index must share a token
   space), producing a base Jaccard-style score.
2. **Structural boosts** — the prompt is scanned for *intent markers*
   ("async", "class", "pydantic", "schema", ...) which map onto the boolean
   ``ChunkTags`` computed from the AST. A chunk whose structure matches the
   detected intent gets an additive boost. This is what lets a query about
   "async" retrieval prefer an ``async def`` even when keyword overlap ties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from dica.vault import ChunkTags, CodeChunk, CodeVault, tokenize

logger = logging.getLogger(__name__)

# Prompt vocabulary -> ChunkTags attribute. Each hit adds STRUCTURAL_BOOST
# to matching chunks. Kept as data so the mapping is trivially extensible.
_INTENT_MARKERS: dict[str, tuple[str, ...]] = {
    "is_async": ("async", "await", "asyncio", "concurrent", "nonblocking", "coroutine"),
    "is_class": ("class", "model", "schema", "object", "dataclass", "repository"),
    "has_pydantic": ("pydantic", "schema", "validation", "validator", "basemodel"),
    "has_decorators": ("decorator", "route", "router", "endpoint", "fixture"),
}

_STRUCTURAL_BOOST: float = 0.35  # per matched structural intent
_KIND_NAME_BOOST: float = 0.15   # query token appears verbatim in chunk name


class DispatchResult(BaseModel):
    """A ranked retrieval hit with its scoring breakdown (for observability)."""

    model_config = ConfigDict(frozen=True)

    chunk: CodeChunk
    lexical_score: float
    structural_score: float

    @property
    def total(self) -> float:
        return self.lexical_score + self.structural_score


@dataclass(slots=True)
class Intent:
    """Parsed representation of what the user structurally asked for."""

    keywords: frozenset[str]
    wanted_tags: dict[str, bool] = field(default_factory=dict)


class IntentDispatcher:
    """Matches user prompts against the code vault."""

    def __init__(self, vault: CodeVault) -> None:
        self._vault = vault

    def parse_intent(self, prompt: str) -> Intent:
        """Extract keywords and structural intent flags from the raw prompt."""
        keywords = tokenize(prompt)
        wanted: dict[str, bool] = {}
        for tag, markers in _INTENT_MARKERS.items():
            if keywords & frozenset(markers):
                wanted[tag] = True
        logger.debug("Parsed intent: keywords=%s wanted_tags=%s", keywords, wanted)
        return Intent(keywords=keywords, wanted_tags=wanted)

    def dispatch(self, prompt: str, *, top_k: int = 2) -> list[DispatchResult]:
        """Return the ``top_k`` most relevant chunks for ``prompt``.

        Structural tags are used as *soft boosts*, not hard filters: a hard
        filter on a small corpus can return nothing, and a slightly-off
        reference is far more useful to the LLM than an empty one.
        """
        intent = self.parse_intent(prompt)
        scored: list[DispatchResult] = []

        for chunk, lexical in self._vault.search(intent.keywords):
            structural = self._structural_score(chunk.tags, intent)
            # Verbatim name hits ("router" in `async_crud_router`) are strong
            # signals that the user is pointing at this exact pattern.
            name_tokens = tokenize(chunk.name)
            if intent.keywords & name_tokens:
                structural += _KIND_NAME_BOOST
            scored.append(
                DispatchResult(
                    chunk=chunk, lexical_score=lexical, structural_score=structural
                )
            )

        scored.sort(key=lambda r: r.total, reverse=True)
        top = scored[:top_k]
        for rank, result in enumerate(top, start=1):
            logger.info(
                "Dispatch #%d: %s (lexical=%.3f structural=%.3f)",
                rank,
                result.chunk.name,
                result.lexical_score,
                result.structural_score,
            )
        return top

    @staticmethod
    def _structural_score(tags: ChunkTags, intent: Intent) -> float:
        """Additive boost for each structural intent the chunk satisfies."""
        return sum(
            _STRUCTURAL_BOOST
            for tag, wanted in intent.wanted_tags.items()
            if wanted and getattr(tags, tag)
        )
