"""Module 2: Intent Dispatcher.

Turns a raw user prompt ("Build an async CRUD router") into the top-K most
*structurally relevant* :class:`~dica.vault.CodeChunk` records from the vault.

Hybrid relevance model
----------------------
1. **Lexical overlap** — the prompt is tokenized with the same tokenizer the
   vault used at ingest time, producing a base score scaled by
   ``DispatchConfig.lexical_weight``.
2. **Structural boosts** — intent markers map onto ``ChunkTags``; matching
   tags add ``structural_boost`` / ``name_boost``.
3. **Semantic similarity** (optional) — when a :class:`~dica.embeddings.SemanticIndex`
   is available and ``semantic_weight > 0``, cosine similarity against the
   local embed model is added (scaled by ``semantic_weight``). If the index
   is missing or Ollama is down, scoring degrades silently to lexical +
   structural only.

Scoring weights and default ``top_k`` come from :class:`~dica.config.DispatchConfig`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from dica.config import DispatchConfig, get_config
from dica.vault import ChunkTags, CodeChunk, CodeVault, tokenize

if TYPE_CHECKING:
    from dica.embeddings import SemanticIndex

logger = logging.getLogger(__name__)

# Prompt vocabulary -> ChunkTags attribute. Each hit adds structural_boost
# to matching chunks. Kept as data so the mapping is trivially extensible.
_INTENT_MARKERS: dict[str, tuple[str, ...]] = {
    "is_async": ("async", "await", "asyncio", "concurrent", "nonblocking", "coroutine"),
    "is_class": ("class", "model", "schema", "object", "dataclass", "repository"),
    "has_pydantic": ("pydantic", "schema", "validation", "validator", "basemodel"),
    "has_decorators": ("decorator", "route", "router", "endpoint", "fixture"),
}


class DispatchResult(BaseModel):
    """A ranked retrieval hit with its scoring breakdown (for observability)."""

    model_config = ConfigDict(frozen=True)

    chunk: CodeChunk
    lexical_score: float
    structural_score: float
    semantic_score: float = Field(
        default=0.0,
        description="Weighted cosine similarity (0 when semantic is off/unavailable).",
    )

    @property
    def total(self) -> float:
        return self.lexical_score + self.structural_score + self.semantic_score


@dataclass(slots=True)
class Intent:
    """Parsed representation of what the user structurally asked for."""

    keywords: frozenset[str]
    wanted_tags: dict[str, bool] = field(default_factory=dict)


class IntentDispatcher:
    """Matches user prompts against the code vault (optionally hybrid semantic)."""

    def __init__(
        self,
        vault: CodeVault,
        config: DispatchConfig | None = None,
        *,
        semantic_index: SemanticIndex | None = None,
    ) -> None:
        self._vault = vault
        self._cfg = config if config is not None else get_config().dispatch
        self._semantic = semantic_index

    def parse_intent(self, prompt: str) -> Intent:
        """Extract keywords and structural intent flags from the raw prompt."""
        keywords = tokenize(prompt)
        wanted: dict[str, bool] = {}
        for tag, markers in _INTENT_MARKERS.items():
            if keywords & frozenset(markers):
                wanted[tag] = True
        logger.debug("Parsed intent: keywords=%s wanted_tags=%s", keywords, wanted)
        return Intent(keywords=keywords, wanted_tags=wanted)

    def dispatch(self, prompt: str, *, top_k: int | None = None) -> list[DispatchResult]:
        """Return the ``top_k`` most relevant chunks for ``prompt``.

        When ``top_k`` is omitted, :attr:`DispatchConfig.top_k` is used.

        Structural tags are *soft boosts*, not hard filters. Semantic scores
        are fail-soft: an unavailable index contributes 0.0 everywhere.
        """
        k = self._cfg.top_k if top_k is None else top_k
        intent = self.parse_intent(prompt)

        lexical_by_id: dict[str, float] = {
            chunk.chunk_id: score
            for chunk, score in self._vault.search(intent.keywords)
        }

        use_semantic = (
            self._cfg.semantic_weight > 0.0
            and self._semantic is not None
            and self._semantic.available
        )

        if use_semantic:
            # Full vault: semantic can surface chunks with weak/zero keyword overlap.
            candidates: list[CodeChunk] = list(self._vault)
            semantic_map = self._semantic.score_sync(
                prompt, (c.chunk_id for c in candidates)
            )
        elif lexical_by_id:
            candidates = [c for c in self._vault if c.chunk_id in lexical_by_id]
            semantic_map = {}
        else:
            logger.info("Dispatch: no lexical hits and semantic unavailable.")
            return []

        scored: list[DispatchResult] = []
        for chunk in candidates:
            weighted_lexical = (
                lexical_by_id.get(chunk.chunk_id, 0.0) * self._cfg.lexical_weight
            )
            structural = self._structural_score(chunk.tags, intent)
            name_tokens = tokenize(chunk.name)
            if intent.keywords & name_tokens:
                structural += self._cfg.name_boost
            weighted_semantic = (
                semantic_map.get(chunk.chunk_id, 0.0) * self._cfg.semantic_weight
            )
            # Drop pure noise when ranking the full vault under semantic mode.
            if (
                weighted_lexical == 0.0
                and structural == 0.0
                and weighted_semantic == 0.0
            ):
                continue
            scored.append(
                DispatchResult(
                    chunk=chunk,
                    lexical_score=weighted_lexical,
                    structural_score=structural,
                    semantic_score=weighted_semantic,
                )
            )

        scored.sort(key=lambda r: r.total, reverse=True)
        top = scored[:k]
        mode = "hybrid" if use_semantic else "lexical"
        for rank, result in enumerate(top, start=1):
            logger.info(
                "Dispatch #%d [%s]: %s "
                "(lexical=%.3f structural=%.3f semantic=%.3f total=%.3f)",
                rank,
                mode,
                result.chunk.name,
                result.lexical_score,
                result.structural_score,
                result.semantic_score,
                result.total,
            )
        return top

    def _structural_score(self, tags: ChunkTags, intent: Intent) -> float:
        """Additive boost for each structural intent the chunk satisfies."""
        boost = self._cfg.structural_boost
        return sum(
            boost
            for tag, wanted in intent.wanted_tags.items()
            if wanted and getattr(tags, tag)
        )
