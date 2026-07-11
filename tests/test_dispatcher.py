"""Tests for hybrid intent dispatch."""

from __future__ import annotations

from dica.config import DispatchConfig
from dica.dispatcher import IntentDispatcher
from dica.vault import CodeVault


class _FakeSemanticIndex:
    def __init__(self, boost_id: str, *, available: bool = True) -> None:
        self.boost_id = boost_id
        self.available = available

    def score_sync(self, query: str, chunk_ids: object) -> dict[str, float]:
        return {
            cid: (1.0 if cid == self.boost_id else 0.05)
            for cid in chunk_ids  # type: ignore[union-attr]
        }


def test_lexical_dispatch_ranks_async_crud(corpus_vault: CodeVault) -> None:
    dispatcher = IntentDispatcher(
        corpus_vault,
        config=DispatchConfig(top_k=3, semantic_weight=0.0),
    )
    hits = dispatcher.dispatch("async CRUD pydantic router")
    assert len(hits) <= 3
    assert hits
    assert all(h.semantic_score == 0.0 for h in hits)
    # Totals should be non-increasing
    totals = [h.total for h in hits]
    assert totals == sorted(totals, reverse=True)


def test_parse_intent_tags(corpus_vault: CodeVault) -> None:
    d = IntentDispatcher(corpus_vault, config=DispatchConfig(semantic_weight=0.0))
    intent = d.parse_intent("build an async pydantic model with validators")
    assert intent.wanted_tags.get("is_async") is True
    assert intent.wanted_tags.get("has_pydantic") is True


def test_semantic_boost_changes_ranking(corpus_vault: CodeVault) -> None:
    target = list(corpus_vault)[-1]
    fake = _FakeSemanticIndex(target.chunk_id)
    dispatcher = IntentDispatcher(
        corpus_vault,
        config=DispatchConfig(
            top_k=3,
            lexical_weight=0.1,
            semantic_weight=5.0,
            structural_boost=0.0,
            name_boost=0.0,
        ),
        semantic_index=fake,  # type: ignore[arg-type]
    )
    hits = dispatcher.dispatch("completely unrelated query xyzzy")
    assert hits
    assert hits[0].chunk.chunk_id == target.chunk_id
    assert hits[0].semantic_score > 0


def test_unavailable_semantic_degrades(corpus_vault: CodeVault) -> None:
    fake = _FakeSemanticIndex("nope", available=False)
    dispatcher = IntentDispatcher(
        corpus_vault,
        config=DispatchConfig(top_k=2, semantic_weight=0.75),
        semantic_index=fake,  # type: ignore[arg-type]
    )
    hits = dispatcher.dispatch("async repository")
    assert hits
    assert all(h.semantic_score == 0.0 for h in hits)


def test_empty_query_falls_back_to_tag_richness(
    corpus_vault: CodeVault,
) -> None:
    """M5: stopword-only prompts still get a gold schedule."""
    dispatcher = IntentDispatcher(
        corpus_vault,
        config=DispatchConfig(top_k=3, semantic_weight=0.0),
    )
    hits = dispatcher.dispatch("the a an")
    assert len(hits) == 3
    assert all(h.lexical_score == 0.0 for h in hits)
    assert all(h.semantic_score == 0.0 for h in hits)
    # Tag-richness structural prior should be non-zero for annotated gold.
    assert any(h.structural_score > 0 for h in hits)
    totals = [h.total for h in hits]
    assert totals == sorted(totals, reverse=True)


def test_no_lexical_hits_falls_back(corpus_vault: CodeVault) -> None:
    dispatcher = IntentDispatcher(
        corpus_vault,
        config=DispatchConfig(top_k=2, semantic_weight=0.0),
    )
    # Nonsense tokens with no corpus overlap (avoid common words like "token").
    hits = dispatcher.dispatch("xyzzyplugh qqxxyyzz")
    assert len(hits) == 2
    assert all(h.lexical_score == 0.0 for h in hits)
