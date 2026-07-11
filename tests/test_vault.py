"""Tests for AST vault ingestion and keyword retrieval."""

from __future__ import annotations

from pathlib import Path

from dica.vault import ChunkKind, CodeVault, tokenize


def test_tokenize_snake_and_camel() -> None:
    tokens = tokenize("AsyncCRUDRouter build_user_index")
    assert "async" in tokens
    assert "crud" in tokens
    assert "router" in tokens
    assert "build" in tokens
    assert "user" in tokens
    assert "index" in tokens


def test_ingest_reference_corpus(corpus_vault: CodeVault) -> None:
    assert len(corpus_vault) > 0
    names = {c.name for c in corpus_vault}
    # Known symbols from seed corpus
    assert names & {"create_item", "list_items", "AsyncUserRepository"}


def test_ingest_attaches_decorators(tmp_path: Path) -> None:
    src = tmp_path / "mod.py"
    src.write_text(
        "@router.get('/x')\n"
        "async def fetch_thing(id: int) -> str:\n"
        "    '''Doc.'''\n"
        "    return str(id)\n",
        encoding="utf-8",
    )
    vault = CodeVault()
    assert vault.ingest_path(src) == 1
    chunk = next(iter(vault))
    assert chunk.kind == ChunkKind.ASYNC_FUNCTION
    assert "@router.get" in chunk.source
    assert chunk.tags.is_async is True
    assert chunk.tags.has_decorators is True
    assert "fetch" in chunk.keywords or "thing" in chunk.keywords


def test_search_finds_keyword_overlap(corpus_vault: CodeVault) -> None:
    hits = corpus_vault.search({"async", "repository"})
    assert hits
    assert all(0.0 <= score <= 1.0 for _, score in hits)
    # Higher overlap should not be negative
    hits.sort(key=lambda p: p[1], reverse=True)
    assert hits[0][1] >= hits[-1][1]


def test_jaccard_penalizes_broad_keyword_bags(tmp_path: Path) -> None:
    """M4: score is |∩| / |∪|, so huge bags matching the query score lower."""
    src = tmp_path / "bags.py"
    # Narrow chunk: keywords ≈ {match}
    # Broad chunk: many extra tokens → larger union → lower Jaccard for same overlap.
    src.write_text(
        "def match_only():\n"
        "    return 1\n"
        "\n"
        "def match_with_extra_alpha_beta_gamma_delta_epsilon():\n"
        "    alpha = beta = gamma = delta = epsilon = 0\n"
        "    return match_only() + alpha + beta + gamma + delta + epsilon\n",
        encoding="utf-8",
    )
    vault = CodeVault()
    vault.ingest_path(src)
    hits = {chunk.name: score for chunk, score in vault.search({"match"})}
    assert "match_only" in hits
    assert "match_with_extra_alpha_beta_gamma_delta_epsilon" in hits
    assert hits["match_only"] > hits["match_with_extra_alpha_beta_gamma_delta_epsilon"]
    # query={match}, match_only keywords ⊇ {match, only} → |∩|/|∪| = 1/2
    assert hits["match_only"] == 0.5


def test_search_empty_query_returns_no_lexical_hits(corpus_vault: CodeVault) -> None:
    assert corpus_vault.search([]) == []
    assert corpus_vault.search(set()) == []


def test_skips_syntax_error_file(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    good.write_text("def ok():\n    return 1\n", encoding="utf-8")
    bad.write_text("def broken(\n", encoding="utf-8")
    vault = CodeVault()
    n = vault.ingest_path(tmp_path)
    assert n == 1
    assert next(iter(vault)).name == "ok"
