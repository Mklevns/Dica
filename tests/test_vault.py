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
    assert all(score >= 0 for _, score in hits)
    # Higher overlap should not be negative
    hits.sort(key=lambda p: p[1], reverse=True)
    assert hits[0][1] >= hits[-1][1]


def test_skips_syntax_error_file(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    good.write_text("def ok():\n    return 1\n", encoding="utf-8")
    bad.write_text("def broken(\n", encoding="utf-8")
    vault = CodeVault()
    n = vault.ingest_path(tmp_path)
    assert n == 1
    assert next(iter(vault)).name == "ok"
