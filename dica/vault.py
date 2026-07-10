"""Module 1: Code Vault Ingestor.

Parses high-quality local Python files with the built-in ``ast`` module and
indexes top-level functions / classes as :class:`CodeChunk` records inside an
in-memory vault.

Design notes
------------
* **Why AST instead of regex/text-splitting?** The AST gives us *structural*
  chunk boundaries (a whole function or class, never half of one), plus free
  metadata: decorator names, base classes, whether a def is ``async``, etc.
  This is what makes the downstream "structural relevance" scoring in the
  dispatcher possible.
* **Storage.** For this MVP the vault is an in-memory dictionary keyed by
  chunk id. The public surface (``ingest_path`` / ``search``) is deliberately
  shaped like a vector-DB collection so it can be swapped for LanceDB or
  Chroma later without touching callers — replace ``_keyword_score`` with a
  cosine similarity over embeddings and you're done.
* **Keyword extraction.** We build a bag of lowercase tokens from the chunk's
  name (snake_case / CamelCase split), its docstring, decorator names, base
  classes, and every attribute/name referenced in the body. That token bag is
  the retrieval index.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Tokens that carry no retrieval signal.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
        "is", "are", "be", "this", "that", "it", "as", "by", "from", "self",
        "cls", "return", "returns", "args", "arg", "kwargs", "none", "true",
        "false", "def", "class", "if", "else", "raise", "raises", "type",
    }
)

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Names that signal Pydantic usage when they appear as bases or references.
_PYDANTIC_MARKERS: frozenset[str] = frozenset(
    {
        "BaseModel", "pydantic", "Field",
        "field_validator", "model_validator", "RootModel",
    }
)


class ChunkKind(StrEnum):
    """Structural category of an indexed AST node."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"


class ChunkTags(BaseModel):
    """Boolean structural metadata derived from the AST node.

    These tags are what the dispatcher uses for *structural* matching, as
    opposed to plain keyword overlap.
    """

    model_config = ConfigDict(frozen=True)

    is_async: bool = False
    is_class: bool = False
    has_pydantic: bool = False
    has_decorators: bool = False
    has_docstring: bool = False
    uses_typing: bool = False  # annotations present on signature / attributes


class CodeChunk(BaseModel):
    """A single indexed unit of gold-standard code."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    name: str
    kind: ChunkKind
    source: str = Field(description="Exact source segment for this node.")
    docstring: str | None = None
    file_path: str
    lineno: int
    tags: ChunkTags
    keywords: frozenset[str] = Field(
        description="Lowercased token bag used for retrieval scoring."
    )


def _split_identifier(identifier: str) -> Iterator[str]:
    """Yield lowercase word tokens from ``snake_case`` or ``CamelCase`` names.

    ``AsyncCRUDRouter`` -> ``async``, ``crud``, ``router``
    ``build_user_index`` -> ``build``, ``user``, ``index``
    """
    for part in identifier.split("_"):
        for token in _CAMEL_RE.split(part):
            token = token.lower().strip()
            if len(token) > 1 and token not in _STOPWORDS:
                yield token


def tokenize(text: str) -> frozenset[str]:
    """Extract the retrieval token bag from arbitrary text (docstrings, prompts)."""
    tokens: set[str] = set()
    for word in _WORD_RE.findall(text):
        tokens.update(_split_identifier(word))
    return frozenset(tokens)


def _collect_referenced_names(node: ast.AST) -> Iterator[str]:
    """Walk a node's body and yield every Name/Attribute identifier.

    This lets a function that *calls* ``asyncio.gather`` or subclasses
    ``BaseModel`` be retrievable via the keywords ``asyncio`` / ``gather`` /
    ``basemodel`` even if its own name and docstring never mention them.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            yield child.id
        elif isinstance(child, ast.Attribute):
            yield child.attr


def _has_pydantic(node: ast.AST) -> bool:
    """Heuristic: does this node reference any Pydantic marker name?"""
    return any(name in _PYDANTIC_MARKERS for name in _collect_referenced_names(node))


def _has_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> bool:
    """True if the signature (or class body) carries type annotations."""
    if isinstance(node, ast.ClassDef):
        return any(isinstance(stmt, ast.AnnAssign) for stmt in node.body)
    args = node.args
    annotated = any(
        a.annotation is not None
        for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)
    )
    return annotated or node.returns is not None


class CodeVault:
    """In-memory index of gold-standard code chunks.

    The interface intentionally mirrors a vector-DB collection:
    ``ingest_path`` ≈ upsert, ``search`` ≈ query. Swap the internals for
    LanceDB/Chroma when the corpus outgrows RAM.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, CodeChunk] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    def __iter__(self) -> Iterator[CodeChunk]:
        return iter(self._chunks.values())

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_path(self, path: Path) -> int:
        """Ingest a ``.py`` file or recursively ingest a directory.

        Returns the number of chunks added. Files that fail to parse are
        logged and skipped — a broken reference file must never take down
        the pipeline.
        """
        count = 0
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            try:
                count += self._ingest_file(file)
            except (SyntaxError, OSError, UnicodeDecodeError) as exc:
                logger.warning("Skipping %s: %s", file, exc)
        return count

    def _ingest_file(self, file: Path) -> int:
        source = file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file))
        count = 0
        # Only index *top-level* defs and classes: they are self-contained
        # units. Methods travel with their class, nested helpers with their
        # parent — exactly the granularity we want to paste into a prompt.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunk = self._build_chunk(node, source, file)
                if chunk is not None:
                    self._chunks[chunk.chunk_id] = chunk
                    count += 1
        logger.info("Indexed %d chunks from %s", count, file)
        return count

    def _build_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        source: str,
        file: Path,
    ) -> CodeChunk | None:
        # get_source_segment recovers the exact original text (including
        # decorators via padded=False on the node itself; we prepend
        # decorator lines manually below so formatting survives verbatim).
        segment = ast.get_source_segment(source, node)
        if segment is None:  # pragma: no cover — only on pathological input
            return None

        # Re-attach decorator lines: get_source_segment on the def node
        # excludes them, but decorators are often the most informative part
        # of a gold snippet (e.g. @router.post, @field_validator).
        if node.decorator_list:
            first_dec = node.decorator_list[0]
            dec_segment = "\n".join(
                source.splitlines()[first_dec.lineno - 1 : node.lineno - 1]
            )
            segment = f"{dec_segment}\n{segment}"

        docstring = ast.get_docstring(node)
        is_class = isinstance(node, ast.ClassDef)
        # A class whose methods are async (e.g. an async repository) should
        # still surface for "async" queries — so scan descendants too.
        is_async = isinstance(node, ast.AsyncFunctionDef) or any(
            isinstance(child, ast.AsyncFunctionDef) for child in ast.walk(node)
        )

        kind = (
            ChunkKind.CLASS
            if is_class
            else ChunkKind.ASYNC_FUNCTION
            if isinstance(node, ast.AsyncFunctionDef)
            else ChunkKind.FUNCTION
        )

        tags = ChunkTags(
            is_async=is_async,
            is_class=is_class,
            has_pydantic=_has_pydantic(node),
            has_decorators=bool(node.decorator_list),
            has_docstring=docstring is not None,
            uses_typing=_has_annotations(node),
        )

        # Keyword bag: name + docstring + every referenced identifier.
        keywords: set[str] = set(_split_identifier(node.name))
        if docstring:
            keywords |= tokenize(docstring)
        for ref in _collect_referenced_names(node):
            keywords.update(_split_identifier(ref))

        chunk_id = hashlib.sha1(
            f"{file}:{node.name}:{node.lineno}".encode()
        ).hexdigest()[:12]

        return CodeChunk(
            chunk_id=chunk_id,
            name=node.name,
            kind=kind,
            source=segment,
            docstring=docstring,
            file_path=str(file),
            lineno=node.lineno,
            tags=tags,
            keywords=frozenset(keywords),
        )

    # ------------------------------------------------------------------ #
    # Retrieval primitive (scoring policy lives in the dispatcher)
    # ------------------------------------------------------------------ #
    def search(
        self,
        keywords: Iterable[str],
        *,
        required_tags: dict[str, bool] | None = None,
    ) -> list[tuple[CodeChunk, float]]:
        """Return ``(chunk, keyword_overlap_score)`` pairs, unranked-filtered.

        ``required_tags`` performs hard filtering (e.g. ``{"is_async": True}``)
        before scoring; soft/boosted ranking is the dispatcher's job.
        """
        query = frozenset(k.lower() for k in keywords)
        results: list[tuple[CodeChunk, float]] = []
        for chunk in self._chunks.values():
            if required_tags and any(
                getattr(chunk.tags, tag) != value
                for tag, value in required_tags.items()
            ):
                continue
            overlap = query & chunk.keywords
            if not overlap and required_tags is None:
                continue
            # Jaccard-flavoured: reward overlap, mildly penalise huge chunks
            # matching everything.
            score = len(overlap) / (len(query) or 1)
            results.append((chunk, score))
        return results
