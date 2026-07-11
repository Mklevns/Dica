"""Module 3a: Local Semantic Index (async).

Optional embedding layer over the vault, backed by the local Ollama
``/api/embed`` endpoint (``nomic-embed-text`` by default). The transport
is now ``httpx.AsyncClient`` so index construction no longer serializes
one blocking HTTP round-trip per batch: all batches are dispatched with
``asyncio.gather`` behind a small semaphore.

Design constraints honoured here
--------------------------------
* **Vectors never live on ``CodeChunk``.** Chunks are frozen Pydantic
  models describing *code identity*; embeddings are *index state*. The
  side-table keyed by ``chunk_id`` keeps the vault swappable for LanceDB /
  Chroma exactly as its docstring promises.
* **Graceful degradation is a hard requirement.** Every network touch is
  wrapped: if Ollama is down, the embed model isn't pulled, or a query
  embed times out, the index reports ``available == False`` (or returns an
  empty score map) and the dispatcher silently falls back to
  lexical+structural scoring. Retrieval quality degrades; the pipeline
  never crashes. Under concurrent build, failure is now *per batch*: a
  failed batch is skipped, every successful batch is kept.
* **Async is the hot path.** ``PipelineEngine`` and
  ``IntentDispatcher.dispatch`` await :meth:`SemanticIndex.score` on the
  running event loop so embedding HTTP never freezes concurrent work.
  ``build_sync`` / ``score_sync`` remain for *synchronous* call sites only
  (engine construction, scripts, tests): they use :func:`run_coro_blocking`
  (``asyncio.run`` when no loop is running, or a worker thread when one is).
  Do **not** call those shims from code already on the asyncio loop.

Concurrency honesty note: a single-GPU Ollama instance largely serializes
embedding forward passes, so the win here is eliminating per-request HTTP
setup/teardown and Python-side blocking — startup goes from
``sum(batch_latencies)`` on the main thread to roughly
``max(server queue drain)`` off it — not a magic N× GPU speedup.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
from collections.abc import Coroutine, Iterable
from typing import Any, TypeVar

import httpx

from dica.config import OllamaConfig
from dica.vault import CodeChunk

logger = logging.getLogger(__name__)

# Cap the text embedded per chunk: nomic-embed-text's useful window is
# ~2k tokens and the head of a chunk (signature + docstring) carries most
# of the semantic signal anyway.
_EMBED_TEXT_CHARS = 1500
_BATCH_SIZE = 16
# Concurrent in-flight embed requests. Ollama queues internally; a small
# cap keeps us from stampeding the daemon while it also serves generation.
_MAX_CONCURRENT_BATCHES = 4

_T = TypeVar("_T")


def run_coro_blocking(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` to completion from a synchronous call site, safely.

    * No running loop (process startup, plain scripts): ``asyncio.run``.
    * Running loop present (sync code nested inside a coroutine, e.g.
      ``SemanticIndex.build_sync`` during engine construction if a loop is
      already active): run the coroutine on a dedicated worker thread with
      its own event loop and block on the future. Prefer ``await`` on the
      hot path instead of this shim so the caller's loop is not frozen.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain-Python cosine; corpora here are far too small to need numpy."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class OllamaEmbedder:
    """Thin, fail-soft async client for Ollama's ``/api/embed`` endpoint."""

    def __init__(self, config: OllamaConfig) -> None:
        self._url = f"{config.host.rstrip('/')}/api/embed"
        self._model = config.embed_model
        self._timeout = config.embed_timeout_seconds

    async def _post(
        self, client: httpx.AsyncClient, texts: list[str]
    ) -> list[list[float]] | None:
        """One embed round-trip on an existing client. ``None`` on failure."""
        try:
            response = await client.post(
                self._url, json={"model": self._model, "input": texts}
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError, OSError) as exc:
            logger.warning(
                "Embedding request failed (%s); degrading to lexical.", exc
            )
            return None
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            logger.warning("Malformed embedding response; degrading to lexical.")
            return None
        return embeddings

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embed one batch on a throwaway client. Returns ``None`` on failure.

        Convenience path for single calls (query-time scoring). Bulk index
        construction should use :meth:`embed_concurrent`, which shares one
        connection pool across all batches.
        """
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._post(client, texts)

    async def embed_concurrent(
        self, batches: list[list[str]]
    ) -> list[list[list[float]] | None]:
        """Embed many batches concurrently over one shared client.

        Returns one entry per input batch, positionally aligned: the
        batch's vectors, or ``None`` if that batch failed. Concurrency is
        capped by a semaphore so a large vault cannot stampede the daemon.
        """
        if not batches:
            return []
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_BATCHES)

        async def _guarded(
            client: httpx.AsyncClient, batch: list[str]
        ) -> list[list[float]] | None:
            async with semaphore:
                return await self._post(client, batch)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await asyncio.gather(
                *(_guarded(client, batch) for batch in batches)
            )


def _chunk_embed_text(chunk: CodeChunk) -> str:
    """Semantic surface of a chunk: name, docstring, then source head."""
    parts = [chunk.name.replace("_", " ")]
    if chunk.docstring:
        parts.append(chunk.docstring)
    parts.append(chunk.source)
    return "\n".join(parts)[:_EMBED_TEXT_CHARS]


class SemanticIndex:
    """``chunk_id -> vector`` side-table with fail-soft query scoring."""

    def __init__(self, embedder: OllamaEmbedder) -> None:
        self._embedder = embedder
        self._vectors: dict[str, list[float]] = {}
        self._available = False

    @property
    def available(self) -> bool:
        """True when the index was built and query embedding is expected to work."""
        return self._available and bool(self._vectors)

    async def build(self, chunks: Iterable[CodeChunk]) -> int:
        """Embed every chunk, batches dispatched concurrently. Returns count.

        Failure is per batch: a failed batch is logged and skipped while
        every successful batch is kept. Partial semantic coverage still
        beats none — unindexed chunks simply score 0.0 semantically (their
        lexical score is unaffected).
        """
        chunk_list = list(chunks)
        batches = [
            chunk_list[i : i + _BATCH_SIZE]
            for i in range(0, len(chunk_list), _BATCH_SIZE)
        ]
        results = await self._embedder.embed_concurrent(
            [[_chunk_embed_text(c) for c in batch] for batch in batches]
        )

        indexed = 0
        failed_batches = 0
        for batch, vectors in zip(batches, results, strict=True):
            if vectors is None:
                failed_batches += 1
                continue
            for chunk, vector in zip(batch, vectors, strict=True):
                self._vectors[chunk.chunk_id] = vector
                indexed += 1
        if failed_batches:
            logger.warning(
                "Semantic index build: %d/%d batch(es) failed; keeping the "
                "%d chunks that embedded successfully.",
                failed_batches,
                len(batches),
                indexed,
            )
        self._available = indexed > 0
        logger.info(
            "Semantic index: %d/%d chunks embedded (available=%s).",
            indexed,
            len(chunk_list),
            self._available,
        )
        return indexed

    def build_sync(self, chunks: Iterable[CodeChunk]) -> int:
        """Blocking wrapper over :meth:`build` for synchronous call sites."""
        return run_coro_blocking(self.build(chunks))

    async def score(
        self, query: str, chunk_ids: Iterable[str]
    ) -> dict[str, float]:
        """Cosine similarity of ``query`` against each id, clamped to [0, 1].

        Returns an *empty* map on any failure — the dispatcher treats a
        missing id as semantic score 0.0, so degradation is automatic.
        """
        if not self.available:
            return {}
        vectors = await self._embedder.embed([query])
        if vectors is None:
            return {}
        query_vec = vectors[0]
        scores: dict[str, float] = {}
        for chunk_id in chunk_ids:
            vector = self._vectors.get(chunk_id)
            if vector is not None:
                scores[chunk_id] = max(0.0, cosine_similarity(query_vec, vector))
        return scores

    def score_sync(
        self, query: str, chunk_ids: Iterable[str]
    ) -> dict[str, float]:
        """Blocking wrapper over :meth:`score` for synchronous call sites only.

        Prefer :meth:`score` (``await``) from any coroutine — including
        :meth:`dica.dispatcher.IntentDispatcher.dispatch` and
        :meth:`dica.pipeline.PipelineEngine.run`. Using this wrapper on the
        asyncio event loop blocks the loop for the full embed round-trip.
        """
        return run_coro_blocking(self.score(query, chunk_ids))
