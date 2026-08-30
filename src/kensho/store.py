"""Vector storage — a thin Qdrant wrapper, embedded mode by default.

Qdrant's "local mode" (``QdrantClient(path=...)``) runs the full engine
in-process against an on-disk store: no server, no Docker, no network. That
is enough for this project's corpus size (tens of thousands of chunks) and it
means the same code path works in development, in CI, and in the demo
deployment — there is no separate vector-DB service to keep alive or pay for.

Phase 1 scope is dense-only search. Payload filtering (by ``lang``, by
``parallel_id``) is included now because the cross-lingual ablations in phase
3 need it and retrofitting filters onto an existing collection is more work
than including the fields from the start.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .embed import Embedder
from .schema import Chunk

COLLECTION = "kensho_chunks"


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class Retriever(Protocol):
    """The shape every retrieval component in phase 3 conforms to.

    ``ChunkStore`` (dense), ``BM25Store`` (sparse), ``HybridRetriever``, and
    ``RerankingRetriever`` all implement this, so the eval harness — and
    ``pipeline.ask`` — can run against any of them without knowing which one
    they got. That's what makes an ablation a one-line swap instead of a
    separate code path per configuration.
    """

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]: ...

    @property
    def name(self) -> str: ...


class ChunkStore:
    """Owns one Qdrant collection sized for a given embedder's vector dim."""

    def __init__(self, embedder: Embedder, path: str | Path | None = "data/qdrant") -> None:
        self._embedder = embedder
        # path=None gives an in-memory instance, used by tests so nothing
        # touches disk and repeated runs never collide on a stale lockfile.
        self._client = QdrantClient(path=str(path) if path is not None else ":memory:")
        self._ensure_collection()

    @property
    def name(self) -> str:
        return f"dense:{self._embedder.name}"

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(COLLECTION):
            existing = self._client.get_collection(COLLECTION).config.params.vectors.size
            if existing != self._embedder.dim:
                raise ValueError(
                    f"Collection {COLLECTION!r} was built with dim={existing}, but "
                    f"embedder {self._embedder.name!r} produces dim={self._embedder.dim}. "
                    "Delete the store or point at a fresh path before switching embedders."
                )
            return
        self._client.create_collection(
            COLLECTION,
            vectors_config=qm.VectorParams(size=self._embedder.dim, distance=qm.Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Embed and store chunks. Returns the number written."""
        n = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = self._embedder.embed_passages([c.for_embedding() for c in batch])
            points = [
                qm.PointStruct(
                    id=_qdrant_id(c.chunk_id),
                    vector=vec,
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "parallel_id": c.parallel_id,
                        "lang": c.lang,
                        "title": c.title,
                        "section_path": list(c.section_path),
                        "text": c.text,
                        "url": c.url,
                        "strategy": c.strategy,
                    },
                )
                for c, vec in zip(batch, vectors, strict=True)
            ]
            self._client.upsert(COLLECTION, points=points)
            n += len(points)
        return n

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]:
        query_filter = None
        if lang is not None:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="lang", match=qm.MatchValue(value=lang))]
            )
        vec = self._embedder.embed_query(query)
        results = self._client.query_points(
            COLLECTION, query=vec, limit=top_k, query_filter=query_filter,
        ).points
        return [SearchHit(chunk_id=r.payload["chunk_id"], score=r.score, payload=r.payload)
                for r in results]

    def count(self) -> int:
        return self._client.count(COLLECTION).count


def _qdrant_id(chunk_id: str) -> int:
    """Qdrant point ids must be an unsigned int or a UUID; our chunk_ids are
    16 hex chars from a sha1 digest, so truncating to 63 bits keeps them
    stable, deterministic, and collision-safe for any corpus this size."""
    return int(chunk_id, 16) & 0x7FFF_FFFF_FFFF_FFFF
