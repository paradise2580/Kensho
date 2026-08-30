"""Cross-encoder reranking, with the same pluggable-backend shape as
embed.py and llm.py.

A reranker scores a (query, passage) pair jointly, rather than comparing two
independently-computed vectors — strictly more expressive than cosine
similarity, at the cost of being too slow to run over the whole corpus. The
standard pattern, used here, is to over-fetch with a cheap retriever and
rerank only the candidates.

``CrossEncoderReranker`` — bge-reranker-v2-m3 via sentence-transformers'
                          CrossEncoder. The real thing; needs the model
                          weights, so it needs Hugging Face Hub access.

``FakeReranker``        — deterministic hash-based scores, meaning-blind,
                          same philosophy as ``FakeEmbedder``: it exists so
                          the over-fetch/score/truncate plumbing can be
                          built and tested without model access, and it
                          must never be mistaken for a real result.
                          ``get_reranker`` defaults to ``allow_fallback=False``
                          for the same reason ``get_embedder`` does — a
                          silently meaning-blind rerank is a correctness
                          gap, not a degraded-but-honest approximation.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import replace
from typing import Protocol

from ..store import Retriever, SearchHit

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...

    @property
    def name(self) -> str: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import CrossEncoder  # heavy; lazy import

        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    @property
    def name(self) -> str:
        return f"ce:{self._model_name}"

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, h.payload.get("text", "")) for h in hits]
        scores = self._model.predict(pairs)
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        return [replace(hits[i], score=float(scores[i])) for i in order[:top_k]]


class FakeReranker:
    """Deterministic, offline, meaning-blind. For plumbing tests only."""

    @property
    def name(self) -> str:
        return "fake"

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        def score(h: SearchHit) -> float:
            text = h.payload.get("text", "")
            digest = hashlib.blake2b(f"{query}:{text}".encode(), digest_size=8).digest()
            return int.from_bytes(digest, "big") / 2**64

        ranked = sorted(hits, key=score, reverse=True)
        return [replace(h, score=score(h)) for h in ranked[:top_k]]


def get_reranker(model_name: str | None = DEFAULT_MODEL, allow_fallback: bool = False) -> Reranker:
    """Return the real reranker, or the offline stand-in if explicitly allowed.

    Mirrors ``get_embedder``: defaults to *not* falling back, because a
    reranker that silently reorders by noise instead of relevance is a
    correctness gap that looks, from the outside, exactly like a working
    reranker that happens to disagree with you.
    """
    if model_name is None:
        return FakeReranker()
    try:
        return CrossEncoderReranker(model_name)
    except Exception as exc:  # noqa: BLE001 - missing dep, no network, bad name
        if not allow_fallback:
            raise RuntimeError(
                f"Could not load reranker {model_name!r} ({exc.__class__.__name__}: {exc}). "
                "Refusing to silently substitute a fake reranker — pass "
                "allow_fallback=True only for plumbing tests."
            ) from exc
        warnings.warn(
            f"Falling back to FakeReranker ({exc.__class__.__name__}). "
            "Rankings from this run are meaningless and must not be reported.",
            RuntimeWarning,
            stacklevel=2,
        )
        return FakeReranker()


class RerankingRetriever:
    """Wraps any ``Retriever``: over-fetch, then rerank down to ``top_k``.

    Implements the ``Retriever`` protocol itself, so it drops into the eval
    harness exactly like a plain ``ChunkStore`` — ablation 3.6 is this class
    wrapping the phase-1 dense store, nothing more.
    """

    def __init__(self, base: Retriever, reranker: Reranker, fetch_k: int = 20) -> None:
        self._base = base
        self._reranker = reranker
        self._fetch_k = fetch_k

    @property
    def name(self) -> str:
        return f"{self._base.name}+rerank:{self._reranker.name}"

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]:
        candidates = self._base.search(query, top_k=max(top_k, self._fetch_k), lang=lang)
        return self._reranker.rerank(query, candidates, top_k)
