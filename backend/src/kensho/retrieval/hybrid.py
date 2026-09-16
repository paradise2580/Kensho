"""Reciprocal Rank Fusion of a dense and a sparse retriever.

Cosine similarity and a BM25 score are not on comparable scales, so summing
or averaging them directly would let whichever retriever happens to produce
larger numbers dominate the fusion regardless of which is actually more
relevant. RRF sidesteps the problem by fusing on *rank*, not raw score:

    score(d) = sum over retrievers of  1 / (k_rrf + rank_of(d))

A document that both retrievers rank highly scores well even if their raw
scores are on entirely different scales; a document only one retriever
found still contributes, just less. ``k_rrf = 60`` is the constant from the
original RRF paper (Cormack et al., 2009) — a reasonable default, not
something this project has tuned, and that's a caveat worth keeping honest
about in the write-up rather than presenting as a chosen value.
"""

from __future__ import annotations

from ..store import Retriever, SearchHit

DEFAULT_K_RRF = 60
DEFAULT_FETCH_K = 50


class HybridRetriever:
    """Implements ``Retriever`` by fusing two other ``Retriever``s."""

    def __init__(self, dense: Retriever, sparse: Retriever,
                k_rrf: int = DEFAULT_K_RRF, fetch_k: int = DEFAULT_FETCH_K) -> None:
        self._dense = dense
        self._sparse = sparse
        self._k_rrf = k_rrf
        self._fetch_k = fetch_k

    @property
    def name(self) -> str:
        return f"hybrid-rrf({self._dense.name}+{self._sparse.name})"

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]:
        dense_hits = self._dense.search(query, top_k=self._fetch_k, lang=lang)
        sparse_hits = self._sparse.search(query, top_k=self._fetch_k, lang=lang)

        fused_score: dict[str, float] = {}
        payload: dict[str, dict] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, h in enumerate(hits, start=1):
                contribution = 1.0 / (self._k_rrf + rank)
                fused_score[h.chunk_id] = fused_score.get(h.chunk_id, 0.0) + contribution
                payload.setdefault(h.chunk_id, h.payload)

        ranked_ids = sorted(fused_score, key=lambda cid: fused_score[cid], reverse=True)[:top_k]
        return [
            SearchHit(chunk_id=cid, score=fused_score[cid], payload=payload[cid])
            for cid in ranked_ids
        ]
