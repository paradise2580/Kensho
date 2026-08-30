from __future__ import annotations

from kensho.retrieval.hybrid import HybridRetriever
from kensho.store import SearchHit


def hit(chunk_id: str, score: float) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, score=score, payload={"parallel_id": chunk_id})


class _StubRetriever:
    def __init__(self, hits: list[SearchHit], name: str) -> None:
        self._hits = hits
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]:
        return self._hits[:top_k]


class TestHybridRetriever:
    def test_document_found_by_both_retrievers_ranks_first(self):
        dense = _StubRetriever([hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)], "dense")
        sparse = _StubRetriever([hit("b", 12.0), hit("a", 8.0), hit("d", 1.0)], "sparse")
        hybrid = HybridRetriever(dense, sparse, fetch_k=10)
        out = hybrid.search("q", top_k=4)
        ids = [h.chunk_id for h in out]
        # "a" and "b" appear near the top of both lists and should fuse above
        # "c"/"d", which only one retriever found at all.
        assert set(ids[:2]) == {"a", "b"}

    def test_document_from_only_one_retriever_still_appears(self):
        dense = _StubRetriever([hit("a", 0.9)], "dense")
        sparse = _StubRetriever([hit("b", 5.0)], "sparse")
        hybrid = HybridRetriever(dense, sparse, fetch_k=10)
        out = hybrid.search("q", top_k=5)
        assert {h.chunk_id for h in out} == {"a", "b"}

    def test_score_scale_mismatch_does_not_break_fusion(self):
        """Sparse scores are on a totally different scale (BM25, unbounded)
        from dense cosine scores (roughly [-1, 1]) — RRF must not let the
        larger numbers dominate just because they're numerically bigger."""
        dense = _StubRetriever([hit("a", 0.99)], "dense")  # best dense match
        sparse = _StubRetriever([hit("z", 999.0), hit("a", 1.0)], "sparse")
        hybrid = HybridRetriever(dense, sparse, fetch_k=10)
        out = hybrid.search("q", top_k=2)
        # "a" is rank 1 for dense and rank 2 for sparse; "z" is rank 1 for
        # sparse only. RRF should not automatically put "z" first just
        # because its raw BM25 score of 999 dwarfs everything else.
        assert out[0].chunk_id == "a"

    def test_empty_from_both_retrievers(self):
        hybrid = HybridRetriever(_StubRetriever([], "dense"), _StubRetriever([], "sparse"))
        assert hybrid.search("q", top_k=5) == []

    def test_name_reflects_both_components(self):
        hybrid = HybridRetriever(_StubRetriever([], "dense:x"), _StubRetriever([], "bm25:y"))
        assert hybrid.name == "hybrid-rrf(dense:x+bm25:y)"

    def test_respects_top_k(self):
        dense = _StubRetriever([hit(str(i), 1.0 / (i + 1)) for i in range(10)], "dense")
        sparse = _StubRetriever([], "sparse")
        hybrid = HybridRetriever(dense, sparse, fetch_k=10)
        assert len(hybrid.search("q", top_k=3)) == 3
