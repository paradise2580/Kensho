from __future__ import annotations

import pytest

from kensho.retrieval.rerank import FakeReranker, RerankingRetriever, get_reranker
from kensho.store import SearchHit


def hit(chunk_id: str, text: str, score: float = 0.0) -> SearchHit:
    payload = {"text": text, "parallel_id": chunk_id}
    return SearchHit(chunk_id=chunk_id, score=score, payload=payload)


class TestFakeReranker:
    def test_deterministic(self):
        hits = [hit("a", "alpha"), hit("b", "beta")]
        r = FakeReranker()
        assert r.rerank("q", hits, top_k=2) == r.rerank("q", hits, top_k=2)

    def test_truncates_to_top_k(self):
        hits = [hit("a", "alpha"), hit("b", "beta"), hit("c", "gamma")]
        out = FakeReranker().rerank("q", hits, top_k=2)
        assert len(out) == 2

    def test_empty_hits(self):
        assert FakeReranker().rerank("q", [], top_k=5) == []

    def test_scores_are_sorted_descending(self):
        hits = [hit("a", "alpha"), hit("b", "beta"), hit("c", "gamma"), hit("d", "delta")]
        out = FakeReranker().rerank("q", hits, top_k=4)
        scores = [h.score for h in out]
        assert scores == sorted(scores, reverse=True)


class TestGetReranker:
    def test_none_returns_fake_directly(self):
        assert get_reranker(model_name=None).name == "fake"

    def test_missing_dep_or_network_raises_by_default(self):
        with pytest.raises(RuntimeError, match="Refusing to silently substitute"):
            get_reranker("BAAI/bge-reranker-v2-m3")

    def test_allow_fallback_warns_and_returns_fake(self):
        with pytest.warns(RuntimeWarning, match="meaningless"):
            r = get_reranker("BAAI/bge-reranker-v2-m3", allow_fallback=True)
        assert r.name == "fake"


class _StubRetriever:
    """Minimal Retriever stub: returns a fixed, over-long candidate list so
    RerankingRetriever's fetch_k / top_k truncation can be tested in isolation."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits

    @property
    def name(self) -> str:
        return "stub"

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]:
        return self._hits[:top_k]


class TestRerankingRetriever:
    def test_over_fetches_then_truncates_to_top_k(self):
        hits = [hit(str(i), f"text {i}") for i in range(10)]
        retriever = RerankingRetriever(_StubRetriever(hits), FakeReranker(), fetch_k=8)
        out = retriever.search("q", top_k=3)
        assert len(out) == 3

    def test_name_reflects_base_and_reranker(self):
        retriever = RerankingRetriever(_StubRetriever([]), FakeReranker())
        assert retriever.name == "stub+rerank:fake"
