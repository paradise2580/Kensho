"""BM25Store tests, run against the real SudachiPy segmenter — these check
genuine lexical retrieval behaviour, not just plumbing, because unlike the
embedder, sparse retrieval needs no network in this sandbox."""

from __future__ import annotations

import importlib.util

import pytest

from kensho.retrieval.sparse import BM25Store
from kensho.schema import Chunk

requires_sudachi = pytest.mark.skipif(
    importlib.util.find_spec("sudachipy") is None,
    reason="sparse extra not installed — see pyproject.toml",
)


def make_chunk(text: str, parallel_id: str, lang: str, ordinal: int = 0) -> Chunk:
    doc_id = f"{lang}:{parallel_id}"
    return Chunk(
        chunk_id=Chunk.make_id(doc_id, ordinal, "structural"),
        doc_id=doc_id, parallel_id=parallel_id, lang=lang, title=parallel_id,
        section_path=(), text=text, token_count=len(text.split()), char_count=len(text),
        ordinal=ordinal, url=f"https://example.invalid/{parallel_id}",
        strategy="structural", commit_sha="deadbeef",
    )


class TestBM25StoreEnglish:
    def test_exact_term_match_ranks_first(self):
        store = BM25Store()
        store.index([
            make_chunk("A Pod can hold several containers.", "docs/pods.md", "en"),
            make_chunk("Quarterly revenue rose sharply in Tokyo.", "docs/unrelated.md", "en"),
        ])
        hits = store.search("Pod containers", top_k=2)
        assert hits[0].payload["parallel_id"] == "docs/pods.md"

    def test_count(self):
        store = BM25Store()
        store.index([make_chunk("x", "docs/a.md", "en"), make_chunk("y", "docs/b.md", "en")])
        assert store.count() == 2

    def test_empty_index_returns_no_hits(self):
        store = BM25Store()
        store.index([])
        assert store.search("anything", top_k=5) == []

    def test_lang_filter_excludes_other_language(self):
        store = BM25Store()
        store.index([
            make_chunk("A Pod can hold several containers.", "docs/pods-en.md", "en"),
            make_chunk("A Pod can hold several containers extra copy.", "docs/pods-en2.md", "en"),
        ])
        hits = store.search("Pod containers", top_k=5, lang="ja")
        assert hits == []


@requires_sudachi
class TestBM25StoreJapanese:
    """Uses the real Sudachi segmenter: this is a genuine test of whether
    morphological segmentation makes Japanese BM25 work at all — the exact
    capability the project's README says whitespace tokenization lacks."""

    def test_real_japanese_query_retrieves_the_matching_chunk(self):
        store = BM25Store()
        store.index([
            make_chunk("ポッドは複数のコンテナを持てます", "docs/pods.md", "ja"),
            make_chunk("四半期の収益は東京で大きく伸びました", "docs/unrelated.md", "ja"),
        ])
        hits = store.search("コンテナ", top_k=2)
        assert hits[0].payload["parallel_id"] == "docs/pods.md"

    def test_cross_lingual_query_does_not_match(self):
        """REGRESSION guard against a false sense of cross-lingual capability:
        BM25 is lexical, so an English query must not match Japanese-only
        text by accident (e.g. via a shared character run scoring nonzero)."""
        store = BM25Store()
        store.index([make_chunk("ポッドは複数のコンテナを持てます", "docs/pods.md", "ja")])
        hits = store.search("A Pod can hold several containers", top_k=1)
        assert hits == [] or hits[0].score == 0.0
