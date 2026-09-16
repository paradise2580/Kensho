"""ChunkStore tests, run entirely in-memory against FakeEmbedder.

These check the plumbing — upsert, count, filtering, dimension guarding — not
retrieval quality, which needs the real embedder and is out of scope until an
environment with Hugging Face Hub access runs the eval harness (phase 2).
"""

from __future__ import annotations

import pytest

from kensho.embed import FakeEmbedder
from kensho.schema import Chunk
from kensho.store import ChunkStore


def make_chunk(text: str, lang: str = "en", ordinal: int = 0, title: str = "T") -> Chunk:
    doc_id = f"{lang}:doc"
    return Chunk(
        chunk_id=Chunk.make_id(doc_id, ordinal, "structural"),
        doc_id=doc_id,
        parallel_id="doc",
        lang=lang,
        title=title,
        section_path=("Intro",),
        text=text,
        token_count=len(text.split()),
        char_count=len(text),
        ordinal=ordinal,
        url="https://example.invalid/doc",
        strategy="structural",
        commit_sha="deadbeef",
    )


@pytest.fixture
def store() -> ChunkStore:
    return ChunkStore(FakeEmbedder(dim=32), path=None)


class TestUpsertAndSearch:
    def test_upsert_reports_count_written(self, store):
        chunks = [make_chunk("A Pod can hold several containers.", ordinal=0),
                  make_chunk("Quarterly revenue rose sharply in Tokyo.", ordinal=1)]
        assert store.upsert(chunks) == 2
        assert store.count() == 2

    def test_search_returns_closest_match_first(self, store):
        pod_chunk = make_chunk("A Pod can hold several containers.", ordinal=0)
        revenue_chunk = make_chunk("Quarterly revenue rose sharply in Tokyo.", ordinal=1)
        store.upsert([pod_chunk, revenue_chunk])
        hits = store.search("A Pod can hold several containers", top_k=2)
        assert hits[0].chunk_id == pod_chunk.chunk_id
        assert hits[0].score >= hits[1].score

    def test_upsert_is_idempotent_on_same_chunk_id(self, store):
        c = make_chunk("A Pod can hold several containers.", ordinal=0)
        store.upsert([c])
        store.upsert([c])
        assert store.count() == 1

    def test_search_respects_lang_filter(self, store):
        en = make_chunk("A Pod can hold several containers.", lang="en", ordinal=0)
        ja = make_chunk("ポッドは複数のコンテナを持てます", lang="ja", ordinal=1)
        store.upsert([en, ja])
        hits = store.search("A Pod can hold several containers", top_k=5, lang="ja")
        assert all(h.payload["lang"] == "ja" for h in hits)
        assert hits[0].chunk_id == ja.chunk_id

    def test_payload_carries_fields_needed_for_citation(self, store):
        c = make_chunk("A Pod can hold several containers.", ordinal=0, title="Pods")
        store.upsert([c])
        hit = store.search("A Pod can hold several containers", top_k=1)[0]
        assert hit.payload["title"] == "Pods"
        assert hit.payload["url"] == c.url
        assert hit.payload["section_path"] == ["Intro"]


class TestDimensionGuard:
    def test_mismatched_embedder_dim_on_reopen_raises(self, tmp_path):
        path = tmp_path / "store"
        ChunkStore(FakeEmbedder(dim=32), path=path)
        with pytest.raises(ValueError, match="dim=32"):
            ChunkStore(FakeEmbedder(dim=64), path=path)
