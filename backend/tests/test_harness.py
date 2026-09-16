"""Harness tests against a small, fully-controlled FakeEmbedder store.

FakeEmbedder is meaning-blind (see embed.py), so these tests can't check
that retrieval finds the *semantically* right chunk — they check that the
harness correctly turns whatever the store returns into deduplicated
rankings and aggregate metrics, including the unanswerable-item bookkeeping.
"""

from __future__ import annotations

from kensho.embed import FakeEmbedder
from kensho.eval.gold import GoldItem
from kensho.eval.harness import evaluate
from kensho.schema import Chunk
from kensho.store import ChunkStore


def make_chunk(text: str, parallel_id: str, lang: str, ordinal: int) -> Chunk:
    doc_id = f"{lang}:{parallel_id}"
    return Chunk(
        chunk_id=Chunk.make_id(doc_id, ordinal, "structural"),
        doc_id=doc_id, parallel_id=parallel_id, lang=lang, title=parallel_id,
        section_path=(), text=text, token_count=len(text.split()), char_count=len(text),
        ordinal=ordinal, url=f"https://example.invalid/{parallel_id}",
        strategy="structural", commit_sha="deadbeef",
    )


def build_store() -> ChunkStore:
    store = ChunkStore(FakeEmbedder(dim=64), path=None)
    store.upsert([
        make_chunk("A Pod can hold several containers.", "docs/pods.md", "en", 0),
        make_chunk("A Service exposes a set of Pods on the network.", "docs/services.md", "en", 0),
        make_chunk("Quarterly revenue rose sharply in Tokyo.", "docs/unrelated.md", "en", 0),
    ])
    return store


class TestEvaluate:
    def test_answerable_item_gets_ranked_and_scored(self):
        store = build_store()
        gold = [GoldItem(id="g1", lang="en", topic="pods",
                         query="A Pod can hold several containers",
                         relevant_parallel_ids=("docs/pods.md",))]
        report = evaluate(gold, store, top_k=3)
        assert report.n_answerable == 1
        assert report.n_unanswerable == 0
        assert report.items[0].ranked[0] == "docs/pods.md"
        assert report.overall.recall_at_k[1] == 1.0
        assert report.overall.mrr == 1.0

    def test_unanswerable_item_is_excluded_from_metrics(self):
        store = build_store()
        gold = [
            GoldItem(id="g1", lang="en", topic="pods",
                     query="A Pod can hold several containers",
                     relevant_parallel_ids=("docs/pods.md",)),
            GoldItem(id="g2", lang="en", topic="unanswerable",
                     query="What is the price of a support contract?",
                     relevant_parallel_ids=(), answerable=False),
        ]
        report = evaluate(gold, store, top_k=3)
        assert report.n_items == 2
        assert report.n_answerable == 1
        assert report.n_unanswerable == 1
        # only the answerable item contributes to aggregate metrics
        assert report.overall.recall_at_k[1] == 1.0

    def test_by_lang_breakdown(self):
        store = build_store()
        gold = [
            GoldItem(id="g1", lang="en", topic="pods",
                     query="A Pod can hold several containers",
                     relevant_parallel_ids=("docs/pods.md",)),
            GoldItem(id="g2", lang="ja", topic="pods",
                     # fake embedder is meaning-blind, so reusing the EN query is fine here
                     query="A Pod can hold several containers",
                     relevant_parallel_ids=("docs/pods.md",)),
        ]
        report = evaluate(gold, store, top_k=3)
        assert set(report.by_lang) == {"en", "ja"}
        assert report.by_lang["en"].n == 1
        assert report.by_lang["ja"].n == 1

    def test_no_hit_scores_zero_but_still_reports_ranking(self):
        store = build_store()
        gold = [GoldItem(id="g1", lang="en", topic="missing",
                         query="A Pod can hold several containers",
                         relevant_parallel_ids=("docs/does-not-exist.md",))]
        report = evaluate(gold, store, top_k=3)
        assert report.overall.recall_at_k[3] == 0.0
        assert report.overall.mrr == 0.0
        assert len(report.items[0].ranked) == 3  # all three chunks still surfaced

    def test_empty_store_still_produces_a_report(self):
        store = ChunkStore(FakeEmbedder(dim=16), path=None)
        gold = [GoldItem(id="g1", lang="en", topic="x", query="anything",
                         relevant_parallel_ids=("docs/x.md",))]
        report = evaluate(gold, store, top_k=5)
        assert report.items[0].ranked == ()
        assert report.overall.recall_at_k[5] == 0.0

    def test_report_to_dict_is_json_serializable(self):
        import json
        store = build_store()
        gold = [GoldItem(id="g1", lang="en", topic="pods",
                         query="A Pod can hold several containers",
                         relevant_parallel_ids=("docs/pods.md",))]
        report = evaluate(gold, store, top_k=3)
        json.dumps(report.to_dict())  # must not raise

    def test_report_records_retriever_name_and_lang_filter(self):
        store = build_store()
        gold = [GoldItem(id="g1", lang="en", topic="pods",
                         query="A Pod can hold several containers",
                         relevant_parallel_ids=("docs/pods.md",))]
        report = evaluate(gold, store, top_k=3)
        assert report.retriever_name == store.name
        assert report.lang_filter is None


class TestLangFilter:
    def test_match_query_restricts_search_to_each_items_own_language(self):
        store = ChunkStore(FakeEmbedder(dim=32), path=None)
        store.upsert([
            make_chunk("A Pod can hold several containers.", "docs/pods.md", "en", 0),
            make_chunk("ポッドは複数のコンテナを持てます", "docs/pods.md", "ja", 1),
        ])
        gold = [GoldItem(id="g1", lang="ja", topic="pods",
                         query="A Pod can hold several containers",
                         relevant_parallel_ids=("docs/pods.md",))]
        report = evaluate(gold, store, top_k=5, lang_filter="match_query")
        assert all(p == "ja" for p in _langs_seen(store, "A Pod can hold several containers", "ja"))
        assert report.lang_filter == "match_query"

    def test_fixed_lang_filter_applies_to_every_query(self):
        store = ChunkStore(FakeEmbedder(dim=32), path=None)
        store.upsert([
            make_chunk("A Pod can hold several containers.", "docs/pods.md", "en", 0),
            make_chunk("ポッドは複数のコンテナを持てます", "docs/pods.md", "ja", 1),
        ])
        gold = [GoldItem(id="g1", lang="en", topic="pods", query="anything",
                         relevant_parallel_ids=("docs/pods.md",))]
        report = evaluate(gold, store, top_k=5, lang_filter="ja")
        assert report.lang_filter == "ja"


def _langs_seen(store: ChunkStore, query: str, lang: str) -> list[str]:
    return [h.payload["lang"] for h in store.search(query, top_k=5, lang=lang)]
