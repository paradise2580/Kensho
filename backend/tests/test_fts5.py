"""FTS5Store tests, run against the real SudachiPy segmenter.

Deliberately the same test cases as test_sparse.py's BM25Store suite where
they apply — same segmenter, same claimed capability, so the same claims
need the same evidence. The tests unique to this module are the ones with
no BM25Store analogue: persistence across a fresh connection (the entire
point of this backend) and staleness detection.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from kensho.retrieval.fts5 import FTS5Store, build_or_load, load_chunks
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
        section_path=("A", "B"), text=text, token_count=len(text.split()),
        char_count=len(text), ordinal=ordinal,
        url=f"https://example.invalid/{parallel_id}",
        strategy="structural", commit_sha="deadbeef",
    )


def stub_source(tmp_path):
    """A placeholder source file for tests that only care about search
    behaviour, not staleness tracking — build() records the source file's
    size, so it has to actually exist."""
    p = tmp_path / "src.jsonl"
    p.touch()
    return p


def write_chunks_jsonl(path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            d = {
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "parallel_id": c.parallel_id,
                "lang": c.lang, "title": c.title, "section_path": list(c.section_path),
                "text": c.text, "token_count": c.token_count, "char_count": c.char_count,
                "ordinal": c.ordinal, "url": c.url, "strategy": c.strategy,
                "commit_sha": c.commit_sha, "meta": c.meta,
            }
            fh.write(json.dumps(d) + "\n")


class TestFTS5StoreEnglish:
    def test_exact_term_match_ranks_first(self, tmp_path):
        store = FTS5Store(tmp_path / "idx.db")
        store.build([
            make_chunk("A Pod can hold several containers.", "docs/pods.md", "en"),
            make_chunk("Quarterly revenue rose sharply in Tokyo.", "docs/unrelated.md", "en"),
        ], source_path=stub_source(tmp_path))
        hits = store.search("Pod containers", top_k=2)
        assert hits[0].payload["parallel_id"] == "docs/pods.md"

    def test_count(self, tmp_path):
        store = FTS5Store(tmp_path / "idx.db")
        store.build(
            [make_chunk("x", "docs/a.md", "en"), make_chunk("y", "docs/b.md", "en")],
            source_path=stub_source(tmp_path),
        )
        assert store.count() == 2

    def test_empty_index_returns_no_hits(self, tmp_path):
        store = FTS5Store(tmp_path / "idx.db")
        store.build([], source_path=stub_source(tmp_path))
        assert store.search("anything", top_k=5) == []

    def test_lang_filter_excludes_other_language(self, tmp_path):
        store = FTS5Store(tmp_path / "idx.db")
        store.build([
            make_chunk("A Pod can hold several containers.", "docs/pods-en.md", "en"),
            make_chunk("A Pod can hold several containers extra copy.", "docs/pods-en2.md", "en"),
        ], source_path=stub_source(tmp_path))
        hits = store.search("Pod containers", top_k=5, lang="ja")
        assert hits == []

    def test_score_is_higher_is_better(self, tmp_path):
        """Every other Retriever in this project (BM25Store, ChunkStore,
        HybridRetriever) is higher-is-better, and this store's scores are
        computed with the same Okapi formula BM25Store uses (see
        fts5.py's module docstring), so they share that convention too —
        this needs a THIRD, neutral document to actually test it. With
        only two documents and "Pod" in both, idf(Pod) goes negative
        (rank_bm25 gives -0.040 on that exact fixture) and BM25 legitimately
        scores the higher-term-frequency document *lower* — verified
        against rank_bm25.BM25Okapi directly on that fixture, not a bug in
        either implementation, just Okapi's epsilon-floor behaviour on a
        near-degenerate corpus. Adding a document that never mentions "Pod"
        pulls idf(Pod) positive and restores the "more mentions ranks
        higher" intuition this test is actually about."""
        store = FTS5Store(tmp_path / "idx.db")
        store.build([
            make_chunk("Pod Pod Pod containers everywhere in this Pod text.",
                       "docs/strong.md", "en"),
            make_chunk("Pod appears exactly once here.", "docs/weak.md", "en"),
            make_chunk("Quarterly revenue rose sharply in Tokyo this year.",
                       "docs/unrelated.md", "en"),
        ], source_path=stub_source(tmp_path))
        hits = store.search("Pod", top_k=3)
        by_id = {h.payload["parallel_id"]: h.score for h in hits}
        assert by_id["docs/strong.md"] > by_id["docs/weak.md"] > 0

    def test_matches_rank_bm25_okapi_exactly(self, tmp_path):
        """The point of reimplementing scoring instead of using FTS5's own
        bm25(): this store's ranking function is exact-formula Okapi BM25
        with this project's k1=1.5/b=0.75/epsilon=0.25 (matching
        BM25Store's rank_bm25.BM25Okapi construction), not an
        approximation of it. Compare scores from both directly on the same
        corpus and query — they should agree to floating-point precision,
        not just agree on ranking order."""
        from rank_bm25 import BM25Okapi

        docs = [
            "A Pod can hold several containers that share a network namespace.",
            "A Service exposes an application running on a set of Pods.",
            "A volume provides persistent storage that outlives a restart.",
            "Quarterly revenue rose sharply in Tokyo this year.",
        ]
        chunks = [make_chunk(t, f"docs/{i}.md", "en", i) for i, t in enumerate(docs)]

        store = FTS5Store(tmp_path / "idx.db")
        store.build(chunks, source_path=stub_source(tmp_path))

        tokenized = [d.lower().split() for d in docs]
        # WhitespaceSegmenter lowercases and strips punctuation via \w+,
        # same as this naive split does for these punctuation-light
        # sentences — close enough for the two to tokenize identically here.
        ref = BM25Okapi([[w.strip(".,") for w in doc] for doc in tokenized])

        query = "pod containers"
        ref_scores = dict(zip(
            [c.chunk_id for c in chunks], ref.get_scores(query.split()), strict=True,
        ))
        hits = store.search(query, top_k=len(chunks))
        for h in hits:
            assert h.score == pytest.approx(ref_scores[h.chunk_id], abs=1e-9)

    def test_query_word_that_collides_with_fts5_syntax_does_not_crash(self, tmp_path):
        """'NOT', 'OR', 'AND', and punctuation like '*' or ':' are FTS5
        query-syntax operators. An ordinary English sentence can contain
        any of them as a plain word, and a query built without quoting
        would either throw a syntax error or silently mean something
        other than "search for this word." """
        store = FTS5Store(tmp_path / "idx.db")
        store.build([
            make_chunk("A container will not restart automatically here.",
                       "docs/a.md", "en"),
        ], source_path=stub_source(tmp_path))
        hits = store.search("will not restart OR fail", top_k=5)
        assert hits[0].payload["parallel_id"] == "docs/a.md"


@requires_sudachi
class TestFTS5StoreJapanese:
    def test_real_japanese_query_retrieves_the_matching_chunk(self, tmp_path):
        store = FTS5Store(tmp_path / "idx.db")
        store.build([
            make_chunk("ポッドは複数のコンテナを持てます", "docs/pods.md", "ja"),
            make_chunk("四半期の収益は東京で大きく伸びました", "docs/unrelated.md", "ja"),
        ], source_path=stub_source(tmp_path))
        hits = store.search("コンテナ", top_k=2)
        assert hits[0].payload["parallel_id"] == "docs/pods.md"

    def test_cross_lingual_query_does_not_match(self, tmp_path):
        store = FTS5Store(tmp_path / "idx.db")
        store.build([make_chunk("ポッドは複数のコンテナを持てます", "docs/pods.md", "ja")],
                    source_path=stub_source(tmp_path))
        hits = store.search("A Pod can hold several containers", top_k=1)
        assert hits == []


class TestPersistence:
    """The reason this backend exists: the index survives the process."""

    def test_reload_from_a_fresh_connection_returns_the_same_top_hit(self, tmp_path):
        db = tmp_path / "idx.db"
        chunks = [
            make_chunk("A Pod can hold several containers.", "docs/pods.md", "en"),
            make_chunk("Quarterly revenue rose sharply in Tokyo.", "docs/unrelated.md", "en"),
        ]
        builder = FTS5Store(db)
        builder.build(chunks, source_path=stub_source(tmp_path))
        builder.close()

        reopened = FTS5Store(db)
        n = reopened.load()
        assert n == 2
        hits = reopened.search("Pod containers", top_k=2)
        assert hits[0].payload["parallel_id"] == "docs/pods.md"

    def test_is_current_for_detects_an_unchanged_source(self, tmp_path):
        src = tmp_path / "chunks.jsonl"
        write_chunks_jsonl(src, [make_chunk("hello", "docs/a.md", "en")])
        db = tmp_path / "idx.db"

        store = build_or_load(db, src)
        assert store.count() == 1
        assert FTS5Store(db).is_current_for(src) is True

    def test_is_current_for_detects_a_changed_source(self, tmp_path):
        src = tmp_path / "chunks.jsonl"
        write_chunks_jsonl(src, [make_chunk("hello", "docs/a.md", "en")])
        db = tmp_path / "idx.db"
        build_or_load(db, src)

        write_chunks_jsonl(src, [
            make_chunk("hello", "docs/a.md", "en"),
            make_chunk("world", "docs/b.md", "en"),
        ])
        assert FTS5Store(db).is_current_for(src) is False

    def test_build_or_load_rebuilds_when_source_changed(self, tmp_path):
        src = tmp_path / "chunks.jsonl"
        write_chunks_jsonl(src, [make_chunk("hello", "docs/a.md", "en")])
        db = tmp_path / "idx.db"
        build_or_load(db, src).close()

        write_chunks_jsonl(src, [
            make_chunk("hello", "docs/a.md", "en"),
            make_chunk("world", "docs/b.md", "en"),
        ])
        store = build_or_load(db, src)
        assert store.count() == 2

    def test_is_current_for_a_missing_db_is_false(self, tmp_path):
        assert FTS5Store(tmp_path / "absent.db").is_current_for(tmp_path / "x.jsonl") is False

    def test_is_current_for_ignores_relative_vs_absolute_spelling(self, tmp_path, monkeypatch):
        """Regression: build() used to store str(source_path) verbatim, so
        building with a relative chunks_path and then checking staleness
        with an absolute one for the *same file* (or vice versa — exactly
        what happens when a build script is run with a relative argument
        and the server is started with KENSHO_CHUNKS as an absolute env
        var) compared unequal path strings and silently triggered a full
        rebuild inside create_app() on every server start, defeating the
        entire point of this backend. Caught via a real RSS measurement
        showing create_app() paying a build, not a load."""
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "chunks.jsonl"
        write_chunks_jsonl(src, [make_chunk("hello", "docs/a.md", "en")])
        db = tmp_path / "idx.db"

        relative_src = Path("chunks.jsonl")
        build_or_load(db, relative_src).close()

        absolute_src = tmp_path / "chunks.jsonl"
        assert FTS5Store(db).is_current_for(absolute_src) is True

    def test_load_chunks_round_trips_section_path_as_a_tuple(self, tmp_path):
        src = tmp_path / "chunks.jsonl"
        write_chunks_jsonl(src, [make_chunk("hello", "docs/a.md", "en")])
        loaded = load_chunks(src)
        assert loaded[0].section_path == ("A", "B")
