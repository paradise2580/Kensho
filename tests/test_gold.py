"""Gold-set schema tests, plus a regression check on the real v1 set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kensho.eval.gold import GoldItem, load_gold, save_gold

_ROOT = Path(__file__).resolve().parents[1]
GOLD_V1 = _ROOT / "data" / "gold" / "gold_v1.jsonl"
STRUCTURAL_CHUNKS = _ROOT / "data" / "chunks" / "structural_t512_o64.jsonl"


class TestSaveLoadRoundTrip:
    def test_round_trips(self, tmp_path):
        items = [
            GoldItem(id="a", lang="en", topic="t", query="q?",
                     relevant_parallel_ids=("docs/x.md",), reference_answer="x"),
            GoldItem(id="b", lang="ja", topic="t", query="質問",
                     relevant_parallel_ids=(), answerable=False, notes="reserved"),
        ]
        path = tmp_path / "gold.jsonl"
        save_gold(items, path)
        loaded = load_gold(path)
        assert loaded == items

    def test_creates_parent_directory(self, tmp_path):
        items = [GoldItem(id="a", lang="en", topic="t", query="q?",
                          relevant_parallel_ids=("docs/x.md",))]
        path = tmp_path / "nested" / "gold.jsonl"
        save_gold(items, path)
        assert path.exists()


class TestValidation:
    def test_duplicate_id_rejected(self, tmp_path):
        items = [
            GoldItem(id="a", lang="en", topic="t", query="q1?",
                    relevant_parallel_ids=("docs/x.md",)),
            GoldItem(id="a", lang="en", topic="t", query="q2?",
                    relevant_parallel_ids=("docs/y.md",)),
        ]
        with pytest.raises(ValueError, match="duplicate gold id"):
            save_gold(items, tmp_path / "gold.jsonl")

    def test_answerable_without_relevant_ids_rejected(self, tmp_path):
        items = [GoldItem(id="a", lang="en", topic="t", query="q?",
                          relevant_parallel_ids=(), answerable=True)]
        with pytest.raises(ValueError, match="marked answerable"):
            save_gold(items, tmp_path / "gold.jsonl")

    def test_unanswerable_with_relevant_ids_rejected(self, tmp_path):
        items = [GoldItem(id="a", lang="en", topic="t", query="q?",
                          relevant_parallel_ids=("docs/x.md",), answerable=False)]
        with pytest.raises(ValueError, match="marked unanswerable"):
            save_gold(items, tmp_path / "gold.jsonl")

    def test_load_rejects_duplicate_ids_written_by_hand(self, tmp_path):
        path = tmp_path / "gold.jsonl"
        path.write_text(
            json.dumps({"id": "a", "lang": "en", "topic": "t", "query": "q1?",
                       "relevant_parallel_ids": ["docs/x.md"]}) + "\n"
            + json.dumps({"id": "a", "lang": "en", "topic": "t", "query": "q2?",
                         "relevant_parallel_ids": ["docs/y.md"]}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate gold id"):
            load_gold(path)


@pytest.mark.skipif(
    not GOLD_V1.exists(), reason="gold_v1.jsonl not built (run scripts/seed_gold_v1.py)"
)
class TestGoldV1:
    def test_loads_without_error(self):
        items = load_gold(GOLD_V1)
        assert len(items) >= 40

    def test_has_both_languages(self):
        items = load_gold(GOLD_V1)
        langs = {it.lang for it in items}
        assert langs == {"en", "ja"}

    def test_has_some_unanswerable_items_reserved_for_phase5(self):
        items = load_gold(GOLD_V1)
        unanswerable = [it for it in items if not it.answerable]
        assert len(unanswerable) >= 2

    @pytest.mark.skipif(not STRUCTURAL_CHUNKS.exists(),
                        reason="corpus not built (run scripts/build_corpus.py)")
    def test_every_relevant_page_exists_in_the_built_corpus(self):
        """REGRESSION guard: a page the gold set points to but the corpus no
        longer contains would silently zero out that item's recall for a
        reason that has nothing to do with retrieval quality."""
        corpus_ids = set()
        with STRUCTURAL_CHUNKS.open(encoding="utf-8") as fh:
            for line in fh:
                corpus_ids.add(json.loads(line)["parallel_id"])
        items = load_gold(GOLD_V1)
        missing = [
            (it.id, pid)
            for it in items
            for pid in it.relevant_parallel_ids
            if pid not in corpus_ids
        ]
        assert missing == []
