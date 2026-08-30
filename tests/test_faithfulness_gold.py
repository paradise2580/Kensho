from __future__ import annotations

from pathlib import Path

import pytest

from kensho.eval.faithfulness_gold import FaithfulnessItem, load_faithfulness, save_faithfulness

_ROOT = Path(__file__).resolve().parents[1]
FAITHFULNESS_V1 = _ROOT / "data" / "gold" / "faithfulness_v1.jsonl"


class TestSaveLoadRoundTrip:
    def test_round_trips(self, tmp_path):
        items = [
            FaithfulnessItem(id="a", lang="en", topic="t", evidence=("ev",),
                            claim="c", true_label="supported"),
            FaithfulnessItem(id="b", lang="ja", topic="t", evidence=("証拠",),
                            claim="主張", true_label="contradicted"),
        ]
        path = tmp_path / "faith.jsonl"
        save_faithfulness(items, path)
        assert load_faithfulness(path) == items


class TestValidation:
    def test_duplicate_id_rejected(self, tmp_path):
        items = [
            FaithfulnessItem(id="a", lang="en", topic="t", evidence=("ev1",),
                            claim="c1", true_label="supported"),
            FaithfulnessItem(id="a", lang="en", topic="t", evidence=("ev2",),
                            claim="c2", true_label="unsupported"),
        ]
        with pytest.raises(ValueError, match="duplicate faithfulness item id"):
            save_faithfulness(items, tmp_path / "faith.jsonl")


@pytest.mark.skipif(not FAITHFULNESS_V1.exists(),
                    reason="faithfulness_v1.jsonl not built (run scripts/seed_faithfulness_v1.py)")
class TestFaithfulnessV1:
    def test_loads_without_error(self):
        assert len(load_faithfulness(FAITHFULNESS_V1)) >= 40

    def test_has_all_three_labels(self):
        items = load_faithfulness(FAITHFULNESS_V1)
        labels = {it.true_label for it in items}
        assert labels == {"supported", "unsupported", "contradicted"}

    def test_labels_are_balanced(self):
        """Not a hard requirement, but a wildly imbalanced set would make
        accuracy numbers from compare_verifiers.py misleading."""
        items = load_faithfulness(FAITHFULNESS_V1)
        counts = {lbl: sum(1 for it in items if it.true_label == lbl)
                 for lbl in ("supported", "unsupported", "contradicted")}
        assert max(counts.values()) - min(counts.values()) <= 4

    def test_has_both_languages(self):
        items = load_faithfulness(FAITHFULNESS_V1)
        assert {it.lang for it in items} == {"en", "ja"}

    def test_every_item_has_nonempty_evidence(self):
        items = load_faithfulness(FAITHFULNESS_V1)
        assert all(it.evidence and all(e.strip() for e in it.evidence) for it in items)
