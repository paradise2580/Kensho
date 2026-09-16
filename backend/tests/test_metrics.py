from __future__ import annotations

import pytest

from kensho.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


class TestRecallAtK:
    def test_hit_at_top(self):
        assert recall_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0

    def test_hit_below_k_does_not_count(self):
        assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0

    def test_hit_within_k_counts(self):
        assert recall_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0

    def test_no_hit(self):
        assert recall_at_k(["a", "b"], {"z"}, k=2) == 0.0

    def test_empty_ranked_list(self):
        assert recall_at_k([], {"a"}, k=5) == 0.0


class TestReciprocalRank:
    def test_hit_at_rank_1(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_hit_at_rank_3(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit_is_zero(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_first_relevant_match_wins(self):
        """Multiple relevant pages present — score by the earliest one."""
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(1 / 2)


class TestNdcgAtK:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1.0)

    def test_no_hit_is_zero(self):
        assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_lower_rank_scores_less_than_top_rank(self):
        top = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
        mid = ndcg_at_k(["x", "a", "y"], {"a"}, k=3)
        assert top > mid > 0

    def test_hit_outside_k_scores_zero(self):
        assert ndcg_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_multi_relevant_ideal_ordering_is_one(self):
        """Two relevant pages, both in the top two slots -> perfect nDCG."""
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_empty_relevant_set_is_zero_not_an_error(self):
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0
