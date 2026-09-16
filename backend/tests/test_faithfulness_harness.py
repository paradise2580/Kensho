from __future__ import annotations

from kensho.eval.faithfulness_gold import FaithfulnessItem
from kensho.eval.faithfulness_harness import evaluate_verifier
from kensho.verify.nli import FakeVerifier, LexicalOverlapVerifier


def item(id_: str, claim: str, evidence: str, true_label: str) -> FaithfulnessItem:
    return FaithfulnessItem(id=id_, lang="en", topic="t", evidence=(evidence,),
                            claim=claim, true_label=true_label)


class TestEvaluateVerifier:
    def test_perfect_verifier_scores_full_accuracy(self):
        class OracleVerifier:
            name = "oracle"

            def verify(self, claim, evidence):
                from kensho.verify.nli import VerificationResult
                # cheat: label encoded in the claim text for this test only
                label = claim.split("::")[1]
                return VerificationResult(claim, label, 1.0, 0, self.name)

        items = [
            item("a", "x::supported", "ev", "supported"),
            item("b", "y::unsupported", "ev", "unsupported"),
            item("c", "z::contradicted", "ev", "contradicted"),
        ]
        summary = evaluate_verifier(items, OracleVerifier())
        assert summary.accuracy == 1.0
        assert summary.n == 3
        for lbl in ("supported", "unsupported", "contradicted"):
            assert summary.per_label[lbl].recall == 1.0
            assert summary.per_label[lbl].precision == 1.0

    def test_lexical_overlap_gets_real_supported_case_right(self):
        items = [item(
            "a",
            "Pods follow a defined lifecycle starting in the Pending phase.",
            "Pods follow a defined lifecycle, starting in the Pending phase.",
            "supported",
        )]
        summary = evaluate_verifier(items, LexicalOverlapVerifier(threshold=0.6))
        assert summary.accuracy == 1.0

    def test_lexical_overlap_structurally_cannot_recognize_contradiction(self):
        """Documented limitation from verify/nli.py: lexical overlap has no
        'contradicted' output at all, so recall on that label is always 0."""
        items = [item("a", "Pods never start in the Pending phase.",
                      "Pods follow a defined lifecycle, starting in the Pending phase.",
                      "contradicted")]
        summary = evaluate_verifier(items, LexicalOverlapVerifier(threshold=0.6))
        assert summary.per_label["contradicted"].recall == 0.0

    def test_confusion_counts_sum_to_n(self):
        items = [
            item("a", "claim a", "ev", "supported"),
            item("b", "claim b", "ev", "unsupported"),
        ]
        summary = evaluate_verifier(items, FakeVerifier())
        assert sum(summary.confusion.values()) == 2

    def test_empty_gold_set(self):
        summary = evaluate_verifier([], LexicalOverlapVerifier())
        assert summary.n == 0
        assert summary.accuracy == 0.0
