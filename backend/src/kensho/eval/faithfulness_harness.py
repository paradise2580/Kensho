"""Scores one ClaimVerifier against the faithfulness gold set.

Accuracy alone would hide the thing that actually matters for this project:
recall on "contradicted" and "unsupported" is what stands between a user and
a hallucinated answer, so a method that's accurate overall but blind to
contradictions (lexical overlap and embedding similarity both are, by
construction — see ``verify/nli.py``) needs that weakness visible in the
per-label breakdown, not averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..verify.nli import ClaimVerifier
from .faithfulness_gold import FaithfulnessItem, Label

LABELS: tuple[Label, ...] = ("supported", "unsupported", "contradicted")


@dataclass(frozen=True, slots=True)
class LabelMetrics:
    support: int
    precision: float | None
    recall: float | None


@dataclass(frozen=True, slots=True)
class ConfusionSummary:
    n: int
    accuracy: float
    per_label: dict[Label, LabelMetrics]
    confusion: dict[tuple[Label, Label], int]  # (true, predicted) -> count

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "per_label": {
                lbl: {"support": m.support, "precision": m.precision, "recall": m.recall}
                for lbl, m in self.per_label.items()
            },
            "confusion": {f"{t}->{p}": c for (t, p), c in self.confusion.items()},
        }


def evaluate_verifier(items: list[FaithfulnessItem], verifier: ClaimVerifier) -> ConfusionSummary:
    confusion: dict[tuple[Label, Label], int] = {(t, p): 0 for t in LABELS for p in LABELS}
    for item in items:
        predicted = verifier.verify(item.claim, list(item.evidence)).label
        confusion[(item.true_label, predicted)] += 1

    n = len(items)
    correct = sum(confusion[(lbl, lbl)] for lbl in LABELS)
    accuracy = correct / n if n else 0.0

    per_label: dict[Label, LabelMetrics] = {}
    for lbl in LABELS:
        support = sum(confusion[(lbl, p)] for p in LABELS)
        predicted_as_lbl = sum(confusion[(t, lbl)] for t in LABELS)
        tp = confusion[(lbl, lbl)]
        precision = tp / predicted_as_lbl if predicted_as_lbl else None
        recall = tp / support if support else None
        per_label[lbl] = LabelMetrics(support=support, precision=precision, recall=recall)

    return ConfusionSummary(n=n, accuracy=accuracy, per_label=per_label, confusion=confusion)
