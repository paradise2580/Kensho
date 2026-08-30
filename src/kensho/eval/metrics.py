"""Binary-relevance ranking metrics, computed over deduplicated page ids.

All three take the same shape: a ``ranked`` sequence of ``parallel_id``
(already deduplicated by the harness, so a page with three retrieved chunks
counts once, at its best rank — not three times) and a ``relevant`` set of
the page ids that count as correct for that query.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant page appears in the top k, else 0.0.

    This is set-level recall ("did we find *a* correct source"), not the
    fraction of relevant pages found — with usually one correct page per
    query, that distinction rarely bites, but it matters for the rare item
    with more than one acceptable source.
    """
    return 1.0 if any(p in relevant for p in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, p in enumerate(ranked, start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    Handles the (rare) multi-relevant-page item correctly: the ideal
    ordering puts all relevant pages first, so IDCG uses
    ``min(len(relevant), k)`` relevant positions rather than assuming
    exactly one.
    """
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, p in enumerate(ranked[:k], start=1)
        if p in relevant
    )
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg
