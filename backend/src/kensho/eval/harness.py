"""Runs the gold set against a ChunkStore and aggregates retrieval metrics.

Phase 2 scope is retrieval only — no generation, no verification. That is
deliberate: the gold set's ``relevant_parallel_ids`` is ground truth for "did
we find the right source," which can be checked mechanically, while judging
whether a *generated answer* is faithful to that source is exactly the
problem phase 4's NLI verification exists to solve. Wiring an eval harness to
an LLM call before that exists would mean eyeballing free text and calling it
measurement.

Unanswerable gold items are counted but excluded from every metric below —
recall/MRR/nDCG are undefined when nothing should be retrieved. They are
reserved for phase 5 (refusal evaluation).

``store`` accepts anything conforming to the ``Retriever`` protocol — a
dense ``ChunkStore``, a ``BM25Store``, a ``HybridRetriever``, or a
``RerankingRetriever`` wrapping any of those. That's what makes each of the
phase 3 ablations a different object passed to the same ``evaluate()`` call
rather than a separate harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..store import Retriever
from .gold import GoldItem
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank

K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
NDCG_K = 10


@dataclass(frozen=True, slots=True)
class ItemResult:
    gold_id: str
    query: str
    lang: str
    answerable: bool
    relevant: tuple[str, ...]
    ranked: tuple[str, ...]  # deduplicated parallel_ids, best rank per page


@dataclass(frozen=True, slots=True)
class MetricSet:
    n: int
    recall_at_k: dict[int, float | None]
    mrr: float | None
    ndcg: float | None

    def to_dict(self) -> dict:
        return {"n": self.n, "recall_at_k": self.recall_at_k, "mrr": self.mrr, "ndcg": self.ndcg}


@dataclass(frozen=True, slots=True)
class EvalReport:
    n_items: int
    n_answerable: int
    n_unanswerable: int
    top_k: int
    retriever_name: str
    lang_filter: str | None
    overall: MetricSet
    by_lang: dict[str, MetricSet]
    items: list[ItemResult] = field(default_factory=list)

    def to_dict(self, *, include_items: bool = True) -> dict:
        d = {
            "n_items": self.n_items,
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "top_k": self.top_k,
            "retriever_name": self.retriever_name,
            "lang_filter": self.lang_filter,
            "overall": self.overall.to_dict(),
            "by_lang": {lang: m.to_dict() for lang, m in self.by_lang.items()},
        }
        if include_items:
            d["items"] = [
                {
                    "gold_id": it.gold_id,
                    "query": it.query,
                    "lang": it.lang,
                    "answerable": it.answerable,
                    "relevant": list(it.relevant),
                    "ranked": list(it.ranked),
                }
                for it in self.items
            ]
        return d


def _dedupe(parallel_ids: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for p in parallel_ids:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def _aggregate(results: list[ItemResult]) -> MetricSet:
    if not results:
        return MetricSet(n=0, recall_at_k=dict.fromkeys(K_VALUES), mrr=None, ndcg=None)
    n = len(results)
    recall = {
        k: sum(recall_at_k(r.ranked, set(r.relevant), k) for r in results) / n
        for k in K_VALUES
    }
    mrr = sum(reciprocal_rank(r.ranked, set(r.relevant)) for r in results) / n
    ndcg = sum(ndcg_at_k(r.ranked, set(r.relevant), NDCG_K) for r in results) / n
    return MetricSet(n=n, recall_at_k=recall, mrr=mrr, ndcg=ndcg)


def evaluate(gold_items: list[GoldItem], store: Retriever, *, top_k: int = 10,
            lang_filter: str | None = None) -> EvalReport:
    """Run every gold item through ``store`` and aggregate metrics.

    ``lang_filter`` controls what language the *search itself* is
    restricted to, independent of the query's own language:

    - ``None`` (default) — unrestricted; retrieval may return chunks in
      either language. This is how the deployed system actually searches.
    - ``"en"`` / ``"ja"`` — force every query to search only that language's
      chunks, regardless of the query's own language.
    - ``"match_query"`` — restrict each query to chunks in its *own*
      language. Comparing this against the unrestricted default is ablation
      3.7: the gap between them is exactly the value cross-lingual
      retrieval adds — a query whose only correct source is untranslated
      into its own language will fail here but can succeed unrestricted.
    """
    results: list[ItemResult] = []
    for g in gold_items:
        query_lang = g.lang if lang_filter == "match_query" else lang_filter
        hits = store.search(g.query, top_k=top_k, lang=query_lang)
        ranked = _dedupe([h.payload["parallel_id"] for h in hits])
        results.append(ItemResult(
            gold_id=g.id, query=g.query, lang=g.lang, answerable=g.answerable,
            relevant=g.relevant_parallel_ids, ranked=ranked,
        ))

    answerable = [r for r in results if r.answerable]
    overall = _aggregate(answerable)
    by_lang = {
        lang: _aggregate([r for r in answerable if r.lang == lang])
        for lang in sorted({r.lang for r in answerable})
    }
    return EvalReport(
        n_items=len(results),
        n_answerable=len(answerable),
        n_unanswerable=len(results) - len(answerable),
        top_k=top_k,
        retriever_name=store.name,
        lang_filter=lang_filter,
        overall=overall,
        by_lang=by_lang,
        items=results,
    )
