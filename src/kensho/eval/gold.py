"""The hand-labelled gold set for retrieval evaluation.

Every item was written by reading the actual retrieved page — in both
languages, where the item is answerable — from the corpus this project
builds (see ``corpus/fetch.py``), not recalled from general Kubernetes
knowledge. That distinction matters: a wrong ``relevant_parallel_ids`` entry
here is a labelling bug to fix, not a knowledge gap to shrug off, because the
whole point of the gold set is to be ground truth the harness can be trusted
against.

Ground truth is recorded at **page** (``parallel_id``) granularity rather
than ``chunk_id``. Chunk boundaries differ between the ``structural`` and
``fixed`` strategies (ablation 3.2 compares them), so a chunk-level label
would silently go stale the moment the chunking strategy under test changes;
a page-level label stays valid across both.

A handful of items are deliberately unanswerable — plausible support queries
the corpus has no coverage for, or ones with a nonsensical premise. They
carry ``answerable=False`` and an empty ``relevant_parallel_ids``, and the
harness excludes them from retrieval metrics (recall/MRR/nDCG have no
meaning when nothing should be retrieved). They exist now, seeded alongside
the rest of the set, so phase 5's refusal evaluation has real cases to run
against instead of writing its own gold set from scratch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GoldItem:
    id: str
    query: str
    lang: str  # "en" | "ja"
    topic: str
    relevant_parallel_ids: tuple[str, ...]
    reference_answer: str | None = None
    answerable: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["relevant_parallel_ids"] = list(self.relevant_parallel_ids)
        return d


def load_gold(path: str | Path) -> list[GoldItem]:
    items = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["relevant_parallel_ids"] = tuple(d["relevant_parallel_ids"])
            items.append(GoldItem(**d))
    _check_unique_ids(items)
    _check_answerable_has_relevant(items)
    return items


def save_gold(items: list[GoldItem], path: str | Path) -> None:
    _check_unique_ids(items)
    _check_answerable_has_relevant(items)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def _check_unique_ids(items: list[GoldItem]) -> None:
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"duplicate gold id: {it.id!r}")
        seen.add(it.id)


def _check_answerable_has_relevant(items: list[GoldItem]) -> None:
    for it in items:
        if it.answerable and not it.relevant_parallel_ids:
            raise ValueError(
                f"gold item {it.id!r} is marked answerable but lists no "
                "relevant_parallel_ids — either it's mislabelled or should "
                "be answerable=False."
            )
        if not it.answerable and it.relevant_parallel_ids:
            raise ValueError(
                f"gold item {it.id!r} is marked unanswerable but lists "
                f"relevant_parallel_ids {it.relevant_parallel_ids!r} — "
                "an unanswerable item should have no ground-truth source."
            )
