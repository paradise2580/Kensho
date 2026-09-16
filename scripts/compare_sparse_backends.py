#!/usr/bin/env python3
"""Recall@5/MRR parity check: the in-memory BM25 backend vs. the disk-backed
FTS5 backend, on the same gold sets, same top_k, same corpus.

    python scripts/compare_sparse_backends.py

This is the comparison README.md and RUNBOOK.md point at before recommending
KENSHO_SPARSE_BACKEND=fts5 for anything beyond a RAM-constrained deployment.
The two backends are different storage for the same lexical retrieval (see
src/kensho/retrieval/fts5.py's docstring), not different algorithms — but
FTS5's own bm25() uses different k1/b constants than rank_bm25.BM25Okapi and
there is no public API to override them, so this script exists to measure
whether that constant drift actually moves retrieval quality rather than
assume it doesn't.

Runs both gold_v1 (CI-pinned) and gold_v2 (broader, +5 topics) separately,
since v1's numbers are what data/eval_floor.json is calibrated against.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.eval.gold import load_gold  # noqa: E402
from kensho.eval.harness import EvalReport, evaluate  # noqa: E402
from kensho.retrieval.fts5 import FTS5Store, load_chunks  # noqa: E402
from kensho.retrieval.sparse import BM25Store  # noqa: E402

CHUNKS = Path("data/chunks/structural_t512_o64.jsonl")
GOLD_SETS = ["data/gold/gold_v1.jsonl", "data/gold/gold_v2.jsonl"]
TOP_K = 10


def fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"


def print_side_by_side(name: str, mem: EvalReport, fts5: EvalReport) -> None:
    print(f"\n{name}  (n={mem.n_answerable} answerable / {mem.n_items} total)")
    print(f"  {'metric':<12} {'memory':>10} {'fts5':>10} {'delta':>10}")
    for k in (1, 3, 5, 10):
        m, f = mem.overall.recall_at_k[k], fts5.overall.recall_at_k[k]
        delta = f - m if (m is not None and f is not None) else None
        print(f"  {'recall@' + str(k):<12} {fmt(m):>10} {fmt(f):>10} "
              f"{(f'{delta:+.3f}') if delta is not None else '—':>10}")
    m, f = mem.overall.mrr, fts5.overall.mrr
    delta = f - m if (m is not None and f is not None) else None
    print(f"  {'MRR':<12} {fmt(m):>10} {fmt(f):>10} "
          f"{(f'{delta:+.3f}') if delta is not None else '—':>10}")

    # per-item ranking divergence: how many items the two backends disagree
    # on for the *top hit specifically*, independent of whether either is
    # right — this is what "the k1/b drift measurably changes ranking" or
    # "doesn't" looks like at the level of individual queries, not just an
    # aggregate that could hide compensating errors.
    mem_top = {r.gold_id: (r.ranked[0] if r.ranked else None) for r in mem.items}
    fts5_top = {r.gold_id: (r.ranked[0] if r.ranked else None) for r in fts5.items}
    disagreements = [gid for gid in mem_top if mem_top[gid] != fts5_top.get(gid)]
    print(f"  top-1 hit differs on {len(disagreements)}/{len(mem_top)} items")


def main() -> int:
    if not CHUNKS.exists():
        print(f"error: {CHUNKS} does not exist. Build it with scripts/build_corpus.py "
              f"first.", file=sys.stderr)
        return 1

    print(f"loading chunks from {CHUNKS} ...")
    chunks = load_chunks(CHUNKS)
    print(f"  {len(chunks):,} chunks")

    print("indexing (memory backend, rank_bm25.BM25Okapi) ...")
    mem_store = BM25Store(allow_fallback=False)
    mem_store.index(chunks)

    print("indexing (fts5 backend, SQLite FTS5) ...")
    with tempfile.TemporaryDirectory() as tmp:
        fts5_store = FTS5Store(Path(tmp) / "compare.db", allow_fallback=False)
        fts5_store.build(chunks, source_path=CHUNKS)

        for gold_path in GOLD_SETS:
            gold = load_gold(gold_path)
            mem_report = evaluate(gold, mem_store, top_k=TOP_K)
            fts5_report = evaluate(gold, fts5_store, top_k=TOP_K)
            print_side_by_side(gold_path, mem_report, fts5_report)

        fts5_store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
