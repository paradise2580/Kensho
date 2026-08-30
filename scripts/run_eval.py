#!/usr/bin/env python3
"""Run the gold set against a built index and report retrieval metrics.

    python scripts/run_eval.py data/gold/gold_v1.jsonl --index-dir data/qdrant

Writes a JSON report to ``eval_reports/`` so a later phase-3 ablation run can
be diffed against this baseline number rather than trusted from memory —
the same "every run leaves a manifest" discipline as build_corpus.py and
build_index.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.embed import DEFAULT_MODEL, get_embedder  # noqa: E402
from kensho.eval.gold import load_gold  # noqa: E402
from kensho.eval.harness import evaluate  # noqa: E402
from kensho.store import ChunkStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold_path", type=Path)
    ap.add_argument("--index-dir", type=Path, default=Path("data/qdrant"))
    ap.add_argument("--embedder", default=DEFAULT_MODEL,
                    help="must match the embedder the index was built with")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="allow the fake embedder if the real one can't load "
                         "— proves the harness runs end to end, produces "
                         "numbers that must never be reported as retrieval quality")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_reports"))
    args = ap.parse_args()

    gold = load_gold(args.gold_path)
    embedder = get_embedder(
        None if args.embedder == "fake" else args.embedder,
        allow_fallback=args.allow_fallback,
    )
    store = ChunkStore(embedder, path=args.index_dir)
    if store.count() == 0:
        print(f"warning: index at {args.index_dir} is empty — did you run build_index.py?",
              file=sys.stderr)

    report = evaluate(gold, store, top_k=args.top_k)

    print(f"gold={args.gold_path.name}  embedder={embedder.name}  "
          f"index_size={store.count()}  top_k={args.top_k}")
    print(f"  answerable={report.n_answerable}  unanswerable={report.n_unanswerable} "
          "(excluded from metrics below, reserved for phase 5)")
    print(f"  overall : recall@k={report.overall.recall_at_k}  "
          f"mrr={_fmt(report.overall.mrr)}  ndcg@10={_fmt(report.overall.ndcg)}")
    for lang, m in report.by_lang.items():
        print(f"  {lang:>4}    : n={m.n}  recall@k={m.recall_at_k}  "
              f"mrr={_fmt(m.mrr)}  ndcg@10={_fmt(m.ndcg)}")
    if embedder.name == "fake":
        print("  NOTE: fake embedder — these numbers are plumbing-only, not retrieval quality.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = args.out_dir / f"{stamp}_{embedder.name.replace('/', '_')}.json"
    payload = {
        "gold_path": str(args.gold_path),
        "embedder": embedder.name,
        "index_dir": str(args.index_dir),
        "index_size": store.count(),
        **report.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote report -> {out_path}")
    return 0


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
