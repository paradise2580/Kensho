#!/usr/bin/env python3
"""Score the full ask -> verify -> correct pipeline's refusal decisions
against the phase-2 retrieval gold set.

    python scripts/eval_refusal.py data/gold/gold_v1.jsonl

``gold_v1.jsonl`` was built with exactly this evaluation in mind: it has 48
answerable items (a real Kubernetes page backs the correct answer) and 4
deliberately unanswerable ones (plausible support questions this corpus has
no coverage for, or with a false premise) — see ``eval/gold.py``'s and
``scripts/seed_gold_v1.py``'s docstrings. A correct pipeline should refuse
on the 4 and answer on the 48; this script measures how close the actual
pipeline comes, in both directions:

- **false refusal rate** — answerable items the pipeline refused anyway.
  Every one is a real question a user would get "I don't know" for no reason.
- **miss rate** — unanswerable items the pipeline answered anyway. Every one
  is exactly the hallucination-that-looks-confident failure this whole
  project exists to catch.

Both numbers matter more than a single accuracy figure: a policy that
refuses everything scores well on one and terribly on the other.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.correct.policy import CorrectionConfig, correct_answer  # noqa: E402
from kensho.embed import DEFAULT_MODEL, get_embedder  # noqa: E402
from kensho.eval.gold import load_gold  # noqa: E402
from kensho.llm import get_llm  # noqa: E402
from kensho.pipeline import ask  # noqa: E402
from kensho.store import ChunkStore  # noqa: E402
from kensho.verify.claims import get_decomposer  # noqa: E402
from kensho.verify.nli import get_verifier  # noqa: E402
from kensho.verify.pipeline import verify_answer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold_path", type=Path)
    ap.add_argument("--index-dir", type=Path, default=Path("data/qdrant"))
    ap.add_argument("--embedder", default=DEFAULT_MODEL)
    ap.add_argument("--llm", default="groq")
    ap.add_argument("--verifier", choices=["lexical", "embedding", "nli", "llm-judge"],
                    default="lexical")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--refusal-threshold", type=float, default=0.5)
    ap.add_argument("--allow-fallback", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("eval_reports"))
    args = ap.parse_args()

    fb = args.allow_fallback
    gold = load_gold(args.gold_path)
    embedder = get_embedder(None if args.embedder == "fake" else args.embedder, allow_fallback=fb)
    store = ChunkStore(embedder, path=args.index_dir)
    llm = get_llm(None if args.llm == "echo" else args.llm, allow_fallback=fb)
    decomposer = get_decomposer("sentence")
    verifier = get_verifier(args.verifier, llm=llm, embedder=embedder, allow_fallback=fb)
    config = CorrectionConfig(refusal_threshold=args.refusal_threshold)

    rows = []
    for g in gold:
        answer = ask(g.query, store, llm, top_k=args.top_k)
        verified = verify_answer(answer, decomposer, verifier)
        corrected = correct_answer(verified, store, verifier, config=config)
        rows.append({
            "id": g.id, "answerable": g.answerable, "refused": corrected.refused,
            "n_claims": len(corrected.decisions),
        })

    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]

    false_refusals = [r for r in answerable_rows if r["refused"]]
    misses = [r for r in unanswerable_rows if not r["refused"]]

    false_refusal_rate = len(false_refusals) / len(answerable_rows) if answerable_rows else None
    miss_rate = len(misses) / len(unanswerable_rows) if unanswerable_rows else None

    print(f"gold={args.gold_path.name}  verifier={verifier.name}  n={len(gold)}")
    print(f"  answerable   : {len(answerable_rows)}  "
          f"false refusals: {len(false_refusals)}  rate={_fmt(false_refusal_rate)}")
    print(f"  unanswerable : {len(unanswerable_rows)}  "
          f"missed (answered anyway): {len(misses)}  rate={_fmt(miss_rate)}")
    if false_refusals:
        print(f"  false-refused ids: {[r['id'] for r in false_refusals]}")
    if misses:
        print(f"  missed ids: {[r['id'] for r in misses]}")
    if fb:
        print("  NOTE: --allow-fallback was set — with Echo/fake components every "
              "answer is uniformly meaningless, so these rates are plumbing-only.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = args.out_dir / f"{stamp}_refusal_eval.json"
    out_path.write_text(json.dumps({
        "verifier": verifier.name, "false_refusal_rate": false_refusal_rate,
        "miss_rate": miss_rate, "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote report -> {out_path}")
    return 0


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
