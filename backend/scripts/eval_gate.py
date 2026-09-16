#!/usr/bin/env python3
"""Fail the build if retrieval or faithfulness quality regresses below a floor.

    python scripts/eval_gate.py

Every other retrieval/verification number in this project needs Hugging
Face Hub access (the embedder, the reranker, the NLI model) or a Groq key,
none of which a CI runner should be assumed to have reliably or fast. This
gate deliberately only uses the two components that need neither: BM25
retrieval (SudachiPy's dictionary ships in the wheel) and lexical-overlap
faithfulness verification. That is a real, if partial, regression check —
it catches a broken tokenizer, a corpus-parsing bug that empties out chunk
text, or a faithfulness-gold-set typo — not a stand-in for the full eval
suite. Floors live in ``data/eval_floor.json`` alongside the reasoning for
their values.

Exit code is 0 if every metric clears its floor, 1 otherwise — that's what
makes this usable as a CI gate rather than just another report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.corpus.chunk import chunk_document  # noqa: E402
from kensho.corpus.fetch import load_documents  # noqa: E402
from kensho.eval.faithfulness_gold import load_faithfulness  # noqa: E402
from kensho.eval.faithfulness_harness import evaluate_verifier  # noqa: E402
from kensho.eval.gold import load_gold  # noqa: E402
from kensho.eval.harness import evaluate  # noqa: E402
from kensho.retrieval.sparse import BM25Store  # noqa: E402
from kensho.tokenizer import get_token_counter  # noqa: E402
from kensho.verify.nli import LexicalOverlapVerifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    floors = json.loads((ROOT / "data" / "eval_floor.json").read_text(encoding="utf-8"))
    repo_dir = ROOT / "data" / "k8s-website"
    if not repo_dir.exists():
        print(f"error: corpus not found at {repo_dir} — run build_corpus.py first",
              file=sys.stderr)
        return 1
    sha = _current_sha(repo_dir)

    counter = get_token_counter(None)  # heuristic — no network dependency in CI
    bm25 = BM25Store()
    chunks = []
    for doc in load_documents(repo_dir, sha):
        chunks.extend(chunk_document(doc, counter, strategy="structural"))
    bm25.index(chunks)
    print(f"indexed {bm25.count()} chunks for the BM25 gate")

    gold = load_gold(ROOT / "data" / "gold" / "gold_v1.jsonl")
    retrieval_report = evaluate(gold, bm25, top_k=10)
    bm25_recall_at_5 = retrieval_report.overall.recall_at_k[5]

    faithfulness = load_faithfulness(ROOT / "data" / "gold" / "faithfulness_v1.jsonl")
    faithfulness_summary = evaluate_verifier(faithfulness, LexicalOverlapVerifier())
    lexical_accuracy = faithfulness_summary.accuracy

    results = {
        "bm25_recall_at_5": bm25_recall_at_5,
        "lexical_faithfulness_accuracy": lexical_accuracy,
    }

    ok = True
    print(f"{'metric':<32} {'value':>8} {'floor':>8}  status")
    for name, value in results.items():
        floor = floors[name]
        passed = value is not None and value >= floor
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        print(f"{name:<32} {value:>8.3f} {floor:>8.3f}  {status}")

    if not ok:
        print("\neval gate FAILED — a metric dropped below its floor in data/eval_floor.json")
        return 1
    print("\neval gate passed.")
    return 0


def _current_sha(repo_dir: Path) -> str:
    import subprocess
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
