#!/usr/bin/env python3
"""Run all four verification methods against the faithfulness gold set.

    python scripts/compare_verifiers.py data/gold/faithfulness_v1.jsonl

This is the "4-arm comparison" the project's name is actually about: cheapest
(lexical overlap) to most expensive (a second LLM call), scored on whether
each one gets the *label* right — not just whether it flags a problem, since
recall on "contradicted" specifically is what a lexical or embedding method
structurally cannot deliver (see verify/nli.py's module docstring).

With ``--allow-fallback``, the ``nli`` arm falls back to a fake, meaning-blind
verifier if mDeBERTa can't load (no Hugging Face access), and the
``llm-judge`` arm falls back to Echo if no ``GROQ_API_KEY`` is set — both
rows are then plumbing-only and the table says so. ``lexical`` never needs a
model and always produces a real, reportable number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.embed import DEFAULT_MODEL, get_embedder  # noqa: E402
from kensho.eval.faithfulness_gold import load_faithfulness  # noqa: E402
from kensho.eval.faithfulness_harness import ConfusionSummary, evaluate_verifier  # noqa: E402
from kensho.llm import get_llm  # noqa: E402
from kensho.verify.nli import get_verifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("faithfulness_path", type=Path)
    ap.add_argument("--embedder", default=DEFAULT_MODEL,
                    help="embedder for the 'embedding' arm; 'fake' for the offline stand-in")
    ap.add_argument("--llm", default="groq", help="'groq' or 'echo' for the 'llm-judge' arm")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="fall back to fake/echo components instead of failing — "
                         "those rows become plumbing-only, not reportable accuracy")
    ap.add_argument("--out-dir", type=Path, default=Path("eval_reports"))
    args = ap.parse_args()

    items = load_faithfulness(args.faithfulness_path)
    fb = args.allow_fallback

    embedder = get_embedder(None if args.embedder == "fake" else args.embedder, allow_fallback=fb)
    llm = get_llm(None if args.llm == "echo" else args.llm, allow_fallback=fb)

    verifiers = {
        "lexical": get_verifier("lexical"),
        "embedding": get_verifier("embedding", embedder=embedder),
        "nli": get_verifier("nli", allow_fallback=fb),
        "llm-judge": get_verifier("llm-judge", llm=llm),
    }

    results: dict[str, ConfusionSummary] = {
        arm: evaluate_verifier(items, verifier) for arm, verifier in verifiers.items()
    }

    print_table(verifiers, results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = args.out_dir / f"{stamp}_verifier_comparison.json"
    payload = {
        arm: {"verifier_name": verifiers[arm].name, **summary.to_dict()}
        for arm, summary in results.items()
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote report -> {out_path}")
    if fb:
        print("NOTE: --allow-fallback was set — any 'fake'/'echo' component "
              "above makes that row plumbing-only, not reportable accuracy.")
    return 0


def print_table(verifiers: dict, results: dict[str, ConfusionSummary]) -> None:
    header = (f"{'arm':<12} {'verifier':<30} {'acc':>6} "
             f"{'rec(unsup)':>11} {'rec(contra)':>12}")
    print(header)
    print("-" * len(header))
    for arm, summary in results.items():
        rec_unsup = _fmt(summary.per_label["unsupported"].recall)
        rec_contra = _fmt(summary.per_label["contradicted"].recall)
        print(f"{arm:<12} {verifiers[arm].name:<30.30} {summary.accuracy:>6.3f} "
              f"{rec_unsup:>11} {rec_contra:>12}")


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
