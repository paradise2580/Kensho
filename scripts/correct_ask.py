#!/usr/bin/env python3
"""End-to-end query with verification and self-correction.

    python scripts/correct_ask.py "How does a Pod's restart policy work?"

The full stack: retrieve, generate (phase 1) -> decompose, verify (phase 4)
-> strike / retry / refuse (phase 5). Prints the final answer plus a
per-claim breakdown so it's visible *why* a claim survived, was struck, or
triggered a retry — a silently-edited answer would defeat the point of
building verification in the first place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.correct.policy import CorrectionConfig, correct_answer  # noqa: E402
from kensho.embed import DEFAULT_MODEL, get_embedder  # noqa: E402
from kensho.llm import get_llm  # noqa: E402
from kensho.pipeline import ask  # noqa: E402
from kensho.store import ChunkStore  # noqa: E402
from kensho.verify.claims import get_decomposer  # noqa: E402
from kensho.verify.nli import get_verifier  # noqa: E402
from kensho.verify.pipeline import verify_answer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--index-dir", type=Path, default=Path("data/qdrant"))
    ap.add_argument("--embedder", default=DEFAULT_MODEL)
    ap.add_argument("--llm", default="groq")
    ap.add_argument("--decomposer", choices=["sentence", "llm"], default="sentence")
    ap.add_argument("--verifier", choices=["lexical", "embedding", "nli", "llm-judge"],
                    default="lexical")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--refusal-threshold", type=float, default=0.5)
    ap.add_argument("--allow-fallback", action="store_true",
                    help="allow fake/echo components instead of failing — "
                         "proves the pipeline runs, the verdicts are not real")
    args = ap.parse_args()

    fb = args.allow_fallback
    embedder = get_embedder(None if args.embedder == "fake" else args.embedder, allow_fallback=fb)
    store = ChunkStore(embedder, path=args.index_dir)
    llm = get_llm(None if args.llm == "echo" else args.llm, allow_fallback=fb)
    decomposer = get_decomposer(args.decomposer, llm=llm)
    verifier = get_verifier(args.verifier, llm=llm, embedder=embedder, allow_fallback=fb)

    answer = ask(args.question, store, llm, top_k=args.top_k)
    verified = verify_answer(answer, decomposer, verifier)
    corrected = correct_answer(verified, store, verifier,
                               config=CorrectionConfig(refusal_threshold=args.refusal_threshold))

    print(f"Q: {args.question}\n")
    print(f"A: {corrected.final_text}\n")
    if corrected.refused:
        print(f"[REFUSED: {corrected.refusal_reason}]\n")

    print(f"Claims ({len(corrected.decisions)}):")
    for d in corrected.decisions:
        tag = {"kept": "kept", "struck": "struck", "recovered": "recovered (retry)"}[d.action]
        print(f"  [{tag:<18}] ({d.final_result.label}) {d.verified.claim.text}")

    if verifier.name in ("fake", "echo") or fb:
        print("\nNOTE: --allow-fallback was set — verdicts above may be plumbing-only, not real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
