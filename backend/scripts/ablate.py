#!/usr/bin/env python3
"""Run all seven phase-3 retrieval ablations and print a comparison table.

    python scripts/ablate.py data/gold/gold_v1.jsonl

Each ablation changes exactly one thing relative to the phase-1 baseline
(dense-only, structural chunks, multilingual-e5-large, top-5, unrestricted
language) so a difference in the numbers can be attributed to that one
change:

  3.1  baseline               dense retrieval, as shipped in phase 1
  3.2  chunking: fixed        same everything else, ``fixed`` chunks instead
                              of ``structural`` — requires its own index,
                              see --fixed-index-dir
  3.3  embedder: bge-m3       same chunks, a different embedding model —
                              requires its own index, see --bge-m3-index-dir
  3.4  sparse (BM25)          lexical retrieval instead of dense; needs no
                              prebuilt index, runs in-memory from the chunk
                              JSONL directly
  3.5  hybrid (RRF)           fuses the baseline dense store with an
                              in-memory BM25 index over the same chunks
  3.6  rerank                 baseline dense store, over-fetch 20 and
                              rerank down to top-5 with a cross-encoder
  3.7  cross-lingual          baseline dense store, but each query is
                              restricted to chunks in its own language —
                              the gap versus 3.1 is what cross-lingual
                              retrieval is worth

3.2 and 3.3 are skipped with a note if their index directory doesn't exist
(building them is a separate ``build_index.py`` invocation — see the
README). 3.4-3.7 always run: BM25 needs no prebuilt index, and 3.5-3.7 all
reuse the same baseline dense index as 3.1.

With ``--allow-fallback``, every dense/reranker step that would otherwise
require Hugging Face Hub access falls back to the fake stand-ins, so the
whole comparison runs as a plumbing smoke test with no model access at all.
The resulting numbers are not retrieval quality and the table says so.
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
from kensho.eval.harness import EvalReport, evaluate  # noqa: E402
from kensho.retrieval.hybrid import HybridRetriever  # noqa: E402
from kensho.retrieval.rerank import RerankingRetriever, get_reranker  # noqa: E402
from kensho.retrieval.sparse import BM25Store  # noqa: E402
from kensho.schema import Chunk  # noqa: E402
from kensho.store import ChunkStore  # noqa: E402

BGE_M3 = "BAAI/bge-m3"


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            d["section_path"] = tuple(d["section_path"])
            chunks.append(Chunk(**d))
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold_path", type=Path)
    ap.add_argument("--structural-chunks", type=Path,
                    default=Path("data/chunks/structural_t512_o64.jsonl"))
    ap.add_argument("--fixed-chunks", type=Path,
                    default=Path("data/chunks/fixed_t512_o64.jsonl"))
    ap.add_argument("--baseline-index-dir", type=Path, default=Path("data/qdrant"))
    ap.add_argument("--fixed-index-dir", type=Path, default=Path("data/qdrant_fixed"))
    ap.add_argument("--bge-m3-index-dir", type=Path, default=Path("data/qdrant_bge_m3"))
    ap.add_argument("--embedder", default=DEFAULT_MODEL)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--allow-fallback", action="store_true",
                    help="fall back to fake embedder/reranker instead of "
                         "failing — proves the comparison runs end to end, "
                         "the numbers are not retrieval quality")
    ap.add_argument("--out-dir", type=Path, default=Path("eval_reports"))
    args = ap.parse_args()

    gold = load_gold(args.gold_path)
    fb = args.allow_fallback

    baseline_embedder = get_embedder(
        None if args.embedder == "fake" else args.embedder, allow_fallback=fb,
    )
    baseline_store = ChunkStore(baseline_embedder, path=args.baseline_index_dir)
    if baseline_store.count() == 0:
        print(f"warning: baseline index at {args.baseline_index_dir} is empty — "
              "run build_index.py first", file=sys.stderr)

    bm25 = BM25Store(allow_fallback=fb)
    bm25.index(load_chunks(args.structural_chunks))

    reports: dict[str, EvalReport] = {}

    reports["3.1 baseline (dense)"] = evaluate(gold, baseline_store, top_k=args.top_k)

    if args.fixed_index_dir.exists():
        fixed_store = ChunkStore(baseline_embedder, path=args.fixed_index_dir)
        reports["3.2 chunking: fixed"] = evaluate(gold, fixed_store, top_k=args.top_k)
    else:
        print(f"skipping 3.2 — no index at {args.fixed_index_dir} (build with: "
              f"build_index.py {args.fixed_chunks} --index-dir {args.fixed_index_dir})")

    if args.bge_m3_index_dir.exists():
        bge_embedder = get_embedder(BGE_M3, allow_fallback=fb)
        bge_store = ChunkStore(bge_embedder, path=args.bge_m3_index_dir)
        reports["3.3 embedder: bge-m3"] = evaluate(gold, bge_store, top_k=args.top_k)
    else:
        print(f"skipping 3.3 — no index at {args.bge_m3_index_dir} "
              f"(build with: build_index.py {args.structural_chunks} "
              f"--index-dir {args.bge_m3_index_dir} --embedder {BGE_M3})")

    reports["3.4 sparse (BM25)"] = evaluate(gold, bm25, top_k=args.top_k)

    hybrid = HybridRetriever(baseline_store, bm25)
    reports["3.5 hybrid (RRF)"] = evaluate(gold, hybrid, top_k=args.top_k)

    reranker = get_reranker(allow_fallback=fb)
    reranking = RerankingRetriever(baseline_store, reranker, fetch_k=20)
    reports["3.6 rerank"] = evaluate(gold, reranking, top_k=args.top_k)

    reports["3.7 cross-lingual restricted"] = evaluate(
        gold, baseline_store, top_k=args.top_k, lang_filter="match_query",
    )

    print_table(reports)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = args.out_dir / f"{stamp}_ablations.json"
    payload = {
        name: {**report.to_dict(include_items=False)}
        for name, report in reports.items()
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote report -> {out_path}")
    if fb:
        print("NOTE: --allow-fallback was set — any 'fake'/'echo' component "
              "above makes that row plumbing-only, not retrieval quality.")
    return 0


def print_table(reports: dict[str, EvalReport]) -> None:
    header = f"{'ablation':<32} {'retriever':<28} {'r@1':>6} {'r@5':>6} {'mrr':>6} {'ndcg':>6}"
    print(header)
    print("-" * len(header))
    for name, report in reports.items():
        m = report.overall
        r1 = _fmt(m.recall_at_k.get(1))
        r5 = _fmt(m.recall_at_k.get(5))
        print(f"{name:<32} {report.retriever_name:<28.28} {r1:>6} {r5:>6} "
              f"{_fmt(m.mrr):>6} {_fmt(m.ndcg):>6}")


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
