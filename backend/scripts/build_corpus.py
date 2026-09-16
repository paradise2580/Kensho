#!/usr/bin/env python3
"""Build the chunked corpus.

    python scripts/build_corpus.py --strategy structural --target-tokens 512

Writes JSONL to ``data/chunks/<strategy>_t<target>_o<overlap>.jsonl`` plus a
sidecar manifest recording the corpus SHA, the tokenizer actually used, and
the resulting statistics. Every ablation in phase 3 is a different invocation
of this script, and the manifest is what makes runs comparable after the fact.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.corpus.chunk import STRATEGIES, chunk_document  # noqa: E402
from kensho.corpus.fetch import clone, corpus_stats, load_documents  # noqa: E402
from kensho.tokenizer import get_token_counter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-dir", type=Path, default=Path("data/k8s-website"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/chunks"))
    ap.add_argument("--ref", default=None,
                    help="commit SHA or tag to pin. Omit for current HEAD of main.")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="structural")
    ap.add_argument("--target-tokens", type=int, default=512)
    ap.add_argument("--overlap-tokens", type=int, default=64)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--tokenizer", default="intfloat/multilingual-e5-large",
                    help="HF tokenizer id, or 'heuristic' for the offline approximation")
    ap.add_argument("--strict-tokenizer", action="store_true",
                    help="fail rather than silently falling back to the heuristic")
    ap.add_argument("--include-reference", action="store_true",
                    help="include docs/reference/ (auto-generated API stubs; "
                         "see fetch.SKIP_PREFIXES for why this is off by default)")
    ap.add_argument("--skip-clone", action="store_true")
    args = ap.parse_args()

    sha = ("local" if args.skip_clone else clone(args.repo_dir, args.ref))
    if args.skip_clone:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo_dir,
                           capture_output=True, text=True)
        sha = r.stdout.strip() or "unknown"

    stats = corpus_stats(args.repo_dir, sha, args.include_reference)
    print(f"corpus @ {sha[:12]}  en={stats.en_pages}  ja={stats.ja_pages}  "
          f"parallel={stats.parallel_pages}  en_only={stats.en_only_pages}")

    counter = get_token_counter(
        None if args.tokenizer == "heuristic" else args.tokenizer,
        allow_fallback=not args.strict_tokenizer,
    )
    print(f"tokenizer: {counter.name}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.strategy}_t{args.target_tokens}_o{args.overlap_tokens}"
    out_path = args.out_dir / f"{stem}.jsonl"

    by_lang: Counter[str] = Counter()
    tokens_by_lang: Counter[str] = Counter()
    empty_docs: list[str] = []
    n = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for doc in load_documents(args.repo_dir, sha,
                                      include_reference=args.include_reference):
            chunks = chunk_document(
                doc, counter, strategy=args.strategy,
                target_tokens=args.target_tokens,
                overlap_tokens=args.overlap_tokens,
                min_tokens=args.min_tokens,
            )
            if not chunks:
                empty_docs.append(doc.doc_id)
            for c in chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                by_lang[c.lang] += 1
                tokens_by_lang[c.lang] += c.token_count
                n += 1

    manifest = {
        "corpus_sha": sha,
        "strategy": args.strategy,
        "target_tokens": args.target_tokens,
        "overlap_tokens": args.overlap_tokens,
        "min_tokens": args.min_tokens,
        "include_reference": args.include_reference,
        "tokenizer": counter.name,
        "pages": dataclasses.asdict(stats),
        "chunks_total": n,
        "chunks_by_lang": dict(by_lang),
        "mean_tokens_by_lang": {
            k: round(tokens_by_lang[k] / by_lang[k], 1) for k in by_lang
        },
        "documents_yielding_no_chunks": len(empty_docs),
        "output": str(out_path),
    }
    (args.out_dir / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {n} chunks -> {out_path}")
    print(f"  by lang       : {dict(by_lang)}")
    print(f"  mean tokens   : {manifest['mean_tokens_by_lang']}")
    print(f"  empty docs    : {len(empty_docs)}")
    if counter.name == "heuristic":
        print("  NOTE: heuristic tokenizer — do not publish numbers from this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
