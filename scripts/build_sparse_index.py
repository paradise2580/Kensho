#!/usr/bin/env python3
"""Build the disk-backed FTS5 sparse index from a chunk JSONL file.

    python scripts/build_sparse_index.py data/chunks/structural_t512_o64.jsonl

Mirrors build_index.py's role for the dense store: a separate step from
chunking, run once and reused, rather than rebuilt in-process on every
server boot the way BM25Store (the in-memory backend) still is. See
src/kensho/retrieval/fts5.py for why this exists at all — the short version
is RSS: the in-memory backend measures ~390MB resident for this corpus,
which OOM-kills on a 512MB host once FastAPI/uvicorn and the rest of the
process are added on top.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.retrieval.fts5 import FTS5Store, load_chunks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chunks_path", type=Path)
    ap.add_argument("--db-path", type=Path, default=Path("data/index/bm25.db"))
    ap.add_argument("--allow-fallback", action="store_true",
                    help="fall back to character-bigram Japanese segmentation "
                         "instead of failing when SudachiPy isn't installed — "
                         "plumbing only, never for an index you intend to serve "
                         "real queries against")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if an up-to-date index already exists")
    args = ap.parse_args()

    if not args.chunks_path.exists():
        print(f"error: {args.chunks_path} does not exist. Build it with "
              f"scripts/build_corpus.py first.", file=sys.stderr)
        return 1

    store = FTS5Store(args.db_path, allow_fallback=args.allow_fallback)
    if not args.force and store.is_current_for(args.chunks_path):
        n = store.load()
        print(f"{args.db_path} is already current for {args.chunks_path} "
              f"({n:,} chunks) — nothing to do. Pass --force to rebuild anyway.")
        return 0

    chunks = load_chunks(args.chunks_path)
    print(f"loaded {len(chunks):,} chunks from {args.chunks_path}")

    t0 = time.monotonic()
    n = store.build(chunks, source_path=args.chunks_path)
    elapsed = time.monotonic() - t0

    size_mb = args.db_path.stat().st_size / 1024**2
    print(f"indexed {n:,} chunks -> {args.db_path} "
          f"({size_mb:.1f} MB on disk) in {elapsed:.1f}s")
    print(f"  tokenizer: {store.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
