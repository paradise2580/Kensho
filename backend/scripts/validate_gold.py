#!/usr/bin/env python3
"""Check that a gold set's relevant_parallel_ids still exist in a built corpus.

    python scripts/validate_gold.py data/gold/gold_v1.jsonl data/chunks/structural_t512_o64.jsonl

The Kubernetes docs this project pins are a live upstream repo; a future
``--ref`` could move a page, rename it, or drop it. A gold set that silently
stops matching real pages would make every retrieval score look worse for a
reason that has nothing to do with the retriever — the same trap the
``docs/reference/`` exclusion in ``fetch.py`` was built to avoid. Run this
after re-pinning the corpus, before trusting any eval number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.eval.gold import load_gold  # noqa: E402


def parallel_ids_in_corpus(chunks_path: Path) -> set[str]:
    ids: set[str] = set()
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            ids.add(json.loads(line)["parallel_id"])
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold_path", type=Path)
    ap.add_argument("chunks_path", type=Path)
    args = ap.parse_args()

    gold = load_gold(args.gold_path)
    corpus_ids = parallel_ids_in_corpus(args.chunks_path)

    missing: list[tuple[str, str]] = []
    for item in gold:
        for pid in item.relevant_parallel_ids:
            if pid not in corpus_ids:
                missing.append((item.id, pid))

    print(f"{len(gold)} gold items, {len(corpus_ids)} distinct pages in {args.chunks_path.name}")
    if missing:
        print(f"STALE: {len(missing)} relevant_parallel_ids no longer found in the corpus:")
        for gold_id, pid in missing:
            print(f"  {gold_id}: {pid}")
        return 1
    print("all relevant_parallel_ids resolve to real pages in this corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
