#!/usr/bin/env python3
"""Embed a chunk JSONL file into the Qdrant store.

    python scripts/build_index.py data/chunks/structural_t512_o64.jsonl

Reads the chunks written by ``build_corpus.py`` and writes their embeddings
to ``data/qdrant`` (or ``--index-dir``). This is a separate step from
chunking, not merged into it, because phase 3 re-embeds the same chunk file
under different retrieval configurations without re-chunking, and phase 1's
embedder is not phase 4's — decoupling the steps is what makes each one
independently swappable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.embed import DEFAULT_MODEL, get_embedder  # noqa: E402
from kensho.schema import Chunk  # noqa: E402
from kensho.store import ChunkStore  # noqa: E402


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
    ap.add_argument("chunks_path", type=Path)
    ap.add_argument("--index-dir", type=Path, default=Path("data/qdrant"))
    ap.add_argument("--embedder", default=DEFAULT_MODEL,
                    help="HF model id, or 'fake' for the offline plumbing stand-in")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="fall back to the fake embedder instead of failing "
                         "when the real model can't load — plumbing only, "
                         "never for a run whose numbers you intend to report")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks_path)
    print(f"loaded {len(chunks)} chunks from {args.chunks_path}")

    embedder = get_embedder(
        None if args.embedder == "fake" else args.embedder,
        allow_fallback=args.allow_fallback,
    )
    print(f"embedder: {embedder.name}  dim={embedder.dim}")

    store = ChunkStore(embedder, path=args.index_dir)
    n = store.upsert(chunks, batch_size=args.batch_size)
    print(f"wrote {n} vectors -> {args.index_dir}  (collection now has {store.count()})")
    if embedder.name == "fake":
        print("  NOTE: fake embedder — this index is plumbing only, not for retrieval eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
