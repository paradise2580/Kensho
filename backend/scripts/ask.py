#!/usr/bin/env python3
"""End-to-end query against the built index — the phase 1 control arm.

    python scripts/ask.py "How does a Pod's restart policy work?"
    python scripts/ask.py "Pod の再起動ポリシーはどう動作しますか?" --lang ja

Retrieves top-k chunks and makes one ungrounded-verification LLM call.
No claim checking yet — that starts in phase 4. Answers here should be read
as "does the baseline work at all," not as trustworthy output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.embed import DEFAULT_MODEL, get_embedder  # noqa: E402
from kensho.llm import get_llm  # noqa: E402
from kensho.pipeline import ask  # noqa: E402
from kensho.store import ChunkStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--index-dir", type=Path, default=Path("data/qdrant"))
    ap.add_argument("--embedder", default=DEFAULT_MODEL)
    ap.add_argument("--llm", default="groq", help="'groq' or 'echo' for the offline stand-in")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--lang", choices=["en", "ja"], default=None,
                    help="restrict retrieval to one language; omit to search both")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="allow the fake embedder / echo LLM if the real ones "
                         "can't load — for smoke-testing the wiring only")
    args = ap.parse_args()

    embedder = get_embedder(
        None if args.embedder == "fake" else args.embedder,
        allow_fallback=args.allow_fallback,
    )
    store = ChunkStore(embedder, path=args.index_dir)
    llm = get_llm(None if args.llm == "echo" else args.llm, allow_fallback=args.allow_fallback)

    print(f"embedder={embedder.name}  llm={llm.name}  index has {store.count()} chunks\n")

    answer = ask(args.question, store, llm, top_k=args.top_k, lang=args.lang)

    print(f"Q: {answer.question}\n")
    print(f"A: {answer.text}\n")
    print(f"Sources ({len(answer.sources)}):")
    for i, hit in enumerate(answer.sources, start=1):
        title = hit.payload.get("title", "")
        section = " › ".join(hit.payload.get("section_path", []))
        print(f"  [{i}] score={hit.score:.3f}  {title} › {section}  {hit.payload.get('url', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
