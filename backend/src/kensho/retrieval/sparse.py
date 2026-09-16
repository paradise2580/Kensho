"""Lexical retrieval via BM25 — an honest sparse baseline, in-memory.

Kept in-memory rather than persisted, unlike ``ChunkStore``: rebuilding a
BM25 index from a chunk JSONL takes seconds even at this corpus's size, so
there is no operational reason to add a second on-disk format to keep in
sync with the source chunks.

One limitation is worth stating plainly rather than discovering by surprise
in ablation 3.7's results: BM25 matches literal terms, so a Japanese query
cannot retrieve an English-only passage (or vice versa) no matter how
relevant it is — there is no shared vocabulary for the terms to match
against. Dense embedding search is the only one of this project's
retrievers that can genuinely cross the language boundary; measuring that
gap, not papering over it, is the point of comparing them.
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ..schema import Chunk
from ..store import SearchHit
from ..tokenizer import script_of
from .segment import Segmenter, get_segmenter


@dataclass(slots=True)
class _Indexed:
    chunk: Chunk
    tokens: list[str]


class BM25Store:
    """A ``Retriever`` backed by ``rank_bm25.BM25Okapi``.

    Each chunk is tokenized with the segmenter for *its own* language
    (``chunk.lang``); each query is tokenized with the segmenter inferred
    from the query's script via ``tokenizer.script_of``. Mixing tokenizers
    within one corpus would be the same mistake the token-counting code
    warns about — one ratio for two scripts corrupts the comparison.
    """

    def __init__(self, allow_fallback: bool = True) -> None:
        self._segmenters: dict[str, Segmenter] = {
            "ja": get_segmenter("ja", allow_fallback=allow_fallback),
            "en": get_segmenter("en", allow_fallback=allow_fallback),
        }
        self._items: list[_Indexed] = []
        self._bm25: BM25Okapi | None = None

    @property
    def name(self) -> str:
        ja_name = self._segmenters["ja"].name
        return f"bm25:{ja_name}"

    def _segmenter_for_lang(self, lang: str) -> Segmenter:
        return self._segmenters.get(lang, self._segmenters["en"])

    def index(self, chunks: list[Chunk]) -> int:
        self._items = [
            _Indexed(chunk=c, tokens=self._segmenter_for_lang(c.lang).segment(c.text))
            for c in chunks
        ]
        self._bm25 = BM25Okapi([it.tokens for it in self._items]) if self._items else None
        return len(self._items)

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]:
        if self._bm25 is None:
            return []
        query_lang = "ja" if script_of(query) == "cjk" else "en"
        tokens = self._segmenter_for_lang(query_lang).segment(query)
        scores = self._bm25.get_scores(tokens)

        candidates = range(len(self._items))
        if lang is not None:
            candidates = [i for i in candidates if self._items[i].chunk.lang == lang]

        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchHit(
                chunk_id=self._items[i].chunk.chunk_id,
                score=float(scores[i]),
                payload=_payload(self._items[i].chunk),
            )
            for i in ranked
        ]

    def count(self) -> int:
        return len(self._items)


def _payload(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "parallel_id": c.parallel_id,
        "lang": c.lang,
        "title": c.title,
        "section_path": list(c.section_path),
        "text": c.text,
        "url": c.url,
        "strategy": c.strategy,
    }
