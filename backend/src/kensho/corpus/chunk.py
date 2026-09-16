"""Chunking strategies.

Two are implemented because ablation 3.2 compares them:

``fixed``       — a sliding token window over the whole page, ignoring
                  structure. The conventional baseline.
``structural``  — respects heading boundaries: text is packed into chunks
                  within a section and never spans two sections. Oversized
                  sections are split at sentence boundaries, undersized ones
                  merged with their neighbours.

The interesting hypothesis is that ``structural`` wins on precision (a chunk
is about exactly one thing) and loses on recall for questions whose answer
straddles two sections. Measuring that tradeoff is the point.

Sentence splitting is script-aware: Japanese terminates sentences with 。！？
and does not use spaces, so a splitter written for English produces one
enormous "sentence" per Japanese paragraph and the overlap logic degenerates.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

from ..schema import Chunk, Document
from ..tokenizer import TokenCounter
from .parse import Block, parse_markdown

# Japanese terminators first; the trailing group keeps closing brackets attached.
SENTENCE_END = re.compile(r"(?<=[。．！？!?])\s*|(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_END.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _pack(units: Sequence[str], counter: TokenCounter, target: int,
          overlap: int, joiner: str) -> Iterator[tuple[int, int, str]]:
    """Greedily pack units into ~target-token groups with token overlap.

    Yields ``(start_index, end_index, text)`` so callers can map a chunk back
    to the units it came from — needed to attribute a section path.
    """
    if not units:
        return
    sizes = [counter.count(u) for u in units]
    i = 0
    n = len(units)
    while i < n:
        total = 0
        j = i
        while j < n and (total + sizes[j] <= target or j == i):
            total += sizes[j]
            j += 1
        yield i, j, joiner.join(units[i:j])
        if j >= n:
            return
        if overlap <= 0:
            i = j
            continue
        # Step back far enough to carry ~overlap tokens into the next chunk,
        # but always make forward progress or this loops forever.
        back = 0
        k = j
        while k > i + 1 and back < overlap:
            k -= 1
            back += sizes[k]
        i = k if k > i else j


def chunk_structural(doc: Document, blocks: Sequence[Block], counter: TokenCounter,
                     target_tokens: int = 512, overlap_tokens: int = 64,
                     min_tokens: int = 32) -> list[Chunk]:
    """Heading-aware chunking. Chunks never span a section boundary."""
    grouped: list[tuple[tuple[str, ...], list[str]]] = []
    for b in blocks:
        if grouped and grouped[-1][0] == b.section_path:
            grouped[-1][1].append(b.text)
        else:
            grouped.append((b.section_path, [b.text]))

    out: list[Chunk] = []
    carry: tuple[tuple[str, ...], str] | None = None
    ordinal = 0

    for section_path, texts in grouped:
        body = "\n\n".join(texts)
        if carry is not None:
            # Previous section was too small to stand alone; prepend it and
            # label the result with the deepest heading that actually contains
            # both. Picking either original path would mis-attribute the text
            # to a section it did not come from.
            body = carry[1] + "\n\n" + body
            section_path = _common_prefix(carry[0], section_path)
            carry = None

        if counter.count(body) < min_tokens:
            carry = (section_path, body)
            continue

        units = split_sentences(body)
        for _s, _e, piece in _pack(units, counter, target_tokens, overlap_tokens, " "):
            tc = counter.count(piece)
            if tc < min_tokens:
                continue
            out.append(_mk(doc, section_path, piece, tc, ordinal, "structural"))
            ordinal += 1

    if carry is not None:
        if out:
            # Trailing scrap: attach to the last chunk rather than dropping
            # content, but re-label to the common ancestor. Keeping the last
            # chunk's own path would claim, for example, that text from
            # section A/B2 lives under A/B1 — a sibling that does not contain
            # it. Retrieval would then cite the wrong section.
            last = out[-1]
            merged = last.text + "\n\n" + carry[1]
            out[-1] = _mk(doc, _common_prefix(last.section_path, carry[0]), merged,
                          counter.count(merged), last.ordinal, "structural")
        else:
            # Whole page is shorter than min_tokens. Emitting it undersized
            # beats silently discarding the document.
            out.append(_mk(doc, carry[0], carry[1], counter.count(carry[1]), 0,
                           "structural"))
    return out


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    """Deepest heading path containing both sections."""
    out: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return tuple(out)


def chunk_fixed(doc: Document, blocks: Sequence[Block], counter: TokenCounter,
                target_tokens: int = 512, overlap_tokens: int = 64,
                min_tokens: int = 32) -> list[Chunk]:
    """Sliding window over the full page, ignoring headings.

    Section path is recorded as the path of the block the chunk starts in, so
    the metadata stays populated for comparability, but it does not constrain
    the boundaries.
    """
    units: list[str] = []
    paths: list[tuple[str, ...]] = []
    for b in blocks:
        for s in split_sentences(b.text):
            units.append(s)
            paths.append(b.section_path)

    out: list[Chunk] = []
    ordinal = 0
    for start, _end, piece in _pack(units, counter, target_tokens, overlap_tokens, " "):
        tc = counter.count(piece)
        if tc < min_tokens:
            continue
        path = paths[start] if paths else ()
        out.append(_mk(doc, path, piece, tc, ordinal, "fixed"))
        ordinal += 1
    return out


def _mk(doc: Document, section_path: tuple[str, ...], text: str, tc: int,
        ordinal: int, strategy: str) -> Chunk:
    return Chunk(
        chunk_id=Chunk.make_id(doc.doc_id, ordinal, strategy),
        doc_id=doc.doc_id,
        parallel_id=doc.parallel_id,
        lang=doc.lang,
        title=doc.title,
        section_path=section_path,
        text=text,
        token_count=tc,
        char_count=len(text),
        ordinal=ordinal,
        url=doc.url,
        strategy=strategy,
        commit_sha=doc.commit_sha,
    )


STRATEGIES = {"structural": chunk_structural, "fixed": chunk_fixed}


def chunk_document(doc: Document, counter: TokenCounter, strategy: str = "structural",
                   **kw) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(STRATEGIES)}")
    page = parse_markdown(doc.body, fallback_title=doc.title)
    return STRATEGIES[strategy](doc, page.blocks, counter, **kw)
