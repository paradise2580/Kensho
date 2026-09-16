"""Core data structures for the Kenshō corpus.

A Chunk is the atomic retrievable unit. Everything downstream — retrieval,
generation, claim verification — refers to chunks by ``chunk_id``.

The one field that earns its keep beyond the obvious: ``parallel_id``. It is
the language-independent identity of a page (its path inside the docs tree),
so ``parallel_id`` links the Japanese and English versions of the same page.
Cross-lingual evaluation is only possible because that link exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Literal

Lang = Literal["en", "ja"]


@dataclass(frozen=True, slots=True)
class Document:
    """One source markdown page.

    ``body`` holds the **raw** markdown, front matter included. Parsing happens
    at chunk time so that a single loaded corpus can be re-chunked under
    different strategies without re-reading the filesystem — which is exactly
    what ablation 3.2 does.
    """

    parallel_id: str  # e.g. "docs/concepts/workloads/pods/_index.md"
    lang: Lang
    title: str
    url: str
    body: str
    commit_sha: str

    @property
    def doc_id(self) -> str:
        return f"{self.lang}:{self.parallel_id}"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable span of one document."""

    chunk_id: str
    doc_id: str
    parallel_id: str
    lang: Lang
    title: str
    section_path: tuple[str, ...]  # heading hierarchy, outermost first
    text: str
    token_count: int
    char_count: int
    ordinal: int  # position within the document
    url: str
    strategy: str  # which chunker produced this — ablation 3.2 needs it
    commit_sha: str
    meta: dict = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, ordinal: int, strategy: str) -> str:
        raw = f"{doc_id}|{strategy}|{ordinal}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def heading_trail(self) -> str:
        """Human-readable breadcrumb, prepended to text before embedding.

        Retrieval quality improves measurably when a chunk carries its own
        context; a bare paragraph about "the default value" is meaningless
        without knowing which setting it belongs to.
        """
        return " › ".join((self.title, *self.section_path))

    def for_embedding(self) -> str:
        return f"{self.heading_trail}\n\n{self.text}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["section_path"] = list(self.section_path)
        return d
