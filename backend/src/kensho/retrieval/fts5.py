"""Lexical retrieval via SQLite FTS5 — disk-backed, so the index is not a
Python object living in RSS for the life of the process.

``BM25Store`` (``sparse.py``) keeps every chunk's text and token list as
live Python objects plus a ``rank_bm25.BM25Okapi`` postings structure —
measured at ~390MB resident for this project's 8,446-chunk corpus. That is
fine on a host with RAM to spare and costly on a free tier sized for a demo.
This module is the same retrieval capability with a different storage
decision: the inverted index lives in a SQLite file, the OS page-caches
whatever pages a query actually touches, and the Python process holds only
a connection handle and (still, unavoidably) the segmenter's dictionary.

**Tokenization is the one part of this that cannot just be "give the text to
FTS5".** FTS5's own tokenizers were checked against this corpus's Japanese
text before writing a line of index-building code here:

- ``unicode61`` (the default) treats an unbroken run of CJK characters as a
  single token — Japanese has no whitespace between words, so a query for
  a single word inside a sentence matches nothing. Confirmed empirically,
  not assumed.
- ``trigram`` does match, but it is a character-n-gram scheme — the same
  category of approximation ``CharBigramSegmenter`` already is in
  ``segment.py``, and that module's docstring is explicit that character
  n-grams are "a lower bound, not a result to publish."

So neither built-in tokenizer is used. Instead, every chunk's text is
segmented with the project's *existing* ``Segmenter`` — the real SudachiPy
mode-C tokenizer for Japanese, unchanged from ``sparse.py`` — and the
resulting tokens are stored space-joined in a second column. FTS5 indexes
that column with plain ``unicode61``, which just needs to split on the
spaces already put there. Segmentation quality is therefore identical to
the in-memory path by construction, not by approximation: same segmenter,
same tokens, different sink.

**Scoring does not use FTS5's built-in ``bm25()``.** It was tried first —
the module used to rank with it — and a recall@5/MRR parity run against
``BM25Store`` on both gold sets showed a real, consistent regression
(-0.034 to -0.042 recall@5), root-caused to SQLite's ``bm25()`` being
hardcoded to k1=1.2 with no public API to request k1=1.5 the way
``rank_bm25.BM25Okapi`` (this project's chosen constants) does. The
mismatch reordered close-scoring candidates rather than causing outright
misses — recall@10 was identical between backends, only the ranking below
that differed — which confirmed it was exactly this constant, not a
tokenization or corpus bug.

So this module computes Okapi BM25 itself, with this project's own k1/b/
epsilon, using FTS5 purely as a fast disk-backed inverted index for
*candidate generation* (which documents contain at least one query term —
exactly the documents that can possibly score above zero) and three small
auxiliary tables for the numbers the formula actually needs:

- ``doc_stats``     — token count per chunk (the ``|D|`` in the formula)
- ``term_freq``     — how many times each term occurs in each chunk (``tf``)
- ``term_df``       — each term's *already epsilon-floored* idf, precomputed
                      once at build time exactly as ``BM25Okapi._calc_idf``
                      computes it (same floor: negative idf — a term in
                      more than half the corpus — is clamped to
                      ``epsilon * average_idf`` rather than left negative)

The result reproduces ``rank_bm25.BM25Okapi.get_scores`` bit-for-bit (up to
floating point summation order) rather than approximating it: same k1, b,
epsilon, same idf formula and floor, same score formula, same corpus
statistics. The only thing that differs from the in-memory backend is
where the numbers live between requests.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..schema import Chunk
from ..store import SearchHit
from ..tokenizer import script_of
from .segment import Segmenter, get_segmenter

# Matches rank_bm25.BM25Okapi's defaults exactly — see sparse.py, which
# constructs BM25Store() with no override and therefore also runs at these
# values. Changing these here without changing them there would silently
# reintroduce the drift this module exists to eliminate.
K1 = 1.5
B = 0.75
EPSILON = 0.25

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id     UNINDEXED,
    lang         UNINDEXED,
    parallel_id  UNINDEXED,
    title        UNINDEXED,
    section_path UNINDEXED,
    url          UNINDEXED,
    strategy     UNINDEXED,
    text         UNINDEXED,
    body_tokens,
    tokenize = 'unicode61'
);

-- Auxiliary tables for exact Okapi scoring — see module docstring for why
-- FTS5's own bm25() isn't used. rowid here is the same integer assigned to
-- the matching row in chunks_fts (set explicitly at insert time), so a
-- candidate rowid from an FTS5 MATCH joins straight into these.
CREATE TABLE doc_stats (
    rowid   INTEGER PRIMARY KEY,
    doc_len INTEGER NOT NULL
);

CREATE TABLE term_freq (
    term  TEXT NOT NULL,
    rowid INTEGER NOT NULL,
    tf    INTEGER NOT NULL,
    PRIMARY KEY (term, rowid)
) WITHOUT ROWID;

CREATE TABLE term_df (
    term TEXT PRIMARY KEY,
    idf  REAL NOT NULL
) WITHOUT ROWID;
"""


@dataclass(slots=True)
class _StaleCheck:
    source_path: str
    source_size: int
    chunk_count: int


class FTS5Store:
    """A ``Retriever`` backed by SQLite FTS5 for candidate generation and
    exact-formula Okapi BM25 scoring on top of it.

    Same segmenter-per-language contract as ``BM25Store`` — see that
    class's docstring for why chunk text and query text must use the
    segmenter matching *their own* language/script, not one shared
    tokenizer for both.
    """

    def __init__(self, db_path: Path, allow_fallback: bool = True) -> None:
        self._db_path = Path(db_path)
        self._segmenters: dict[str, Segmenter] = {
            "ja": get_segmenter("ja", allow_fallback=allow_fallback),
            "en": get_segmenter("en", allow_fallback=allow_fallback),
        }
        self._con: sqlite3.Connection | None = None
        self._count = 0
        self._avgdl = 0.0
        # FastAPI runs sync path-operation functions in a worker threadpool,
        # so search() can be entered from more than one thread concurrently.
        # A single sqlite3.Connection is not safe for that without either a
        # connection per thread or serializing access — this lock does the
        # latter, which is the right trade for a read-only, low-latency
        # query: contention here is microseconds, not worth a connection
        # pool for.
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"fts5:{self._segmenters['ja'].name}"

    def _segmenter_for_lang(self, lang: str) -> Segmenter:
        return self._segmenters.get(lang, self._segmenters["en"])

    # -- building -----------------------------------------------------

    def build(self, chunks: list[Chunk], source_path: Path) -> int:
        """Create the on-disk index fresh, overwriting any existing file.

        Three passes over the tokenized corpus, each doing one job:
        insert the searchable rows, then derive the term-level statistics
        (df -> idf) that need the *whole* corpus seen first, matching
        ``BM25Okapi._initialize`` / ``_calc_idf``'s own two-pass shape —
        document frequency can't be known until every document has been
        tokenized.
        """
        self.close()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path.unlink(missing_ok=True)

        con = sqlite3.connect(str(self._db_path), check_same_thread=False)
        con.executescript(SCHEMA)

        doc_freq: dict[str, int] = {}  # term -> number of docs containing it
        doc_len_total = 0

        for rowid, c in enumerate(chunks):
            tokens = self._segmenter_for_lang(c.lang).segment(c.text)
            con.execute(
                "INSERT INTO chunks_fts "
                "(rowid, chunk_id, lang, parallel_id, title, section_path, "
                " url, strategy, text, body_tokens) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (rowid, c.chunk_id, c.lang, c.parallel_id, c.title,
                 "/".join(c.section_path), c.url, c.strategy, c.text,
                 " ".join(tokens)),
            )
            con.execute(
                "INSERT INTO doc_stats VALUES (?, ?)", (rowid, len(tokens)),
            )
            doc_len_total += len(tokens)

            tf = Counter(tokens)
            con.executemany(
                "INSERT INTO term_freq VALUES (?, ?, ?)",
                ((term, rowid, count) for term, count in tf.items()),
            )
            for term in tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        corpus_size = len(chunks)
        avgdl = (doc_len_total / corpus_size) if corpus_size else 0.0

        # rank_bm25.BM25Okapi._calc_idf, reproduced exactly: compute raw idf
        # for every term, average it across the vocabulary, then floor any
        # negative idf (a term appearing in more than half the corpus) to
        # epsilon * average_idf rather than leaving it negative — a term
        # that common should contribute a small positive weight, not
        # actively penalise documents that contain it.
        raw_idf: dict[str, float] = {}
        idf_sum = 0.0
        negative_terms: list[str] = []
        for term, df in doc_freq.items():
            idf = math.log(corpus_size - df + 0.5) - math.log(df + 0.5)
            raw_idf[term] = idf
            idf_sum += idf
            if idf < 0:
                negative_terms.append(term)
        average_idf = (idf_sum / len(raw_idf)) if raw_idf else 0.0
        eps_floor = EPSILON * average_idf
        for term in negative_terms:
            raw_idf[term] = eps_floor

        con.executemany(
            "INSERT INTO term_df VALUES (?, ?)",
            ((term, idf) for term, idf in raw_idf.items()),
        )

        stale = _StaleCheck(
            # Resolved to an absolute, canonical path so identity depends
            # only on which file this is, not on how the caller happened to
            # spell its path — a build run with a relative chunks_path and
            # a server started with KENSHO_CHUNKS as an absolute one point
            # at the same file and must be recognised as current, not
            # trigger a silent full rebuild inside create_app() because the
            # strings differ. That rebuild is exactly what this method
            # exists to avoid.
            source_path=str(source_path.resolve()),
            source_size=source_path.stat().st_size,
            chunk_count=corpus_size,
        )
        con.execute("INSERT INTO meta VALUES ('stale_check', ?)",
                    (json.dumps({
                        "source_path": stale.source_path,
                        "source_size": stale.source_size,
                        "chunk_count": stale.chunk_count,
                    }),))
        con.execute("INSERT INTO meta VALUES ('bm25_stats', ?)",
                    (json.dumps({
                        "corpus_size": corpus_size, "avgdl": avgdl,
                        "k1": K1, "b": B, "epsilon": EPSILON,
                    }),))

        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        con.commit()

        self._con = con
        self._count = corpus_size
        self._avgdl = avgdl
        return self._count

    # -- loading --------------------------------------------------------

    def is_current_for(self, source_path: Path) -> bool:
        """Whether an on-disk index at ``self._db_path`` matches
        ``source_path`` well enough to load instead of rebuild.

        A byte-size comparison, not a content hash — see the module
        docstring on why this check stays cheap. False if the file is
        missing, unreadable, or predates the ``meta`` table this class
        writes (an index built some other way).
        """
        if not self._db_path.exists():
            return False
        try:
            con = sqlite3.connect(str(self._db_path))
            row = con.execute(
                "SELECT value FROM meta WHERE key = 'stale_check'"
            ).fetchone()
            con.close()
        except sqlite3.DatabaseError:
            return False
        if row is None:
            return False
        stale = json.loads(row[0])
        return (
            stale.get("source_path") == str(source_path.resolve())
            and stale.get("source_size") == source_path.stat().st_size
        )

    def load(self) -> int:
        """Open an existing on-disk index. Call ``is_current_for`` first."""
        self.close()
        con = sqlite3.connect(str(self._db_path), check_same_thread=False)
        row = con.execute("SELECT value FROM meta WHERE key = 'bm25_stats'").fetchone()
        if row is not None:
            stats = json.loads(row[0])
            self._count = stats["corpus_size"]
            self._avgdl = stats["avgdl"]
        else:
            self._count = con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            self._avgdl = 0.0
        self._con = con
        return self._count

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    # -- searching --------------------------------------------------------

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]:
        if self._con is None or self._count == 0:
            return []

        query_lang = "ja" if script_of(query) == "cjk" else "en"
        tokens = self._segmenter_for_lang(query_lang).segment(query)
        cleaned = [t.replace('"', "") for t in tokens if t.replace('"', "").strip()]
        if not cleaned:
            return []

        with self._lock:
            candidates = self._candidate_rowids(cleaned, lang)
            if not candidates:
                return []
            scores = self._score(cleaned, candidates)
            ranked = sorted(scores, key=lambda rid: scores[rid], reverse=True)[:top_k]
            rows = self._fetch_payloads(ranked)

        return [
            SearchHit(chunk_id=rows[rid]["chunk_id"], score=scores[rid],
                      payload=rows[rid])
            for rid in ranked
        ]

    def _candidate_rowids(self, tokens: list[str], lang: str | None) -> set[int]:
        """Every document containing at least one query term — exactly the
        documents ``rank_bm25`` would score above zero, since a term absent
        from a document contributes nothing to its score. Matching on this
        set rather than scanning every row is what makes this a fast
        disk-backed lookup instead of an O(corpus) scan per query."""
        match = " OR ".join(f'"{t}"' for t in set(tokens))
        sql = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
        params: list = [match]
        if lang is not None:
            sql += " AND lang = ?"
            params.append(lang)
        return {r[0] for r in self._con.execute(sql, params).fetchall()}

    def _score(self, tokens: list[str], candidates: set[int]) -> dict[int, float]:
        """``rank_bm25.BM25Okapi.get_scores``, reproduced term by term.

        Duplicate tokens in the query are deliberately NOT deduplicated
        before this loop — ``BM25Okapi.get_scores`` iterates the raw query
        token list and adds each occurrence's contribution again, so a
        repeated term really does count twice there, and matching that
        exactly (not "improving" on it) is the point of this module.
        """
        placeholders = ",".join("?" * len(candidates))
        doc_len = dict(self._con.execute(
            f"SELECT rowid, doc_len FROM doc_stats WHERE rowid IN ({placeholders})",
            list(candidates),
        ).fetchall())

        scores: dict[int, float] = dict.fromkeys(candidates, 0.0)
        for term in tokens:
            idf_row = self._con.execute(
                "SELECT idf FROM term_df WHERE term = ?", (term,),
            ).fetchone()
            if idf_row is None:
                continue  # term never seen at build time -> contributes 0
            idf = idf_row[0]

            tf_rows = self._con.execute(
                f"SELECT rowid, tf FROM term_freq WHERE term = ? AND rowid IN "
                f"({placeholders})",
                [term, *candidates],
            ).fetchall()
            for rowid, tf in tf_rows:
                dl = doc_len.get(rowid, self._avgdl)
                denom = tf + K1 * (1 - B + B * dl / self._avgdl) if self._avgdl else tf
                scores[rowid] += idf * (tf * (K1 + 1)) / denom if denom else 0.0
        return scores

    def _fetch_payloads(self, rowids: list[int]) -> dict[int, dict]:
        if not rowids:
            return {}
        placeholders = ",".join("?" * len(rowids))
        rows = self._con.execute(
            f"SELECT rowid, chunk_id, lang, parallel_id, title, section_path, "
            f"url, strategy, text FROM chunks_fts WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        return {
            r[0]: {
                "chunk_id": r[1], "lang": r[2], "parallel_id": r[3], "title": r[4],
                "section_path": r[5].split("/") if r[5] else [],
                "url": r[6], "strategy": r[7], "text": r[8],
            }
            for r in rows
        }

    def count(self) -> int:
        return self._count


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            d["section_path"] = tuple(d["section_path"])
            chunks.append(Chunk(**d))
    return chunks


def build_or_load(
    db_path: Path, chunks_path: Path, allow_fallback: bool = True,
) -> FTS5Store:
    """The one entry point ``serve/app.py`` needs: an ``FTS5Store`` that is
    either loaded from an already-current on-disk index, or built fresh
    from ``chunks_path`` and persisted to ``db_path`` for next time —
    mirroring the marker-check ``docker/entrypoint.sh`` already does for
    the dense index, so a container restart with an unchanged corpus never
    pays the index-build cost twice.
    """
    store = FTS5Store(db_path, allow_fallback=allow_fallback)
    if store.is_current_for(chunks_path):
        store.load()
        return store
    chunks = load_chunks(chunks_path)
    store.build(chunks, source_path=chunks_path)
    return store
