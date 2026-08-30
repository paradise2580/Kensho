"""Structured per-request traces, appended to a JSONL sink.

A trace exists for two reasons that both matter once this is a running
service rather than a script someone runs by hand: a support engineer
answering "why did it say that" needs to see the retrieved chunks, the raw
generation, and every claim's verdict without re-running the query, and a
future eval round can mine real production questions as gold-set
candidates precisely because their retrieval and verification results were
captured at the time, not reconstructed after the fact from a log line of
just the final answer.

Unlike almost everything else pluggable in this project, ``TraceWriter``
needs no model and no network — it's file I/O — so there's no fallback
story to tell for it; it's simply always real.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


def new_trace_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    parallel_id: str
    title: str
    lang: str
    score: float


@dataclass(frozen=True, slots=True)
class ClaimTrace:
    text: str
    label: str
    action: str
    method: str


@dataclass(frozen=True, slots=True)
class Trace:
    trace_id: str
    timestamp: float
    question: str
    lang_filter: str | None
    retrieved: list[RetrievedChunk]
    answer_text: str
    llm_name: str
    embedder_name: str
    verifier_name: str
    claims: list[ClaimTrace]
    final_text: str
    refused: bool
    refusal_reason: str | None
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class TraceWriter:
    """Appends one JSON object per line. Safe to share across requests in a
    single-process server — each write is one atomic ``open(... "a")``
    append, which is as far as this project's concurrency story needs to go
    for a demo-scale deployment (see the plan's deployment notes on
    Hugging Face Spaces' single-process constraints)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, trace: Trace) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def read_traces(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Timer:
    """Tiny context-manager stopwatch — ``with Timer() as t: ...; t.ms``.

    Exists so latency breakdowns in a trace read as
    ``{"retrieval": t1.ms, "generation": t2.ms, ...}`` instead of manual
    ``time.perf_counter()`` arithmetic repeated at every call site.
    """

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000
