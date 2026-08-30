from __future__ import annotations

import time

from kensho.serve.trace import ClaimTrace, RetrievedChunk, Timer, Trace, TraceWriter, read_traces


def make_trace(trace_id: str = "abc123") -> Trace:
    return Trace(
        trace_id=trace_id, timestamp=1234.5, question="What is a Pod?",
        lang_filter=None,
        retrieved=[RetrievedChunk(chunk_id="c1", parallel_id="docs/pods.md",
                                  title="Pods", lang="en", score=0.9)],
        answer_text="A Pod is...", llm_name="stub", embedder_name="stub-embed",
        verifier_name="lexical-overlap",
        claims=[ClaimTrace(text="A Pod is...", label="supported", action="kept",
                           method="lexical-overlap")],
        final_text="A Pod is...", refused=False, refusal_reason=None,
        latency_ms={"total": 12.3},
    )


class TestTraceWriter:
    def test_write_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "traces.jsonl"
        writer = TraceWriter(path)
        writer.write(make_trace())
        assert path.exists()

    def test_write_appends_one_line_per_trace(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        writer = TraceWriter(path)
        writer.write(make_trace("a"))
        writer.write(make_trace("b"))
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_round_trips_through_read_traces(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        writer = TraceWriter(path)
        writer.write(make_trace("xyz"))
        traces = read_traces(path)
        assert len(traces) == 1
        assert traces[0]["trace_id"] == "xyz"
        assert traces[0]["claims"][0]["label"] == "supported"

    def test_read_traces_on_missing_file_is_empty(self, tmp_path):
        assert read_traces(tmp_path / "does-not-exist.jsonl") == []


class TestTimer:
    def test_measures_nonzero_elapsed_time(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.ms > 0

    def test_ms_available_after_context_exits(self):
        with Timer() as t:
            pass
        assert isinstance(t.ms, float)
