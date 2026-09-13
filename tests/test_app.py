"""App tests via FastAPI's TestClient — no real network server is started,
and every model is the fake/echo stand-in, so these run fully offline."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kensho.embed import get_embedder
from kensho.schema import Chunk
from kensho.serve.app import create_app
from kensho.serve.config import ServerConfig
from kensho.serve.trace import read_traces
from kensho.store import ChunkStore


def make_chunk(text: str, parallel_id: str, lang: str, ordinal: int = 0) -> Chunk:
    doc_id = f"{lang}:{parallel_id}"
    return Chunk(
        chunk_id=Chunk.make_id(doc_id, ordinal, "structural"),
        doc_id=doc_id, parallel_id=parallel_id, lang=lang, title=parallel_id,
        section_path=(), text=text, token_count=len(text.split()), char_count=len(text),
        ordinal=ordinal, url=f"https://example.invalid/{parallel_id}",
        strategy="structural", commit_sha="deadbeef",
    )


@pytest.fixture
def seeded_index_dir(tmp_path):
    index_dir = tmp_path / "qdrant"
    embedder = get_embedder(None)  # FakeEmbedder, dim=64 — matches config below
    store = ChunkStore(embedder, path=index_dir)
    store.upsert([make_chunk("A Pod can hold several containers.", "docs/pods.md", "en", 0)])
    return index_dir


@pytest.fixture
def client(seeded_index_dir, tmp_path):
    config = ServerConfig(
        index_dir=seeded_index_dir, embedder_model=None, llm_provider=None,
        verifier_method="lexical", decomposer_mode="sentence",
        allow_fallback=True, trace_path=tmp_path / "traces.jsonl",
    )
    app = create_app(config)
    return TestClient(app), config


class TestHealthz:
    def test_reports_ok_with_loaded_index(self, client):
        test_client, _config = client
        resp = test_client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["index_size"] == 1
        assert body["embedder"] == "fake"
        assert body["llm"] == "echo"
        assert body["verifier"] == "lexical-overlap"


class TestAsk:
    def test_returns_full_response_shape(self, client):
        test_client, _config = client
        resp = test_client.post("/ask", json={"question": "What is a Pod?"})
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"trace_id", "answer", "refused", "refusal_reason",
                             "claims", "sources", "latency_ms"}
        assert body["trace_id"]
        assert len(body["sources"]) == 1
        assert body["sources"][0]["parallel_id"] == "docs/pods.md"

    def test_echo_llm_with_no_real_answer_gets_refused(self, client):
        """EchoProvider's placeholder text shares no vocabulary with the
        retrieved chunk, so lexical-overlap verification finds nothing
        supported and the policy refuses — the same behaviour proven at the
        CLI layer in phase 5, now proven through the HTTP boundary too."""
        test_client, _config = client
        resp = test_client.post("/ask", json={"question": "What is a Pod?"})
        body = resp.json()
        assert body["refused"] is True
        assert body["refusal_reason"] is not None

    def test_writes_a_trace_for_every_request(self, client):
        test_client, config = client
        test_client.post("/ask", json={"question": "First question?"})
        test_client.post("/ask", json={"question": "Second question?"})
        traces = read_traces(config.trace_path)
        assert len(traces) == 2
        assert traces[0]["question"] == "First question?"
        assert traces[1]["question"] == "Second question?"

    def test_trace_id_in_response_matches_written_trace(self, client):
        test_client, config = client
        resp = test_client.post("/ask", json={"question": "What is a Pod?"})
        trace_id = resp.json()["trace_id"]
        traces = read_traces(config.trace_path)
        assert traces[-1]["trace_id"] == trace_id

    def test_latency_breakdown_has_expected_keys(self, client):
        test_client, _config = client
        resp = test_client.post("/ask", json={"question": "What is a Pod?"})
        latency = resp.json()["latency_ms"]
        assert set(latency) == {"retrieval_and_generation", "verification",
                                "correction", "total"}

    def test_top_k_override_is_respected(self, client):
        test_client, _config = client
        resp = test_client.post("/ask", json={"question": "What is a Pod?", "top_k": 1})
        assert len(resp.json()["sources"]) <= 1

    def test_missing_question_is_a_validation_error(self, client):
        test_client, _config = client
        resp = test_client.post("/ask", json={})
        assert resp.status_code == 422


@pytest.fixture
def chunks_file(tmp_path):
    """A small parallel EN/JA chunk file for BM25 to index at startup.

    Deliberately six chunks across three topics rather than two: BM25 scores
    on inverse document frequency, and in a two-document corpus every term
    appears in nearly every document, so IDF collapses and all scores tie at
    zero. A fixture that small would test the plumbing while telling us
    nothing about whether ranking works.
    """
    import json
    path = tmp_path / "chunks.jsonl"
    chunks = [
        make_chunk("A Pod can hold several containers that share a network namespace.",
                   "docs/pods.md", "en", 0),
        make_chunk("Podは同じネットワーク名前空間を共有する複数のコンテナを保持できます。",
                   "docs/pods.md", "ja", 0),
        make_chunk("A Service exposes an application running on a set of Pods.",
                   "docs/services.md", "en", 0),
        make_chunk("Serviceは一連のPod上で動作するアプリケーションを公開します。",
                   "docs/services.md", "ja", 0),
        make_chunk("A volume provides persistent storage that outlives a container restart.",
                   "docs/volumes.md", "en", 0),
        make_chunk("ボリュームはコンテナの再起動後も残る永続的なストレージを提供します。",
                   "docs/volumes.md", "ja", 0),
    ]
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def sparse_client(chunks_file, tmp_path):
    config = ServerConfig(
        retriever="sparse", chunks_path=chunks_file,
        embedder_model=None, llm_provider=None, verifier_method="lexical",
        allow_fallback=True, trace_path=tmp_path / "sparse-traces.jsonl",
    )
    return TestClient(create_app(config)), config


class TestSparseRetrieval:
    """BM25 serving is the configuration a host with no model storage runs.

    These assert the two things that make that deployment worth having: no
    embedding model is loaded at all, and retrieval still returns real hits.
    """

    def test_healthz_reports_bm25_and_no_embedder(self, sparse_client):
        test_client, _config = sparse_client
        body = test_client.get("/healthz").json()
        assert body["status"] == "ok"
        assert body["retriever"].startswith("bm25:")
        assert body["embedder"] == "none"
        assert body["index_size"] == 6

    def test_ask_retrieves_without_any_embedder(self, sparse_client):
        test_client, _config = sparse_client
        resp = test_client.post("/ask", json={"question": "Can a Pod hold several containers?"})
        assert resp.status_code == 200
        assert resp.json()["sources"], "BM25 returned no sources for a lexically obvious query"

    def test_japanese_query_ranks_the_right_japanese_chunk_first(self, sparse_client):
        """Segmentation is the whole game here.

        Japanese has no word boundaries, so a whitespace tokenizer would see
        this query as one enormous token matching nothing. Getting the right
        document back is evidence the morphological segmenter actually ran.
        """
        test_client, _config = sparse_client
        resp = test_client.post("/ask", json={"question": "ボリュームとは何ですか?"})
        sources = resp.json()["sources"]
        assert sources, "BM25 returned nothing for a Japanese query"
        assert sources[0]["lang"] == "ja"
        assert sources[0]["parallel_id"] == "docs/volumes.md"


class TestRetrieverValidation:
    def test_unknown_retriever_fails_loudly(self, tmp_path):
        from kensho.serve.app import build_retriever
        config = ServerConfig(retriever="magic")
        with pytest.raises(ValueError, match="Unknown KENSHO_RETRIEVER"):
            build_retriever(config, embedder=None)

    def test_sparse_without_chunk_file_names_the_missing_path(self, tmp_path):
        from kensho.serve.app import build_retriever
        missing = tmp_path / "nope.jsonl"
        config = ServerConfig(retriever="sparse", chunks_path=missing)
        with pytest.raises(FileNotFoundError, match="nope.jsonl"):
            build_retriever(config, embedder=None)
