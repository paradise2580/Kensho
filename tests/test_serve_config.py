from __future__ import annotations

from pathlib import Path

from kensho.serve.config import ServerConfig


class TestServerConfigDefaults:
    def test_defaults_point_at_a_real_embedder_and_groq(self):
        config = ServerConfig()
        assert config.embedder_model is not None
        assert config.llm_provider == "groq"
        assert config.allow_fallback is False


class TestFromEnv:
    def test_reads_index_dir(self, monkeypatch):
        monkeypatch.setenv("KENSHO_INDEX_DIR", "/tmp/some-index")
        config = ServerConfig.from_env()
        assert config.index_dir == Path("/tmp/some-index")

    def test_fake_and_echo_sentinels_map_to_none(self, monkeypatch):
        monkeypatch.setenv("KENSHO_EMBEDDER", "fake")
        monkeypatch.setenv("KENSHO_LLM", "echo")
        config = ServerConfig.from_env()
        assert config.embedder_model is None
        assert config.llm_provider is None

    def test_allow_fallback_parses_truthy_strings(self, monkeypatch):
        monkeypatch.setenv("KENSHO_ALLOW_FALLBACK", "true")
        assert ServerConfig.from_env().allow_fallback is True

    def test_allow_fallback_defaults_false(self, monkeypatch):
        monkeypatch.delenv("KENSHO_ALLOW_FALLBACK", raising=False)
        assert ServerConfig.from_env().allow_fallback is False

    def test_numeric_fields_are_parsed(self, monkeypatch):
        monkeypatch.setenv("KENSHO_TOP_K", "8")
        monkeypatch.setenv("KENSHO_REFUSAL_THRESHOLD", "0.75")
        config = ServerConfig.from_env()
        assert config.top_k == 8
        assert config.refusal_threshold == 0.75

    def test_unset_vars_fall_back_to_defaults(self, monkeypatch):
        for var in ("KENSHO_INDEX_DIR", "KENSHO_EMBEDDER", "KENSHO_LLM", "KENSHO_VERIFIER"):
            monkeypatch.delenv(var, raising=False)
        config = ServerConfig.from_env()
        assert config.verifier_method == "lexical"


class TestRetrieverSelection:
    def test_defaults_to_dense(self):
        assert ServerConfig().retriever == "dense"

    def test_reads_retriever_and_chunks_path(self, monkeypatch):
        monkeypatch.setenv("KENSHO_RETRIEVER", "sparse")
        monkeypatch.setenv("KENSHO_CHUNKS", "/tmp/chunks.jsonl")
        config = ServerConfig.from_env()
        assert config.retriever == "sparse"
        assert config.chunks_path == Path("/tmp/chunks.jsonl")


class TestNeedsEmbedder:
    """The point of sparse serving is that no embedding model is loaded.

    If this ever silently flips to True, a deployment sized for BM25 would
    start downloading model weights at boot and fall over on a small host -
    so it is asserted rather than assumed.
    """

    def test_sparse_with_lexical_verifier_needs_no_embedder(self):
        assert ServerConfig(retriever="sparse", verifier_method="lexical").needs_embedder is False

    def test_sparse_with_embedding_verifier_still_needs_one(self):
        assert ServerConfig(retriever="sparse", verifier_method="embedding").needs_embedder is True

    def test_dense_needs_one(self):
        assert ServerConfig(retriever="dense").needs_embedder is True

    def test_hybrid_needs_one(self):
        assert ServerConfig(retriever="hybrid").needs_embedder is True
