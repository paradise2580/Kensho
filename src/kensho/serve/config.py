"""Server configuration, read from the environment.

A config object rather than reading ``os.environ`` scattered through
``app.py`` for the usual reason: it's the one place that has to be right,
and it's what tests override to point at a temp trace file and fake/echo
components instead of real models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..embed import DEFAULT_MODEL as DEFAULT_EMBEDDER


@dataclass(frozen=True, slots=True)
class ServerConfig:
    index_dir: Path = Path("data/qdrant")
    embedder_model: str | None = DEFAULT_EMBEDDER  # None -> FakeEmbedder
    llm_provider: str | None = "groq"  # None -> EchoProvider
    verifier_method: str = "lexical"
    decomposer_mode: str = "sentence"
    # "dense" (embeddings), "sparse" (BM25), or "hybrid" (RRF over both).
    #
    # Phase 3 measured all three; only dense was reachable at serving time,
    # which meant any deployment without Hugging Face Hub access had to fall
    # back to the fake embedder and serve nonsense. Sparse needs no model
    # weights at all - SudachiPy ships its dictionary in the wheel - so
    # exposing it here is what makes a genuinely grounded deployment
    # possible on a host with no model storage and little RAM.
    retriever: str = "dense"
    # Sparse and hybrid retrieval index chunk text directly at startup
    # rather than reading prebuilt vectors, so they need the chunk file.
    chunks_path: Path = Path("data/chunks/structural_t512_o64.jsonl")
    top_k: int = 5
    refusal_threshold: float = 0.5
    allow_fallback: bool = False
    trace_path: Path = Path("data/traces/kensho.jsonl")

    @property
    def needs_embedder(self) -> bool:
        """Whether this configuration actually loads an embedding model.

        Sparse retrieval with a non-embedding verifier touches no embedder,
        and constructing one anyway would download several hundred MB of
        weights that nothing then uses - the difference between a container
        that starts in seconds and one that does not fit in a free tier.
        """
        return self.retriever in ("dense", "hybrid") or self.verifier_method == "embedding"

    @classmethod
    def from_env(cls) -> ServerConfig:
        def _opt(name: str, default: str | None) -> str | None:
            val = os.environ.get(name, default)
            return None if val in ("", "none", "fake", "echo") else val

        return cls(
            index_dir=Path(os.environ.get("KENSHO_INDEX_DIR", "data/qdrant")),
            embedder_model=_opt("KENSHO_EMBEDDER", DEFAULT_EMBEDDER),
            llm_provider=_opt("KENSHO_LLM", "groq"),
            verifier_method=os.environ.get("KENSHO_VERIFIER", "lexical"),
            decomposer_mode=os.environ.get("KENSHO_DECOMPOSER", "sentence"),
            retriever=os.environ.get("KENSHO_RETRIEVER", "dense"),
            chunks_path=Path(os.environ.get(
                "KENSHO_CHUNKS", "data/chunks/structural_t512_o64.jsonl")),
            top_k=int(os.environ.get("KENSHO_TOP_K", "5")),
            refusal_threshold=float(os.environ.get("KENSHO_REFUSAL_THRESHOLD", "0.5")),
            allow_fallback=os.environ.get("KENSHO_ALLOW_FALLBACK", "").lower() in
            ("1", "true", "yes"),
            trace_path=Path(os.environ.get("KENSHO_TRACE_PATH", "data/traces/kensho.jsonl")),
        )
