"""Text embedding, with a pluggable backend — same shape as tokenizer.py.

``HFEmbedder``   — sentence-transformers wrapping the real model. Correct, and
                   what any reportable eval run must use.

``FakeEmbedder`` — a deterministic hash-based vector, no model, no network.
                   It preserves nothing about meaning — two paraphrases of the
                   same sentence get unrelated vectors. It exists solely so
                   the retrieval/storage/pipeline *plumbing* can be built and
                   tested in an environment with no model access, without ever
                   being mistaken for a real result. ``allow_fallback=False``
                   is the default for exactly that reason: a fallback silently
                   accepted is a benchmark silently invalidated.

Both implement the same query/passage asymmetry that e5-family models
require: a query and the passage it should match are embedded with different
prefixes. Forgetting this measurably degrades retrieval and is a documented,
easy-to-make mistake with these specific models.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_DIM = {"intfloat/multilingual-e5-large": 1024,
              "intfloat/multilingual-e5-small": 384,
              "BAAI/bge-m3": 1024}


class Embedder(Protocol):
    dim: int

    def embed_query(self, text: str) -> list[float]: ...
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def name(self) -> str: ...


class HFEmbedder:
    """sentence-transformers backend for the e5 / bge-m3 family.

    e5 models are trained on prefixed inputs and silently underperform without
    them — there is no error, only worse retrieval, which is exactly the kind
    of bug that survives into a published eval by looking like a modelling
    limitation rather than a usage mistake.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # heavy; lazy import

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return f"hf:{self._model_name}"

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(f"query: {text}", normalize_embeddings=True)
        return vec.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self._model.encode(prefixed, normalize_embeddings=True,
                                  show_progress_bar=len(texts) > 64)
        return vecs.tolist()


class FakeEmbedder:
    """Deterministic, offline, meaning-blind. For plumbing tests only.

    Built from character n-gram hashing rather than pure random noise so that
    near-identical strings (e.g. a chunk and itself) land near-identical
    vectors — enough to sanity-check that storage and cosine search round-trip
    correctly — while still being nowhere close to a real embedding model.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return "fake"

    def _vec(self, text: str, salt: str) -> list[float]:
        buckets = np.zeros(self.dim, dtype=np.float64)
        norm = text.strip().lower()
        for n in (3, 4, 5):
            for i in range(max(0, len(norm) - n + 1)):
                gram = norm[i:i + n]
                h = int(hashlib.blake2b(f"{salt}:{gram}".encode(), digest_size=8).hexdigest(), 16)
                buckets[h % self.dim] += 1.0
        length = np.linalg.norm(buckets)
        return (buckets / length if length > 0 else buckets).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text, "q")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t, "p") for t in texts]


def get_embedder(model_name: str | None = DEFAULT_MODEL,
                 allow_fallback: bool = False) -> Embedder:
    """Return the real embedder, or the offline stand-in if explicitly allowed.

    Defaults to *not* falling back, unlike ``get_token_counter`` — an
    embedding is the substance of the whole retrieval system, and a silent
    swap to a meaning-blind vector is a much larger correctness gap than an
    approximate token count.
    """
    if model_name is None:
        return FakeEmbedder()
    try:
        return HFEmbedder(model_name)
    except Exception as exc:  # noqa: BLE001 - missing dep, no network, bad name
        if not allow_fallback:
            raise RuntimeError(
                f"Could not load embedder {model_name!r} ({exc.__class__.__name__}: {exc}). "
                "Refusing to silently substitute a fake embedder for a retrieval run — "
                "pass allow_fallback=True only for plumbing tests that do not measure "
                "retrieval quality."
            ) from exc
        warnings.warn(
            f"Falling back to FakeEmbedder ({exc.__class__.__name__}). "
            "Retrieval results from this run are meaningless and must not be reported.",
            RuntimeWarning,
            stacklevel=2,
        )
        return FakeEmbedder()
