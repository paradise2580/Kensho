"""Verifier tests. LexicalOverlapVerifier is exercised for real (it needs no
model); EmbeddingSimilarityVerifier is tested against FakeEmbedder, which
only validates surface lexical similarity, not semantic entailment — see
its own docstring caveat, repeated here in the tests that rely on it.
NLIVerifier and LLMJudgeVerifier's real backends need Hugging Face Hub /
a Groq key, so those are tested through their fallback contracts instead.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from kensho.embed import FakeEmbedder
from kensho.llm import EchoProvider
from kensho.verify.nli import (
    EmbeddingSimilarityVerifier,
    FakeVerifier,
    LexicalOverlapVerifier,
    LLMJudgeVerifier,
    get_verifier,
)


class _SymmetricEmbedder:
    """Test double for EmbeddingSimilarityVerifier's own arithmetic
    (thresholding, best-match selection) — unlike ``FakeEmbedder``, it
    encodes queries and passages identically, so near-duplicate text gets a
    high, unambiguous cosine regardless of which method wrapped it.
    ``FakeEmbedder``'s query/passage salt asymmetry is deliberate (it's what
    makes forgetting the e5 prefix convention a real bug to catch — see
    embed.py), but it also means a query-vs-passage comparison through it
    doesn't cleanly separate near-duplicates from unrelated text at small
    dimensions. This double isolates the verifier's own logic from that
    unrelated quirk; it is not a claim about semantic embedding realism —
    see EmbeddingSimilarityVerifier's own docstring caveat for that."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return "symmetric-fake"

    def _vec(self, text: str) -> list[float]:
        buckets = np.zeros(self.dim)
        norm = text.strip().lower()
        for n in (3, 4, 5):
            for i in range(max(0, len(norm) - n + 1)):
                gram = norm[i:i + n]
                h = int(hashlib.blake2b(gram.encode(), digest_size=8).hexdigest(), 16)
                buckets[h % self.dim] += 1.0
        length = np.linalg.norm(buckets)
        return (buckets / length if length > 0 else buckets).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class TestLexicalOverlapVerifier:
    def test_high_overlap_is_supported(self):
        v = LexicalOverlapVerifier(threshold=0.6)
        result = v.verify(
            "Pods follow a defined lifecycle starting in the Pending phase.",
            ["This page describes the lifecycle of a Pod. Pods follow a "
             "defined lifecycle, starting in the Pending phase."],
        )
        assert result.label == "supported"
        assert result.best_evidence_index == 0

    def test_low_overlap_is_unsupported(self):
        v = LexicalOverlapVerifier(threshold=0.6)
        result = v.verify(
            "A Pod's lifecycle includes a Suspended phase.",
            ["Pods follow a defined lifecycle: Pending, Running, Succeeded, or Failed."],
        )
        assert result.label == "unsupported"

    def test_cannot_detect_contradiction(self):
        """Documented limitation: a direct negation shares almost all of its
        vocabulary with the true statement, so lexical overlap alone reads
        it as (nearly) supported, not contradicted."""
        v = LexicalOverlapVerifier(threshold=0.6)
        result = v.verify(
            "A Pod never starts in the Pending phase.",
            ["Pods follow a defined lifecycle, starting in the Pending phase."],
        )
        assert result.label != "contradicted"

    def test_no_evidence_is_unsupported(self):
        assert LexicalOverlapVerifier().verify("anything", []).label == "unsupported"

    def test_japanese_claim_uses_real_segmentation(self):
        v = LexicalOverlapVerifier(threshold=0.5)
        result = v.verify(
            "Podは複数のコンテナを持てます",
            ["Podは複数のコンテナを持てます。ライフサイクルがあります。"],
        )
        assert result.label == "supported"

    def test_name(self):
        assert LexicalOverlapVerifier().name == "lexical-overlap"


class TestEmbeddingSimilarityVerifier:
    def test_near_identical_text_is_supported(self):
        """Checks near-duplicate phrasing, not real semantic entailment —
        see _SymmetricEmbedder's docstring for why it's used here instead
        of FakeEmbedder directly."""
        v = EmbeddingSimilarityVerifier(_SymmetricEmbedder(dim=64), threshold=0.7)
        result = v.verify(
            "A Pod can hold several containers",
            ["A Pod can hold several containers."],
        )
        assert result.label == "supported"

    def test_unrelated_text_is_unsupported(self):
        v = EmbeddingSimilarityVerifier(_SymmetricEmbedder(dim=64), threshold=0.7)
        result = v.verify(
            "Quarterly revenue rose sharply in Tokyo.",
            ["A Pod can hold several containers."],
        )
        assert result.label == "unsupported"

    def test_no_evidence_is_unsupported(self):
        v = EmbeddingSimilarityVerifier(FakeEmbedder(dim=16))
        assert v.verify("anything", []).label == "unsupported"

    def test_name_includes_embedder(self):
        assert EmbeddingSimilarityVerifier(FakeEmbedder(dim=16)).name == "embedding-sim:fake"


class TestLLMJudgeVerifier:
    def test_parses_supported(self):
        class StubLLM:
            name = "stub"

            def generate(self, prompt, *, max_tokens=1024, temperature=0.0):
                return "SUPPORTED"

        result = LLMJudgeVerifier(StubLLM()).verify("claim", ["evidence"])
        assert result.label == "supported"

    def test_parses_contradicted(self):
        class StubLLM:
            name = "stub"

            def generate(self, prompt, *, max_tokens=1024, temperature=0.0):
                return "CONTRADICTED."

        result = LLMJudgeVerifier(StubLLM()).verify("claim", ["evidence"])
        assert result.label == "contradicted"

    def test_unparseable_response_defaults_to_unsupported(self):
        result = LLMJudgeVerifier(EchoProvider()).verify("claim", ["evidence"])
        assert result.label == "unsupported"

    def test_no_evidence_is_unsupported(self):
        result = LLMJudgeVerifier(EchoProvider()).verify("claim", [])
        assert result.label == "unsupported"


class TestFakeVerifier:
    def test_deterministic(self):
        v = FakeVerifier()
        assert v.verify("same claim", ["ev"]) == v.verify("same claim", ["ev"])

    def test_no_evidence_is_unsupported(self):
        assert FakeVerifier().verify("anything", []).label == "unsupported"


class TestGetVerifier:
    def test_lexical(self):
        assert get_verifier("lexical").name == "lexical-overlap"

    def test_embedding_requires_embedder(self):
        with pytest.raises(ValueError, match="requires embedder"):
            get_verifier("embedding")

    def test_embedding_with_embedder(self):
        assert get_verifier("embedding", embedder=FakeEmbedder(dim=16)).name == "embedding-sim:fake"

    def test_llm_judge_requires_llm(self):
        with pytest.raises(ValueError, match="requires llm"):
            get_verifier("llm-judge")

    def test_llm_judge_with_llm(self):
        assert get_verifier("llm-judge", llm=EchoProvider()).name == "llm-judge:echo"

    def test_nli_raises_by_default_when_model_unavailable(self):
        with pytest.raises(RuntimeError, match="Refusing to silently substitute"):
            get_verifier("nli")

    def test_nli_allow_fallback_warns_and_returns_fake(self):
        with pytest.warns(RuntimeWarning, match="meaningless"):
            v = get_verifier("nli", allow_fallback=True)
        assert v.name == "fake"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown verification method"):
            get_verifier("vibes")
