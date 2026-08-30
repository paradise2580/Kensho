from __future__ import annotations

from kensho.correct.policy import (
    REFUSAL_EN,
    REFUSAL_JA,
    CorrectionConfig,
    correct_answer,
)
from kensho.pipeline import Answer
from kensho.store import SearchHit
from kensho.verify.claims import Claim
from kensho.verify.nli import VerificationResult
from kensho.verify.pipeline import VerifiedAnswer, VerifiedClaim


def vc(text: str, index: int, label: str) -> VerifiedClaim:
    return VerifiedClaim(
        claim=Claim(text=text, index=index),
        result=VerificationResult(claim=text, label=label, score=0.0,
                                  best_evidence_index=None, method="stub"),
    )


def make_verified(question: str, claims: list[VerifiedClaim]) -> VerifiedAnswer:
    answer = Answer(question=question, text=" ".join(c.claim.text for c in claims),
                    sources=[], llm_name="stub")
    return VerifiedAnswer(answer=answer, claims=claims)


class _NoResultsRetriever:
    """Every re-retrieval attempt finds nothing — the claim can't recover."""

    @property
    def name(self) -> str:
        return "no-results"

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]:
        return []


class _AlwaysHelpfulRetriever:
    """Every re-retrieval attempt finds evidence that will verify as supported."""

    @property
    def name(self) -> str:
        return "always-helpful"

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[SearchHit]:
        return [SearchHit(chunk_id="x", score=1.0, payload={"text": query})]


class _AlwaysSupportedVerifier:
    """Verifies anything with any evidence as supported — for isolating the
    retry mechanics from a specific verifier's own logic."""

    @property
    def name(self) -> str:
        return "always-supported"

    def verify(self, claim: str, evidence: list[str]) -> VerificationResult:
        label = "supported" if evidence else "unsupported"
        return VerificationResult(claim, label, 1.0, 0 if evidence else None, self.name)


class TestSupportedAndContradicted:
    def test_supported_claim_is_kept(self):
        verified = make_verified("q?", [vc("A is true.", 0, "supported")])
        result = correct_answer(verified, _NoResultsRetriever(), _AlwaysSupportedVerifier())
        assert result.decisions[0].action == "kept"
        assert not result.refused
        assert result.final_text == "A is true."

    def test_contradicted_claim_is_struck_without_retry(self):
        retriever = _AlwaysHelpfulRetriever()  # would recover it if retried
        verified = make_verified("q?", [
            vc("A is true.", 0, "supported"),
            vc("B is false.", 1, "contradicted"),
        ])
        result = correct_answer(verified, retriever, _AlwaysSupportedVerifier())
        assert result.decisions[1].action == "struck"
        # contradicted claims never get a retry — confirm no evidence text
        # from the "always helpful" retriever leaked into a recovery.
        assert result.decisions[1].updated_result is None


class TestUnsupportedRetry:
    def test_recovers_when_retry_finds_supporting_evidence(self):
        verified = make_verified("q?", [vc("C is plausible.", 0, "unsupported")])
        result = correct_answer(verified, _AlwaysHelpfulRetriever(), _AlwaysSupportedVerifier())
        assert result.decisions[0].action == "recovered"
        assert result.decisions[0].updated_result is not None
        assert not result.refused

    def test_strikes_when_retry_finds_nothing(self):
        verified = make_verified("q?", [
            vc("A is true.", 0, "supported"),
            vc("C is plausible.", 1, "unsupported"),
        ])
        result = correct_answer(verified, _NoResultsRetriever(), _AlwaysSupportedVerifier())
        assert result.decisions[1].action == "struck"


class TestRefusal:
    def test_refuses_when_survival_rate_below_threshold(self):
        verified = make_verified("q?", [
            vc("A is true.", 0, "supported"),
            vc("B is false.", 1, "contradicted"),
            vc("C is false too.", 2, "contradicted"),
        ])
        result = correct_answer(verified, _NoResultsRetriever(), _AlwaysSupportedVerifier(),
                                config=CorrectionConfig(refusal_threshold=0.5))
        assert result.refused
        assert result.final_text == REFUSAL_EN
        assert "1/3" in result.refusal_reason

    def test_does_not_refuse_when_survival_rate_meets_threshold(self):
        verified = make_verified("q?", [
            vc("A is true.", 0, "supported"),
            vc("B is true.", 1, "supported"),
            vc("C is false.", 2, "contradicted"),
        ])
        result = correct_answer(verified, _NoResultsRetriever(), _AlwaysSupportedVerifier(),
                                config=CorrectionConfig(refusal_threshold=0.5))
        assert not result.refused
        assert result.final_text == "A is true. B is true."

    def test_refusal_text_matches_japanese_question(self):
        verified = make_verified("これは何ですか?", [vc("B is false.", 0, "contradicted")])
        result = correct_answer(verified, _NoResultsRetriever(), _AlwaysSupportedVerifier())
        assert result.refused
        assert result.final_text == REFUSAL_JA

    def test_no_claims_is_not_treated_as_refusal(self):
        """An already-empty answer (e.g. the generator itself declined) is
        left untouched, not overwritten with a different refusal message."""
        answer = Answer(question="q?", text="I don't know.", sources=[], llm_name="stub")
        verified = VerifiedAnswer(answer=answer, claims=[])
        result = correct_answer(verified, _NoResultsRetriever(), _AlwaysSupportedVerifier())
        assert not result.refused
        assert result.final_text == "I don't know."
        assert result.decisions == []

    def test_kept_claims_preserve_original_order(self):
        verified = make_verified("q?", [
            vc("Second fact.", 1, "supported"),
            vc("First fact.", 0, "supported"),
        ])
        result = correct_answer(verified, _NoResultsRetriever(), _AlwaysSupportedVerifier())
        assert result.final_text == "First fact. Second fact."


class TestKeptAndStruckProperties:
    def test_kept_claims_includes_recovered(self):
        verified = make_verified("q?", [
            vc("A is true.", 0, "supported"),
            vc("C is plausible.", 1, "unsupported"),
        ])
        result = correct_answer(verified, _AlwaysHelpfulRetriever(), _AlwaysSupportedVerifier())
        assert len(result.kept_claims) == 2
        assert result.struck_claims == []
