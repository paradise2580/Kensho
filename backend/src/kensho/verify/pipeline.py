"""Score a generated Answer's claims against the evidence it was given.

Phase 4 stops at scoring. Deciding what to *do* about an unsupported or
contradicted claim — strike it, re-retrieve, refuse the whole answer — is
phase 5. Scoring first, acting second means the acting logic can be
evaluated against a faithfulness rate that already means something, instead
of the two being built and debugged together.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pipeline import Answer
from .claims import Claim, Decomposer
from .nli import ClaimVerifier, VerificationResult


@dataclass(frozen=True, slots=True)
class VerifiedClaim:
    claim: Claim
    result: VerificationResult


@dataclass(frozen=True, slots=True)
class VerifiedAnswer:
    answer: Answer
    claims: list[VerifiedClaim]

    @property
    def faithfulness_rate(self) -> float | None:
        """Fraction of claims labelled "supported". ``None`` for an answer
        with no extractable claims (e.g. a refusal), where a rate would be
        a division by zero dressed up as a number."""
        if not self.claims:
            return None
        supported = sum(1 for c in self.claims if c.result.label == "supported")
        return supported / len(self.claims)

    @property
    def contradicted_claims(self) -> list[VerifiedClaim]:
        return [c for c in self.claims if c.result.label == "contradicted"]

    @property
    def unsupported_claims(self) -> list[VerifiedClaim]:
        return [c for c in self.claims if c.result.label == "unsupported"]


def verify_answer(answer: Answer, decomposer: Decomposer,
                  verifier: ClaimVerifier) -> VerifiedAnswer:
    claims = decomposer.decompose(answer.text)
    evidence = [h.payload.get("text", "") for h in answer.sources]
    verified = [VerifiedClaim(claim=c, result=verifier.verify(c.text, evidence)) for c in claims]
    return VerifiedAnswer(answer=answer, claims=verified)
