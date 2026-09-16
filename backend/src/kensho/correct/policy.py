"""Acting on a verification verdict: strike, recover, or refuse.

Phase 4 only scores an answer's claims. This module is what does something
about a bad score:

- a **contradicted** claim is always struck. There is no recovery path for
  it — the model asserted the opposite of what the evidence says, and
  re-retrieving more evidence for a claim that's already actively wrong
  doesn't fix the claim, it just risks finding a passage the (flawed)
  lexical/embedding arms would call supporting.
- an **unsupported** claim gets one retry: search specifically for *that
  claim's text* (not the original question) against the full corpus, and
  re-verify against whatever comes back. Sometimes the original top-k for
  the question simply didn't surface the one passage that supports a
  specific detail; a claim-targeted search sometimes finds it. If it still
  doesn't verify, the claim is struck.
- if what survives striking is too thin — below `refusal_threshold` of the
  original claims, or nothing at all — the whole answer is replaced with an
  explicit refusal rather than shown as a quietly-shortened answer. A
  mutilated answer that still reads as confident is arguably worse than no
  answer.

This is deliberately simple policy, not a search over strategies: one retry,
one threshold, no iteration. Measuring whether *this* policy's refusal
decisions land on the right side of the answerable/unanswerable line in the
phase-2 gold set is what `scripts/eval_refusal.py` is for — improving the
policy is future work informed by that measurement, not something to
guess at up front.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..store import Retriever
from ..tokenizer import script_of
from ..verify.nli import ClaimVerifier, VerificationResult
from ..verify.pipeline import VerifiedAnswer, VerifiedClaim

Action = Literal["kept", "struck", "recovered"]

REFUSAL_EN = ("I don't have enough verified information in the retrieved "
             "sources to answer this confidently.")
REFUSAL_JA = ("取得した情報からは、この質問に確信を持って回答するための"
             "十分な検証済み情報が得られませんでした。")


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    refusal_threshold: float = 0.5
    re_retrieve_top_k: int = 3


@dataclass(frozen=True, slots=True)
class ClaimDecision:
    verified: VerifiedClaim
    action: Action
    updated_result: VerificationResult | None = None

    @property
    def final_result(self) -> VerificationResult:
        return self.updated_result or self.verified.result


@dataclass(frozen=True, slots=True)
class CorrectedAnswer:
    verified: VerifiedAnswer
    decisions: list[ClaimDecision] = field(default_factory=list)
    final_text: str = ""
    refused: bool = False
    refusal_reason: str | None = None

    @property
    def kept_claims(self) -> list[ClaimDecision]:
        return [d for d in self.decisions if d.action != "struck"]

    @property
    def struck_claims(self) -> list[ClaimDecision]:
        return [d for d in self.decisions if d.action == "struck"]


def _refusal_text(question: str) -> str:
    return REFUSAL_JA if script_of(question) == "cjk" else REFUSAL_EN


def correct_answer(verified: VerifiedAnswer, retriever: Retriever, verifier: ClaimVerifier,
                   config: CorrectionConfig | None = None) -> CorrectedAnswer:
    config = config or CorrectionConfig()
    decisions: list[ClaimDecision] = []
    for vc in verified.claims:
        label = vc.result.label
        if label == "supported":
            decisions.append(ClaimDecision(vc, "kept"))
        elif label == "contradicted":
            decisions.append(ClaimDecision(vc, "struck"))
        else:  # unsupported: one targeted retry
            hits = retriever.search(vc.claim.text, top_k=config.re_retrieve_top_k)
            new_evidence = [h.payload.get("text", "") for h in hits]
            retry_result = verifier.verify(vc.claim.text, new_evidence)
            if retry_result.label == "supported":
                decisions.append(ClaimDecision(vc, "recovered", updated_result=retry_result))
            else:
                decisions.append(ClaimDecision(vc, "struck"))

    total = len(decisions)
    if total == 0:
        # Nothing to correct — e.g. the answer was already a refusal-shaped
        # response with no extractable claims. Leave it exactly as it was;
        # inventing a *different* refusal here would be putting words in
        # the generator's mouth that it didn't produce.
        return CorrectedAnswer(verified=verified, decisions=decisions,
                               final_text=verified.answer.text, refused=False)

    kept = [d for d in decisions if d.action != "struck"]
    survival_rate = len(kept) / total

    if not kept or survival_rate < config.refusal_threshold:
        reason = (f"only {len(kept)}/{total} claims verified against the "
                 "retrieved sources (or a claim-targeted retry)")
        return CorrectedAnswer(
            verified=verified, decisions=decisions,
            final_text=_refusal_text(verified.answer.question),
            refused=True, refusal_reason=reason,
        )

    kept_in_order = sorted(kept, key=lambda d: d.verified.claim.index)
    final_text = " ".join(d.verified.claim.text for d in kept_in_order)
    return CorrectedAnswer(verified=verified, decisions=decisions, final_text=final_text,
                           refused=False)
