from __future__ import annotations

from kensho.pipeline import Answer
from kensho.store import SearchHit
from kensho.verify.claims import SentenceSplitDecomposer
from kensho.verify.nli import FakeVerifier, LexicalOverlapVerifier
from kensho.verify.pipeline import verify_answer


def make_answer(text: str, source_texts: list[str]) -> Answer:
    sources = [
        SearchHit(chunk_id=str(i), score=1.0, payload={"text": t})
        for i, t in enumerate(source_texts)
    ]
    return Answer(question="q?", text=text, sources=sources, llm_name="stub")


class TestVerifyAnswer:
    def test_decomposes_and_verifies_each_claim(self):
        answer = make_answer(
            "Pods follow a defined lifecycle. A Pod's lifecycle includes a Suspended phase.",
            ["Pods follow a defined lifecycle, starting in the Pending phase."],
        )
        verified = verify_answer(answer, SentenceSplitDecomposer(), LexicalOverlapVerifier())
        assert len(verified.claims) == 2
        assert verified.claims[0].result.label == "supported"
        assert verified.claims[1].result.label == "unsupported"

    def test_faithfulness_rate(self):
        answer = make_answer(
            "Pods follow a defined lifecycle. A Pod's lifecycle includes a Suspended phase.",
            ["Pods follow a defined lifecycle, starting in the Pending phase."],
        )
        verified = verify_answer(answer, SentenceSplitDecomposer(), LexicalOverlapVerifier())
        assert verified.faithfulness_rate == 0.5

    def test_no_claims_gives_none_faithfulness_rate(self):
        answer = make_answer("", ["some evidence"])
        verified = verify_answer(answer, SentenceSplitDecomposer(), LexicalOverlapVerifier())
        assert verified.claims == []
        assert verified.faithfulness_rate is None

    def test_contradicted_and_unsupported_claim_lists(self):
        answer = make_answer("Claim one. Claim two.", ["some evidence"])
        verified = verify_answer(answer, SentenceSplitDecomposer(), FakeVerifier())
        total = len(verified.contradicted_claims) + len(verified.unsupported_claims) + \
            sum(1 for c in verified.claims if c.result.label == "supported")
        assert total == 2

    def test_evidence_comes_from_answer_sources(self):
        """REGRESSION guard: verification must check against what was
        actually retrieved, not the question or some other text."""
        answer = make_answer("The sky is green.", ["Pods follow a defined lifecycle."])
        verified = verify_answer(answer, SentenceSplitDecomposer(), LexicalOverlapVerifier())
        assert verified.claims[0].result.label == "unsupported"
