from __future__ import annotations

import pytest

from kensho.llm import EchoProvider
from kensho.verify.claims import LLMDecomposer, SentenceSplitDecomposer, get_decomposer


class TestSentenceSplitDecomposer:
    def test_splits_english_sentences(self):
        claims = SentenceSplitDecomposer().decompose(
            "A Pod runs one or more containers. It has a defined lifecycle."
        )
        assert [c.text for c in claims] == [
            "A Pod runs one or more containers.", "It has a defined lifecycle.",
        ]

    def test_splits_japanese_sentences(self):
        text = "Podは複数のコンテナを持てます。ライフサイクルがあります。"
        assert len(SentenceSplitDecomposer().decompose(text)) == 2

    def test_indexes_are_sequential(self):
        claims = SentenceSplitDecomposer().decompose("One. Two. Three.")
        assert [c.index for c in claims] == [0, 1, 2]

    def test_empty_answer_produces_no_claims(self):
        assert SentenceSplitDecomposer().decompose("") == []

    def test_name(self):
        assert SentenceSplitDecomposer().name == "sentence-split"


class TestLLMDecomposer:
    def test_parses_one_claim_per_line(self):
        class StubLLM:
            name = "stub"

            def generate(self, prompt, *, max_tokens=1024, temperature=0.0):
                return "Pods have a lifecycle.\n- Phases include Pending and Running.\n"

        claims = LLMDecomposer(StubLLM()).decompose("anything")
        assert [c.text for c in claims] == [
            "Pods have a lifecycle.", "Phases include Pending and Running.",
        ]

    def test_blank_lines_are_dropped(self):
        class StubLLM:
            name = "stub"

            def generate(self, prompt, *, max_tokens=1024, temperature=0.0):
                return "Claim one.\n\n\nClaim two.\n"

        claims = LLMDecomposer(StubLLM()).decompose("anything")
        assert len(claims) == 2

    def test_no_claims_output_is_empty_list(self):
        class StubLLM:
            name = "stub"

            def generate(self, prompt, *, max_tokens=1024, temperature=0.0):
                return ""

        assert LLMDecomposer(StubLLM()).decompose("I don't know.") == []

    def test_name_includes_underlying_llm(self):
        assert LLMDecomposer(EchoProvider()).name == "llm:echo"


class TestGetDecomposer:
    def test_default_is_sentence_split(self):
        assert get_decomposer().name == "sentence-split"

    def test_llm_mode_requires_llm(self):
        with pytest.raises(ValueError, match="requires an llm"):
            get_decomposer("llm")

    def test_llm_mode_with_llm(self):
        assert get_decomposer("llm", llm=EchoProvider()).name == "llm:echo"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown decomposer mode"):
            get_decomposer("paragraph")
