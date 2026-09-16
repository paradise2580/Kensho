from __future__ import annotations

import pytest

from kensho.llm import EchoProvider, get_llm


class TestEchoProvider:
    def test_name(self):
        assert EchoProvider().name == "echo"

    def test_generate_labels_itself(self):
        out = EchoProvider().generate("hello")
        assert "ECHO" in out


class TestGetLLM:
    def test_none_returns_echo_directly(self):
        assert get_llm(provider=None).name == "echo"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown provider"):
            get_llm(provider="not-a-real-provider")

    def test_missing_api_key_falls_back_by_default(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.warns(RuntimeWarning, match="placeholders"):
            llm = get_llm(provider="groq")
        assert llm.name == "echo"

    def test_missing_api_key_raises_when_fallback_disallowed(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            get_llm(provider="groq", allow_fallback=False)
