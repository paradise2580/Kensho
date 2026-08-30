"""Answer generation, with a pluggable provider — same shape as tokenizer.py
and embed.py.

``GroqProvider`` — the real generator, via Groq's OpenAI-compatible chat
                   completions API. Chosen for phase 1 because it is free-
                   tier, fast (LPU inference), and swappable — nothing else
                   in the pipeline assumes Groq specifically.

``EchoProvider``  — no network, no model. Returns a fixed, clearly-labelled
                   string so pipeline wiring (prompt assembly, source
                   attachment, error handling) can be built and tested
                   without an API key, without ever being mistaken for a
                   real answer.

Unlike ``get_embedder``, the factory here defaults to *allowing* fallback.
A generation failure is visible the moment a human reads the answer — it
says "[ECHO]" — whereas a bad embedding silently degrades every downstream
score. The two factories make opposite default choices for that reason.
"""

from __future__ import annotations

import os
import warnings
from typing import Protocol


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str: ...

    @property
    def name(self) -> str: ...


class GroqProvider:
    """Groq chat completions. Requires GROQ_API_KEY."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from groq import Groq  # heavy-ish import, and fails loudly with no key

        self._model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it or pass api_key= explicitly — "
                "see .env.example."
            )
        self._client = Groq(api_key=key)

    @property
    def name(self) -> str:
        return f"groq:{self._model}"

    def generate(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class EchoProvider:
    """No model, no network. For pipeline plumbing tests only."""

    @property
    def name(self) -> str:
        return "echo"

    def generate(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        return f"[ECHO — no model called. Prompt was {len(prompt)} chars.]"


def get_llm(provider: str | None = "groq", allow_fallback: bool = True,
           **kw) -> LLMProvider:
    """Return the requested provider, or Echo if it fails and fallback is allowed.

    ``provider=None`` returns Echo directly, same convention as
    ``get_embedder``/``get_token_counter``.
    """
    if provider is None:
        return EchoProvider()
    if provider not in ("groq",):
        raise ValueError(f"unknown provider {provider!r}; expected 'groq' or None")
    try:
        return GroqProvider(**kw)
    except Exception as exc:  # noqa: BLE001 - missing key, missing dep, network
        if not allow_fallback:
            raise RuntimeError(
                f"Could not initialize LLM provider {provider!r} "
                f"({exc.__class__.__name__}: {exc})."
            ) from exc
        warnings.warn(
            f"Falling back to EchoProvider ({exc.__class__.__name__}: {exc}). "
            "Answers from this run are placeholders, not generations.",
            RuntimeWarning,
            stacklevel=2,
        )
        return EchoProvider()
