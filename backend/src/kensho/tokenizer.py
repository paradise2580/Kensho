"""Token counting, with a pluggable backend.

Chunk sizes are specified in *tokens*, not characters, because that is what the
embedding model's context window is measured in — and because the
character-to-token ratio differs by almost 4x between English and Japanese.
Sizing chunks by character count would silently give Japanese chunks a quarter
of the semantic content of English ones, which would then show up as "Japanese
retrieval is worse" and be misread as a model problem.

Two backends:

``HFTokenCounter``  — the real XLM-R tokenizer shared by multilingual-e5 and
                      bge-m3. Correct, and what you should use for any run
                      whose numbers you intend to report.

``HeuristicTokenCounter`` — offline fallback, calibrated per script. Within
                      roughly 10-15% of the real count, which is fine for
                      smoke tests and CI but not for published results.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

# XLM-R (used by multilingual-e5 and bge-m3) is the reference tokenizer.
DEFAULT_HF_TOKENIZER = "intfloat/multilingual-e5-large"

_CJK = re.compile(
    r"[぀-ゟ"  # hiragana
    r"゠-ヿ"   # katakana
    r"一-鿿"   # CJK unified ideographs
    r"ｦ-ﾟ]"  # halfwidth katakana
)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    @property
    def name(self) -> str: ...


class HeuristicTokenCounter:
    """Script-aware approximation, for when the real tokenizer is unavailable.

    Calibration rationale: XLM-R's SentencePiece vocabulary splits Japanese
    into roughly one token per 1.1 characters (kana and kanji are mostly
    single pieces), while English averages closer to 4 characters per token.
    Counting both at one ratio is the mistake this class exists to avoid.
    """

    JA_CHARS_PER_TOKEN = 1.1
    LATIN_CHARS_PER_TOKEN = 4.0

    @property
    def name(self) -> str:
        return "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        cjk = len(_CJK.findall(text))
        other = len(text) - cjk
        return max(1, round(cjk / self.JA_CHARS_PER_TOKEN + other / self.LATIN_CHARS_PER_TOKEN))


class HFTokenCounter:
    """Wraps a Hugging Face tokenizer. Use this for reportable runs."""

    def __init__(self, model_name: str = DEFAULT_HF_TOKENIZER) -> None:
        from transformers import AutoTokenizer  # imported lazily, it is heavy

        self._model_name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)

    @property
    def name(self) -> str:
        return f"hf:{self._model_name}"

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


def get_token_counter(model_name: str | None = DEFAULT_HF_TOKENIZER,
                      allow_fallback: bool = True) -> TokenCounter:
    """Return the real tokenizer if it loads, else the heuristic.

    Fails loudly rather than silently degrading when ``allow_fallback`` is
    False — set that for any run whose chunk statistics you will publish.
    """
    if model_name is None:
        return HeuristicTokenCounter()
    try:
        return HFTokenCounter(model_name)
    except Exception as exc:  # noqa: BLE001 - network, missing dep, bad name
        if not allow_fallback:
            raise RuntimeError(
                f"Could not load tokenizer {model_name!r} and fallback is disabled. "
                f"Chunk sizes would not match the embedding model. Original error: {exc}"
            ) from exc
        import warnings

        warnings.warn(
            f"Falling back to heuristic token counting ({exc.__class__.__name__}). "
            "Chunk boundaries will be approximate — do not publish numbers from this run.",
            RuntimeWarning,
            stacklevel=2,
        )
        return HeuristicTokenCounter()


def script_of(text: str) -> str:
    """Rough script label, used for corpus statistics and sanity checks."""
    cjk = len(_CJK.findall(text))
    letters = sum(1 for c in text if unicodedata.category(c).startswith("L"))
    if letters == 0:
        return "none"
    return "cjk" if cjk / letters > 0.3 else "latin"
