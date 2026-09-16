"""Tokenization for BM25 — the piece dense embedding search doesn't need.

BM25 matches on literal terms, so how a string is cut into terms is not a
detail, it's most of the algorithm. English can get away with whitespace
splitting. Japanese cannot: it has no word boundaries, so whitespace
splitting indexes whole clauses as single "terms" and BM25 degenerates to
near-useless, string equality between entire sentences — the same fact that
makes morphological segmentation mandatory for the tokenizer used to size
chunks (see ``tokenizer.py``), now mandatory again here for a different
reason.

``SudachiSegmenter`` — real morphological segmentation via SudachiPy. Unlike
                       the embedding model and the LLM, its dictionary ships
                       in the pip package, so it needs no network at run
                       time and works in this sandbox exactly as it would
                       anywhere else — there is no fallback story to tell
                       for it the way there is for HFEmbedder.

``CharBigramSegmenter`` — the fallback if SudachiPy genuinely isn't
                       installed: overlapping character bigrams. This is a
                       real, if crude, CJK indexing technique (it's what
                       naive n-gram analyzers in full-text search engines
                       use for Japanese), not a strawman — but it inflates
                       the vocabulary and loses word identity, so treat any
                       BM25 numbers produced with it as a lower bound, not a
                       result to publish.

``WhitespaceSegmenter`` — plain lowercase word splitting, used for English
                       text, where this is simply the standard approach.
"""

from __future__ import annotations

import re
import warnings
from typing import Protocol

_WORD = re.compile(r"\w+", re.UNICODE)


class Segmenter(Protocol):
    def segment(self, text: str) -> list[str]: ...

    @property
    def name(self) -> str: ...


class WhitespaceSegmenter:
    @property
    def name(self) -> str:
        return "whitespace"

    def segment(self, text: str) -> list[str]:
        return _WORD.findall(text.lower())


class CharBigramSegmenter:
    """Overlapping character bigrams. Fallback only — see module docstring."""

    @property
    def name(self) -> str:
        return "char-bigram"

    def segment(self, text: str) -> list[str]:
        s = re.sub(r"\s+", "", text)
        if len(s) < 2:
            return [s] if s else []
        return [s[i:i + 2] for i in range(len(s) - 1)]


class SudachiSegmenter:
    """Real Japanese morphological segmentation via SudachiPy.

    Mode C merges the most aggressively (longest units — closer to "words"
    a human would recognize), which suits BM25 better than the finer A/B
    modes: fewer, more meaningful terms beats maximal decomposition for
    lexical matching.
    """

    def __init__(self) -> None:
        from sudachipy import dictionary, tokenizer  # heavy-ish; lazy import

        self._tokenizer = dictionary.Dictionary().create()
        self._mode = tokenizer.Tokenizer.SplitMode.C

    @property
    def name(self) -> str:
        return "sudachi"

    def segment(self, text: str) -> list[str]:
        return [
            m.surface() for m in self._tokenizer.tokenize(text, self._mode)
            if m.surface().strip()
        ]


def get_segmenter(lang: str, allow_fallback: bool = True) -> Segmenter:
    """Return the appropriate segmenter for a language.

    English always gets ``WhitespaceSegmenter`` — there's no fallback
    question for it. Japanese tries SudachiPy first; unlike the embedder,
    SudachiPy's dictionary is bundled in the package rather than fetched
    over the network, so a failure here almost always means the dependency
    truly isn't installed, not an environment quirk — but the same
    fail-loudly-by-default convention as ``get_embedder`` still applies,
    because ``CharBigramSegmenter`` changes BM25 results enough to matter.
    """
    if lang != "ja":
        return WhitespaceSegmenter()
    try:
        return SudachiSegmenter()
    except Exception as exc:  # noqa: BLE001 - missing dependency
        if not allow_fallback:
            raise RuntimeError(
                f"Could not load SudachiPy ({exc.__class__.__name__}: {exc}). "
                "Refusing to silently substitute character-bigram segmentation "
                "for a BM25 run — pass allow_fallback=True only for plumbing tests."
            ) from exc
        warnings.warn(
            f"Falling back to CharBigramSegmenter ({exc.__class__.__name__}). "
            "Japanese BM25 results from this run are a crude lower bound, not "
            "a result to publish.",
            RuntimeWarning,
            stacklevel=2,
        )
        return CharBigramSegmenter()
