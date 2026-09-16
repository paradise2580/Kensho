"""Splitting a generated answer into the units claim verification checks.

Two decomposition granularities, because which one is right is an empirical
question, not an assumption — the same reason ``corpus/chunk.py`` keeps two
chunking strategies instead of picking one:

``SentenceSplitDecomposer`` — reuses the script-aware sentence splitter
                              already built for chunking. Real, needs no
                              model, works identically in every environment.
                              Coarser: a sentence with two facts in it is one
                              claim, so a single false detail sinks the
                              whole sentence's verdict.

``LLMDecomposer``            — asks the generator itself to rewrite the
                              answer as one atomic factual claim per line.
                              Finer-grained and closer to what "claim-level"
                              verification usually means in the literature,
                              at the cost of depending on the LLM doing the
                              decomposition faithfully — decomposition
                              errors (merging, dropping, inventing detail)
                              are a real failure mode of this method, not a
                              hypothetical one, which is precisely why
                              having the cruder sentence-level method to
                              compare against matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..corpus.chunk import split_sentences
from ..llm import LLMProvider

DECOMPOSE_PROMPT = (
    "Rewrite the following answer as a list of atomic factual claims — one "
    "self-contained statement per line, no numbering, no commentary. Each "
    "claim must be understandable without reading the others. If the answer "
    "already declines to answer or contains no factual claims, output "
    "nothing.\n\nAnswer:\n{answer}"
)


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    index: int


class Decomposer(Protocol):
    def decompose(self, answer_text: str) -> list[Claim]: ...

    @property
    def name(self) -> str: ...


class SentenceSplitDecomposer:
    @property
    def name(self) -> str:
        return "sentence-split"

    def decompose(self, answer_text: str) -> list[Claim]:
        sentences = split_sentences(answer_text)
        return [Claim(text=s, index=i) for i, s in enumerate(sentences)]


class LLMDecomposer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return f"llm:{self._llm.name}"

    def decompose(self, answer_text: str) -> list[Claim]:
        raw = self._llm.generate(DECOMPOSE_PROMPT.format(answer=answer_text), temperature=0.0)
        lines = [line.strip(" -*\t") for line in raw.splitlines()]
        lines = [line for line in lines if line]
        return [Claim(text=line, index=i) for i, line in enumerate(lines)]


def get_decomposer(mode: str = "sentence", llm: LLMProvider | None = None) -> Decomposer:
    if mode == "sentence":
        return SentenceSplitDecomposer()
    if mode == "llm":
        if llm is None:
            raise ValueError("mode='llm' requires an llm= LLMProvider instance")
        return LLMDecomposer(llm)
    raise ValueError(f"unknown decomposer mode {mode!r}; expected 'sentence' or 'llm'")
