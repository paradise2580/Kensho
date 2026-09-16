"""Phase 1 control-arm pipeline: retrieve, then generate, once.

This is deliberately the simplest thing that could work — dense-only search,
top-5, one LLM call, no reranking, no verification. Every later phase is an
ablation measured *against* this baseline, so resist the urge to improve it
here; improvements belong in phase 3 where they can be scored against this
exact configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMProvider
from .prompts import build_prompt
from .store import ChunkStore, SearchHit


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    text: str
    sources: list[SearchHit]
    llm_name: str


def ask(question: str, store: ChunkStore, llm: LLMProvider, *,
       top_k: int = 5, lang: str | None = None) -> Answer:
    hits = store.search(question, top_k=top_k, lang=lang)
    prompt = build_prompt(question, hits)
    text = llm.generate(prompt)
    return Answer(question=question, text=text, sources=hits, llm_name=llm.name)
