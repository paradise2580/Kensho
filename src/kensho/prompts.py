"""Prompt assembly for grounded answering.

One template, kept in one place, because phase 4 (claim verification) needs
to know exactly what the model was told to justify why a claim should or
should not be traceable to a source — that traceability starts here.

The instructions are deliberately explicit about refusal and about not
mixing languages, because both are places a general-purpose chat model
defaults to the wrong behaviour: it would rather guess than say "the
documents don't cover this," and it will happily answer a Japanese question
in English if not told to match the query's language.
"""

from __future__ import annotations

from .store import SearchHit

SYSTEM_PREAMBLE = (
    "You are an enterprise support assistant. Answer only using the numbered "
    "sources below. If the sources do not contain the answer, say so plainly "
    "instead of guessing. Answer in the same language as the question. Cite "
    "sources inline as [1], [2], etc., matching the numbers given."
)


def format_sources(hits: list[SearchHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        title = h.payload.get("title", "")
        section = " › ".join(h.payload.get("section_path", []))
        heading = f"{title} › {section}" if section else title
        blocks.append(f"[{i}] ({heading})\n{h.payload.get('text', '')}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    if not hits:
        return (
            f"{SYSTEM_PREAMBLE}\n\n"
            "No sources were retrieved for this question. Say that you don't "
            "have information to answer it.\n\n"
            f"Question: {question}"
        )
    return (
        f"{SYSTEM_PREAMBLE}\n\n"
        f"Sources:\n{format_sources(hits)}\n\n"
        f"Question: {question}"
    )
