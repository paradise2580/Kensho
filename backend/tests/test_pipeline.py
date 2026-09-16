from __future__ import annotations

from kensho.embed import FakeEmbedder
from kensho.llm import EchoProvider
from kensho.pipeline import ask
from kensho.prompts import build_prompt
from kensho.schema import Chunk
from kensho.store import ChunkStore


def make_chunk(text: str, title: str = "Pods", lang: str = "en", ordinal: int = 0) -> Chunk:
    doc_id = f"{lang}:doc"
    return Chunk(
        chunk_id=Chunk.make_id(doc_id, ordinal, "structural"),
        doc_id=doc_id,
        parallel_id="doc",
        lang=lang,
        title=title,
        section_path=("Intro",),
        text=text,
        token_count=len(text.split()),
        char_count=len(text),
        ordinal=ordinal,
        url="https://example.invalid/doc",
        strategy="structural",
        commit_sha="deadbeef",
    )


class TestPrompts:
    def test_no_hits_tells_model_to_say_so(self):
        prompt = build_prompt("What is a Pod?", [])
        assert "don't have information" in prompt

    def test_hits_are_numbered_and_include_heading(self):
        store = ChunkStore(FakeEmbedder(dim=16), path=None)
        store.upsert([make_chunk("A Pod can hold several containers.")])
        hits = store.search("A Pod can hold several containers", top_k=1)
        prompt = build_prompt("What is a Pod?", hits)
        assert "[1]" in prompt
        assert "Pods › Intro" in prompt


class TestAsk:
    def test_returns_answer_with_sources_and_llm_name(self):
        store = ChunkStore(FakeEmbedder(dim=16), path=None)
        store.upsert([make_chunk("A Pod can hold several containers.")])
        answer = ask("What is a Pod?", store, EchoProvider())
        assert answer.question == "What is a Pod?"
        assert answer.llm_name == "echo"
        assert len(answer.sources) == 1
        assert "ECHO" in answer.text

    def test_no_matching_docs_still_returns_an_answer(self):
        store = ChunkStore(FakeEmbedder(dim=16), path=None)
        answer = ask("What is a Pod?", store, EchoProvider())
        assert answer.sources == []
        assert "ECHO" in answer.text
