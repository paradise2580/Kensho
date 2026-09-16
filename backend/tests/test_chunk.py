"""Chunker tests, including the invariants each strategy must not violate."""

from __future__ import annotations

import pytest

from kensho.corpus.chunk import chunk_document, split_sentences
from kensho.schema import Document
from kensho.tokenizer import HeuristicTokenCounter

COUNTER = HeuristicTokenCounter()


def make_doc(body: str, lang: str = "en") -> Document:
    return Document(
        parallel_id="docs/concepts/test.md", lang=lang, title="Test",
        url="https://example.invalid/", body=body, commit_sha="deadbeef",
    )


LONG_EN = ("The controller manages the desired state. " * 40)
LONG_JA = ("コントローラーは望ましい状態を管理します。" * 40)
SHORT_EN = ("The controller manages the desired state. " * 6)


class TestSentenceSplitting:
    def test_japanese_terminator(self):
        """Japanese has no spaces; an English-only splitter returns one blob
        and the overlap logic silently stops working."""
        parts = split_sentences("これは一つ目です。これは二つ目です。")
        assert len(parts) == 2

    def test_english_terminator(self):
        assert len(split_sentences("First one. Second one. Third one.")) == 3

    def test_text_without_terminator_survives(self):
        assert split_sentences("no terminator here") == ["no terminator here"]

    def test_empty(self):
        assert split_sentences("   ") == []


@pytest.mark.parametrize("strategy", ["structural", "fixed"])
class TestBothStrategies:
    def test_produces_chunks(self, strategy):
        doc = make_doc(f"---\ntitle: T\n---\n# H\n{LONG_EN}\n")
        assert chunk_document(doc, COUNTER, strategy=strategy)

    def test_chunk_ids_unique(self, strategy):
        doc = make_doc(f"---\ntitle: T\n---\n# H\n{LONG_EN}\n")
        ids = [c.chunk_id for c in chunk_document(doc, COUNTER, strategy=strategy)]
        assert len(ids) == len(set(ids))

    def test_metadata_is_populated(self, strategy):
        doc = make_doc(f"---\ntitle: T\n---\n# H\n{LONG_EN}\n")
        for c in chunk_document(doc, COUNTER, strategy=strategy):
            assert c.parallel_id and c.lang and c.commit_sha
            assert c.token_count > 0
            assert c.strategy == strategy

    def test_no_runaway_chunk(self, strategy):
        """A single oversized sentence may exceed the target, but packing must
        never accumulate far beyond it."""
        doc = make_doc(f"---\ntitle: T\n---\n# H\n{LONG_EN}\n")
        chunks = chunk_document(doc, COUNTER, strategy=strategy, target_tokens=256)
        assert max(c.token_count for c in chunks) < 256 * 2

    def test_terminates_on_japanese(self, strategy):
        """Guards the packing loop's back-step: if it fails to make forward
        progress this hangs rather than fails, so it is worth asserting."""
        doc = make_doc(f"---\ntitle: T\n---\n# H\n{LONG_JA}\n", lang="ja")
        assert chunk_document(doc, COUNTER, strategy=strategy, overlap_tokens=128)

    def test_empty_body_yields_nothing(self, strategy):
        assert chunk_document(make_doc("---\ntitle: T\n---\n"), COUNTER,
                              strategy=strategy) == []


class TestStructuralInvariant:
    def test_chunk_text_stays_within_its_section_subtree(self):
        """The premise of the strategy: a chunk labelled with a section path
        must only contain text from inside that subtree.

        REGRESSION: a section too small to stand alone was merged into the
        previous chunk and inherited *that* chunk's path. Where the neighbour
        was a sibling rather than an ancestor, the merged text was attributed
        to a section it never came from — so retrieval would cite the wrong
        heading. Small sections now re-label to the common ancestor.
        """
        body = ("---\ntitle: T\n---\n"
                f"# A\n## B1\n{LONG_EN}\n## B2\nTiny orphan sentence.\n")
        for c in chunk_document(make_doc(body), COUNTER, strategy="structural"):
            if "Tiny orphan" in c.text and "controller manages" in c.text:
                # Merged with B1's content, so it may only claim their ancestor.
                assert c.section_path == ("A",)

    def test_page_shorter_than_min_tokens_is_not_dropped(self):
        body = "---\ntitle: T\n---\n# A\nShort.\n"
        assert chunk_document(make_doc(body), COUNTER, strategy="structural")

    def test_headings_reach_the_embedding_text(self):
        body = f"---\ntitle: T\n---\n# Phases\n{LONG_EN}\n"
        c = chunk_document(make_doc(body), COUNTER, strategy="structural")[0]
        assert "Test" in c.for_embedding()      # document title
        assert "Phases" in c.for_embedding()    # heading path


def test_strategies_differ_measurably():
    """If both strategies produced identical output the ablation would be
    meaningless. With sections smaller than the token target, structural is
    forced to fragment at boundaries where fixed packs straight through."""
    body = "---\ntitle: T\n---\n" + "".join(
        f"# Section {i}\n{SHORT_EN}\n" for i in range(8)
    )
    doc = make_doc(body)
    s = chunk_document(doc, COUNTER, strategy="structural", target_tokens=512)
    f = chunk_document(doc, COUNTER, strategy="fixed", target_tokens=512)
    assert len(s) > len(f)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        chunk_document(make_doc("x"), COUNTER, strategy="nope")
