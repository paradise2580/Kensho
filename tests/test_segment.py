"""Segmenter tests. SudachiPy's dictionary ships in the pip package, so
unlike the embedder and LLM, the *real* backend is actually exercised here —
this sandbox has no network dependency to fall back from. Only the
Sudachi-specific tests are skipped when the optional `sparse` extra isn't
installed; Whitespace/CharBigram need no such guard."""

from __future__ import annotations

import importlib.util

import pytest

from kensho.retrieval.segment import (
    CharBigramSegmenter,
    SudachiSegmenter,
    WhitespaceSegmenter,
    get_segmenter,
)

requires_sudachi = pytest.mark.skipif(
    importlib.util.find_spec("sudachipy") is None,
    reason="sparse extra not installed — see pyproject.toml",
)


class TestWhitespaceSegmenter:
    def test_lowercases_and_splits(self):
        assert WhitespaceSegmenter().segment("A Pod Can Hold Containers") == [
            "a", "pod", "can", "hold", "containers",
        ]

    def test_empty_string(self):
        assert WhitespaceSegmenter().segment("") == []


class TestCharBigramSegmenter:
    def test_produces_overlapping_bigrams(self):
        assert CharBigramSegmenter().segment("abc") == ["ab", "bc"]

    def test_strips_whitespace_before_bigramming(self):
        assert CharBigramSegmenter().segment("a b") == ["ab"]

    def test_single_char_is_kept_whole(self):
        assert CharBigramSegmenter().segment("a") == ["a"]

    def test_empty_string(self):
        assert CharBigramSegmenter().segment("") == []


@requires_sudachi
class TestSudachiSegmenter:
    """SudachiPy is genuinely installed and offline-capable here — these are
    real assertions about real morphological segmentation, not plumbing."""

    def test_splits_a_real_japanese_sentence_into_morphemes(self):
        tokens = SudachiSegmenter().segment("ポッドは複数のコンテナを持てます")
        assert "ポッド" in tokens
        assert "コンテナ" in tokens
        # REGRESSION guard: this is the exact failure mode morphological
        # segmentation exists to avoid — the whole sentence as one "token".
        assert "ポッドは複数のコンテナを持てます" not in tokens

    def test_particle_is_its_own_token(self):
        tokens = SudachiSegmenter().segment("Podは")
        assert "は" in tokens


class TestGetSegmenter:
    def test_english_always_gets_whitespace(self):
        assert get_segmenter("en").name == "whitespace"

    @requires_sudachi
    def test_japanese_gets_sudachi_when_available(self):
        assert get_segmenter("ja").name == "sudachi"

    def test_unknown_lang_falls_back_to_whitespace(self):
        assert get_segmenter("fr").name == "whitespace"
