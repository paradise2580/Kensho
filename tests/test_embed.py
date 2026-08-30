"""Embedder tests.

HFEmbedder is not exercised here — Hugging Face Hub is unreachable in this
environment (see README). These tests cover the contract FakeEmbedder must
satisfy to be a safe plumbing stand-in: stable, normalized, and structured so
that near-identical text lands near-identical vectors — the minimum needed to
sanity-check that storage and cosine search round-trip correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from kensho.embed import FakeEmbedder, get_embedder


def cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestFakeEmbedder:
    def test_deterministic(self):
        e = FakeEmbedder(dim=32)
        assert e.embed_query("hello world") == e.embed_query("hello world")

    def test_normalized(self):
        e = FakeEmbedder(dim=32)
        vec = e.embed_query("A Pod can hold several containers")
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)

    def test_empty_string_does_not_divide_by_zero(self):
        e = FakeEmbedder(dim=16)
        vec = e.embed_query("")
        assert vec == [0.0] * 16

    def test_near_identical_strings_land_close(self):
        e = FakeEmbedder(dim=128)
        a = e.embed_passages(["A Pod can hold several containers."])[0]
        b = e.embed_passages(["A pod can hold several containers"])[0]  # case + punctuation
        assert cosine(a, b) > 0.8

    def test_unrelated_strings_land_far(self):
        e = FakeEmbedder(dim=128)
        a = e.embed_passages(["A Pod can hold several containers."])[0]
        b = e.embed_passages(["Quarterly revenue rose sharply in Tokyo."])[0]
        assert cosine(a, b) < 0.5

    def test_query_and_passage_use_different_salts(self):
        """The query/passage asymmetry must survive even the fake backend,
        or plumbing tests would not catch a caller that forgets to call the
        right method."""
        e = FakeEmbedder(dim=32)
        assert e.embed_query("same text") != e.embed_passages(["same text"])[0]

    def test_dim_is_respected(self):
        e = FakeEmbedder(dim=17)
        assert len(e.embed_query("x")) == 17
        assert e.dim == 17

    def test_batch_matches_individual(self):
        e = FakeEmbedder(dim=32)
        texts = ["first passage", "second passage"]
        batch = e.embed_passages(texts)
        individual = [e.embed_passages([t])[0] for t in texts]
        assert batch == individual


class TestGetEmbedder:
    def test_none_model_returns_fake_directly(self):
        e = get_embedder(model_name=None)
        assert e.name == "fake"

    def test_missing_dep_or_network_raises_by_default(self):
        """HF Hub is unreachable in this sandbox, so a real model name must
        fail loudly rather than silently hand back a meaning-blind vector."""
        with pytest.raises(RuntimeError, match="Refusing to silently substitute"):
            get_embedder("intfloat/multilingual-e5-small")

    def test_allow_fallback_warns_and_returns_fake(self):
        with pytest.warns(RuntimeWarning, match="meaningless"):
            e = get_embedder("intfloat/multilingual-e5-small", allow_fallback=True)
        assert e.name == "fake"
