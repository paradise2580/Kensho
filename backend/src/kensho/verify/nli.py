"""Four ways to check whether a claim is supported by its evidence,
cheapest to most expensive — the "4-arm" comparison this project's name
(検証, "verification") is actually about.

``LexicalOverlapVerifier``    — real, free, script-aware token-coverage.
                                No model, no network. Can say "supported" /
                                "unsupported" but has no way to recognize a
                                claim that *contradicts* its evidence — high
                                lexical overlap with the opposite polarity
                                looks identical to it.

``EmbeddingSimilarityVerifier`` — cosine similarity between the claim and
                                each evidence passage, via any ``Embedder``.
                                Same contradiction blind spot as lexical
                                overlap, for the same reason: "Pods restart
                                automatically" and "Pods never restart
                                automatically" embed close together because
                                they're topically identical.

``NLIVerifier``                — a real entailment model
                                (mDeBERTa-v3-base-xnli-multilingual-nli-2mil7,
                                MIT licensed, genuinely trained to
                                distinguish entailment / neutral /
                                contradiction). The only one of the four
                                that can detect a contradicted claim rather
                                than merely an unsupported one. Needs
                                Hugging Face Hub access.

``LLMJudgeVerifier``           — ask the generator's own LLM whether the
                                evidence supports the claim. Flexible and
                                needs no dedicated model, but it is a
                                second LLM call trusting a first LLM's
                                output, which is a real, worth-measuring
                                risk of correlated failure — if the
                                generator hallucinates in a way its own
                                judgment would defend, this arm is blind to
                                exactly that case, unlike NLI's independent
                                model.

Every arm returns "unsupported" as its default when it cannot form a
confident verdict — an unparseable LLM response, no evidence supplied. That
default is chosen deliberately: verification exists to catch unjustified
claims, so an ambiguous result should read as "not yet justified," not
"assume it's fine."
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Literal, Protocol

from ..embed import Embedder
from ..llm import LLMProvider
from ..retrieval.segment import get_segmenter
from ..tokenizer import script_of

Label = Literal["supported", "unsupported", "contradicted"]

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

JUDGE_PROMPT = (
    "Evidence:\n{evidence}\n\nClaim: {claim}\n\n"
    "Does the evidence support the claim? Reply with exactly one word: "
    "SUPPORTED, CONTRADICTED, or UNSUPPORTED."
)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    claim: str
    label: Label
    score: float
    best_evidence_index: int | None
    method: str


class ClaimVerifier(Protocol):
    def verify(self, claim: str, evidence: list[str]) -> VerificationResult: ...

    @property
    def name(self) -> str: ...


class LexicalOverlapVerifier:
    """Fraction of the claim's tokens found in an evidence passage,
    maximized over the evidence set. Script-aware: Japanese needs real
    segmentation (whitespace splitting would treat a whole clause as one
    "token" and make overlap meaningless), same as BM25 does.
    """

    def __init__(self, threshold: float = 0.6, allow_fallback: bool = True) -> None:
        self._threshold = threshold
        self._segmenters = {
            "ja": get_segmenter("ja", allow_fallback=allow_fallback),
            "en": get_segmenter("en", allow_fallback=allow_fallback),
        }

    @property
    def name(self) -> str:
        return "lexical-overlap"

    def _tokens(self, text: str) -> set[str]:
        lang = "ja" if script_of(text) == "cjk" else "en"
        return set(self._segmenters[lang].segment(text))

    def verify(self, claim: str, evidence: list[str]) -> VerificationResult:
        claim_tokens = self._tokens(claim)
        if not claim_tokens or not evidence:
            return VerificationResult(claim, "unsupported", 0.0, None, self.name)
        best_score, best_idx = 0.0, None
        for i, ev in enumerate(evidence):
            ev_tokens = self._tokens(ev)
            if not ev_tokens:
                continue
            coverage = len(claim_tokens & ev_tokens) / len(claim_tokens)
            if coverage > best_score:
                best_score, best_idx = coverage, i
        label: Label = "supported" if best_score >= self._threshold else "unsupported"
        return VerificationResult(claim, label, best_score, best_idx, self.name)


class EmbeddingSimilarityVerifier:
    def __init__(self, embedder: Embedder, threshold: float = 0.85) -> None:
        self._embedder = embedder
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"embedding-sim:{self._embedder.name}"

    def verify(self, claim: str, evidence: list[str]) -> VerificationResult:
        if not evidence:
            return VerificationResult(claim, "unsupported", 0.0, None, self.name)
        claim_vec = self._embedder.embed_query(claim)
        ev_vecs = self._embedder.embed_passages(evidence)
        sims = [_cosine(claim_vec, v) for v in ev_vecs]
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        best = sims[best_idx]
        label: Label = "supported" if best >= self._threshold else "unsupported"
        return VerificationResult(claim, label, best, best_idx, self.name)


class NLIVerifier:
    """Real cross-lingual entailment via mDeBERTa. See module docstring for
    why this is the only arm that can label a claim "contradicted"."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL,
                entail_threshold: float = 0.5, contradict_threshold: float = 0.5) -> None:
        import torch  # heavy; lazy import
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._model_name = model_name
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._entail_threshold = entail_threshold
        self._contradict_threshold = contradict_threshold
        self._id2label = {i: lbl.lower() for i, lbl in self._model.config.id2label.items()}

    @property
    def name(self) -> str:
        return f"nli:{self._model_name}"

    def _probs(self, premise: str, hypothesis: str) -> dict[str, float]:
        inputs = self._tokenizer(premise, hypothesis, return_tensors="pt",
                                 truncation=True, max_length=512)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = self._torch.softmax(logits, dim=-1).tolist()
        return {self._id2label[i]: p for i, p in enumerate(probs)}

    def verify(self, claim: str, evidence: list[str]) -> VerificationResult:
        if not evidence:
            return VerificationResult(claim, "unsupported", 0.0, None, self.name)
        best_idx, best_entail, best_probs = None, -1.0, None
        for i, ev in enumerate(evidence):
            probs = self._probs(premise=ev, hypothesis=claim)
            entail = probs.get("entailment", 0.0)
            if entail > best_entail:
                best_idx, best_entail, best_probs = i, entail, probs
        contradict = (best_probs or {}).get("contradiction", 0.0)
        if best_entail >= self._entail_threshold:
            label: Label = "supported"
        elif contradict >= self._contradict_threshold:
            label = "contradicted"
        else:
            label = "unsupported"
        return VerificationResult(claim, label, best_entail, best_idx, self.name)


class LLMJudgeVerifier:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return f"llm-judge:{self._llm.name}"

    def verify(self, claim: str, evidence: list[str]) -> VerificationResult:
        if not evidence:
            return VerificationResult(claim, "unsupported", 0.0, None, self.name)
        ev_text = "\n\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence))
        raw = self._llm.generate(
            JUDGE_PROMPT.format(evidence=ev_text, claim=claim), max_tokens=8, temperature=0.0,
        )
        label = _parse_judge_label(raw)
        score = 0.0 if label == "unsupported" else 1.0
        return VerificationResult(claim, label, score, None, self.name)


def _parse_judge_label(raw: str) -> Label:
    text = raw.strip().upper()
    if "CONTRADICT" in text:
        return "contradicted"
    if "UNSUPPORTED" in text:
        return "unsupported"
    if "SUPPORTED" in text:
        return "supported"
    # Unparseable (e.g. EchoProvider's "[ECHO ...]") — refuse rather than
    # assume the claim is fine.
    return "unsupported"


class FakeVerifier:
    """Deterministic, offline, meaning-blind — the same role FakeEmbedder
    and FakeReranker play elsewhere in this project. For plumbing only."""

    @property
    def name(self) -> str:
        return "fake"

    def verify(self, claim: str, evidence: list[str]) -> VerificationResult:
        if not evidence:
            return VerificationResult(claim, "unsupported", 0.0, None, self.name)
        digest = hashlib.blake2b(claim.encode("utf-8"), digest_size=8).digest()
        r = int.from_bytes(digest, "big") / 2**64
        label: Label = "supported" if r < 0.34 else ("contradicted" if r < 0.5 else "unsupported")
        return VerificationResult(claim, label, r, 0, self.name)


def get_verifier(method: str, *, llm: LLMProvider | None = None,
                 embedder: Embedder | None = None, allow_fallback: bool = False,
                 **kwargs) -> ClaimVerifier:
    """Dispatch to one of the four arms.

    Only ``method="nli"`` has a fallback story of its own — it lazily loads
    a Hugging Face model, same as ``get_embedder``. ``"embedding"`` and
    ``"llm-judge"`` take an already-resolved ``Embedder``/``LLMProvider``,
    so any fallback decision for *those* was already made by
    ``get_embedder``/``get_llm`` before this function is called.
    """
    if method == "lexical":
        return LexicalOverlapVerifier(**kwargs)
    if method == "embedding":
        if embedder is None:
            raise ValueError("method='embedding' requires embedder=")
        return EmbeddingSimilarityVerifier(embedder, **kwargs)
    if method == "llm-judge":
        if llm is None:
            raise ValueError("method='llm-judge' requires llm=")
        return LLMJudgeVerifier(llm)
    if method == "nli":
        try:
            return NLIVerifier(**kwargs)
        except Exception as exc:  # noqa: BLE001 - missing dep, no network, bad name
            if not allow_fallback:
                raise RuntimeError(
                    f"Could not load NLI model ({exc.__class__.__name__}: {exc}). "
                    "Refusing to silently substitute a fake verifier — pass "
                    "allow_fallback=True only for plumbing tests."
                ) from exc
            warnings.warn(
                f"Falling back to FakeVerifier ({exc.__class__.__name__}). "
                "Verdicts from this run are meaningless and must not be reported.",
                RuntimeWarning,
                stacklevel=2,
            )
            return FakeVerifier()
    raise ValueError(f"unknown verification method {method!r}; expected one of "
                     "'lexical', 'embedding', 'nli', 'llm-judge'")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
