"""The FastAPI service: one process holding one loaded pipeline, answering
``POST /ask`` and reporting its own health at ``GET /healthz``.

Every component is built once at startup (``PipelineState``) and reused
across requests — re-loading an embedding model or an NLI model per request
would be the difference between a demo that answers in two seconds and one
that answers in twenty. ``create_app(config)`` takes an explicit
``ServerConfig`` rather than always reading the environment, which is what
lets the test suite spin up a fully in-process app with fake/echo
components and a throwaway trace file, with no real model and no network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel

from ..correct.policy import CorrectionConfig, correct_answer
from ..embed import Embedder, get_embedder
from ..llm import LLMProvider, get_llm
from ..pipeline import ask as run_ask
from ..store import ChunkStore
from ..verify.claims import Decomposer, get_decomposer
from ..verify.nli import ClaimVerifier, get_verifier
from ..verify.pipeline import verify_answer
from .config import ServerConfig
from .trace import ClaimTrace, RetrievedChunk, Timer, Trace, TraceWriter, new_trace_id


@dataclass(slots=True)
class PipelineState:
    config: ServerConfig
    embedder: Embedder
    store: ChunkStore
    llm: LLMProvider
    decomposer: Decomposer
    verifier: ClaimVerifier
    trace_writer: TraceWriter


def build_pipeline_state(config: ServerConfig) -> PipelineState:
    embedder = get_embedder(config.embedder_model, allow_fallback=config.allow_fallback)
    store = ChunkStore(embedder, path=config.index_dir)
    llm = get_llm(config.llm_provider, allow_fallback=config.allow_fallback)
    decomposer = get_decomposer(config.decomposer_mode, llm=llm)
    verifier = get_verifier(config.verifier_method, llm=llm, embedder=embedder,
                            allow_fallback=config.allow_fallback)
    trace_writer = TraceWriter(config.trace_path)
    return PipelineState(config=config, embedder=embedder, store=store, llm=llm,
                         decomposer=decomposer, verifier=verifier, trace_writer=trace_writer)


class AskRequest(BaseModel):
    question: str
    lang: str | None = None
    top_k: int | None = None


class ClaimOut(BaseModel):
    text: str
    label: str
    action: str


class SourceOut(BaseModel):
    parallel_id: str
    title: str
    lang: str
    score: float


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    refused: bool
    refusal_reason: str | None
    claims: list[ClaimOut]
    sources: list[SourceOut]
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    index_size: int
    embedder: str
    llm: str
    verifier: str


def create_app(config: ServerConfig | None = None) -> FastAPI:
    config = config or ServerConfig.from_env()
    state = build_pipeline_state(config)

    app = FastAPI(
        title="Kenshō",
        description="Bilingual JA/EN enterprise-support RAG with claim-level "
                   "citation verification.",
    )

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok", index_size=state.store.count(), embedder=state.embedder.name,
            llm=state.llm.name, verifier=state.verifier.name,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask_endpoint(req: AskRequest) -> AskResponse:
        top_k = req.top_k or state.config.top_k
        total_timer = Timer()
        with total_timer:
            with Timer() as t_ask:
                answer = run_ask(req.question, state.store, state.llm, top_k=top_k, lang=req.lang)
            with Timer() as t_verify:
                verified = verify_answer(answer, state.decomposer, state.verifier)
            with Timer() as t_correct:
                corrected = correct_answer(
                    verified, state.store, state.verifier,
                    config=CorrectionConfig(refusal_threshold=state.config.refusal_threshold),
                )

        trace = Trace(
            trace_id=new_trace_id(),
            timestamp=_now(),
            question=req.question,
            lang_filter=req.lang,
            retrieved=[
                RetrievedChunk(
                    chunk_id=h.chunk_id, parallel_id=h.payload.get("parallel_id", ""),
                    title=h.payload.get("title", ""), lang=h.payload.get("lang", ""),
                    score=h.score,
                )
                for h in answer.sources
            ],
            answer_text=answer.text,
            llm_name=answer.llm_name,
            embedder_name=state.embedder.name,
            verifier_name=state.verifier.name,
            claims=[
                ClaimTrace(text=d.verified.claim.text, label=d.final_result.label,
                          action=d.action, method=d.final_result.method)
                for d in corrected.decisions
            ],
            final_text=corrected.final_text,
            refused=corrected.refused,
            refusal_reason=corrected.refusal_reason,
            latency_ms={
                "retrieval_and_generation": round(t_ask.ms, 2),
                "verification": round(t_verify.ms, 2),
                "correction": round(t_correct.ms, 2),
                "total": round(total_timer.ms, 2),
            },
        )
        state.trace_writer.write(trace)

        return AskResponse(
            trace_id=trace.trace_id,
            answer=corrected.final_text,
            refused=corrected.refused,
            refusal_reason=corrected.refusal_reason,
            claims=[ClaimOut(text=c.text, label=c.label, action=c.action) for c in trace.claims],
            sources=[
                SourceOut(parallel_id=r.parallel_id, title=r.title, lang=r.lang, score=r.score)
                for r in trace.retrieved
            ],
            latency_ms=trace.latency_ms,
        )

    return app


def _now() -> float:
    return time.time()
