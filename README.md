# Kenshō 検証

A bilingual (Japanese / English) enterprise-support RAG assistant that verifies
every sentence it writes against the passages it retrieved — and refuses to
answer when it cannot.

Most RAG systems stop at retrieve-and-generate. Kenshō adds a verification
stage: the generated answer is decomposed into atomic claims, each claim is
checked against its retrieved sources with a multilingual entailment model, and
claims that nothing supports are struck, re-retrieved, or the answer is
declined outright.

> **Status: phase 7 complete — all seven phases built.** Corpus, chunking,
> dense retrieval, single-shot generation, a hand-labelled evaluation
> harness, seven retrieval ablations, claim-level verification,
> self-correction, and serving (a FastAPI service, per-request tracing, and
> an eval-gated CI check) are built and tested end to end. See
> [`WRITEUP.md`](WRITEUP.md) for the shorter narrative version of this
> README: the problem, the architecture, the key decisions, and exactly
> which numbers are real versus plumbing-only.

![Kenshō architecture: retrieve, generate, decompose, verify, correct, trace](docs/architecture.svg)

---

## Why Japanese–English

Multilingual retrieval is one of the standard hard problems in production RAG.
Japanese–English is the pair that removes every crutch:

- **No word boundaries.** Japanese does not separate words with spaces, so BM25
  with whitespace tokenization indexes whole clauses as single terms and is
  close to useless. Morphological segmentation is not optional.
- **Non-Latin script.** No cognate overlap, so lexical search cannot succeed by
  accident and cross-lingual embedding quality is tested honestly.
- **Distant morphology and syntax.** Alignment is genuinely hard.

An English–Spanish system can look competent on shared vocabulary alone. This
one cannot.

## Corpus

The Kubernetes documentation, from
[`kubernetes/website`](https://github.com/kubernetes/website) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Nothing is scraped:
`content/en/` and `content/ja/` mirror each other at identical relative paths,
so a sparse checkout is the entire ingestion step and **the shared relative
path is the parallel id** — the JA/EN alignment that cross-lingual evaluation
depends on comes for free.

Measured at the pinned commit, with `docs/reference/` excluded (see below):

| | pages |
|---|---:|
| English | 406 |
| Japanese | 324 |
| **Parallel (both languages)** | **305** |
| English-only | 101 |

`docs/reference/` is excluded by default. It is 928 of the 1,370 English pages,
almost all auto-generated API stubs with a median body of 453 characters, and
only 12 are translated. Including it makes the corpus 68% English-only
boilerplate, which buries real answers under near-empty stubs *and* stacks ~900
untranslated English pages against every Japanese query — so Japanese retrieval
would score badly for a reason that has nothing to do with the retriever. That
is a measurement artefact waiting to be misread as a finding. Pass
`--include-reference` to opt back in.

## Chunking

Two strategies, because comparing them is ablation 3.2:

| strategy | chunks | mean tokens (en / ja) |
|---|---:|---:|
| `structural` — never spans a heading boundary | 8,446 | 222 / 270 |
| `fixed` — sliding window over the page | 5,413 | 435 / 448 |

The contrast is the hypothesis made concrete: `structural` yields more, smaller,
topically-pure chunks; `fixed` packs closer to the token target but lets one
chunk cover two subjects. Which trade wins is an empirical question, and phase 2
exists to answer it.

Two details that matter more than they look:

**Token counts are script-aware.** The character-to-token ratio differs almost
fourfold between English and Japanese. Sizing chunks by character count would
silently give Japanese chunks a quarter of the semantic content of English ones,
and that would surface later as "Japanese retrieval is worse" — a bug misread as
a model limitation.

**Headings are prepended before embedding.** A paragraph reading "The default is
30 seconds" is unretrievable on its own; carrying its `title › section › path`
makes it findable by a question that names the setting.

## Quickstart

```bash
pip install -e ".[dev,corpus,retrieval,generation]"

# Clone the docs and build the chunked corpus
python scripts/build_corpus.py --strategy structural --target-tokens 512

# Embed the chunks into a local Qdrant store (no server, no Docker)
python scripts/build_index.py data/chunks/structural_t512_o64.jsonl

# Ask a question — retrieval + one grounded generation call
export GROQ_API_KEY=...   # see .env.example
python scripts/ask.py "How does a Pod's restart policy work?"
python scripts/ask.py "Pod の再起動ポリシーはどう動作しますか?" --lang ja

# Score retrieval against the hand-labelled gold set
python scripts/run_eval.py data/gold/gold_v1.jsonl

# Run all seven phase-3 retrieval ablations and print a comparison table
python scripts/ablate.py data/gold/gold_v1.jsonl

# Compare all four claim-verification methods on the faithfulness gold set
python scripts/compare_verifiers.py data/gold/faithfulness_v1.jsonl

# Full pipeline with self-correction: retrieve, generate, verify, correct
python scripts/correct_ask.py "How does a Pod's restart policy work?"

# Score refusal decisions against the phase-2 gold set's answerable/unanswerable split
python scripts/eval_refusal.py data/gold/gold_v1.jsonl

# Run the API server (see Serving section for env vars)
python scripts/serve.py

# Offline / CI: skip model downloads and API calls entirely
python scripts/build_corpus.py --skip-clone --tokenizer heuristic
python scripts/build_index.py data/chunks/structural_t512_o64.jsonl --embedder fake --allow-fallback
python scripts/ask.py "test" --embedder fake --llm echo --allow-fallback
python scripts/run_eval.py data/gold/gold_v1.jsonl --embedder fake --allow-fallback

pytest -q && ruff check .
```

Each run writes `data/chunks/<stem>.jsonl` plus a `<stem>.manifest.json`
recording the corpus SHA, the tokenizer actually used, and the resulting
statistics — so any later result can be traced back to the exact corpus that
produced it.

### On the tokenizer

Chunk sizes are specified in tokens, so they only mean anything if the counter
matches the embedding model. `--tokenizer` defaults to the real XLM-R tokenizer
shared by `multilingual-e5` and `bge-m3`. If it cannot be loaded, the build
falls back to a script-calibrated heuristic, warns, and stamps `heuristic` into
the manifest. Pass `--strict-tokenizer` to fail instead — use that for any run
whose numbers you intend to publish.

## Retrieval and generation (phase 1)

Dense-only, top-5, one LLM call — deliberately the simplest thing that works.
Every later phase is an ablation scored against this exact configuration, so
it stays unimproved on purpose.

- `embed.py` — same pluggable-backend shape as `tokenizer.py`. `HFEmbedder`
  wraps sentence-transformers and applies the `query: ` / `passage: ` prefix
  the e5 family requires and silently underperforms without. `FakeEmbedder`
  is a deterministic character n-gram hash — meaning-blind, offline, for
  plumbing only. Unlike the tokenizer, `get_embedder()` defaults to
  `allow_fallback=False`: a silently swapped-in meaning-blind vector is a far
  bigger correctness gap than an approximate token count.
- `store.py` — Qdrant in embedded/local mode (`QdrantClient(path=...)`), no
  server or Docker. Payload carries `lang` for the cross-lingual filtering
  ablation 3.x needs, plus everything a citation requires (title, section
  path, URL).
- `llm.py` — Groq as the primary generator (free tier, fast, swappable).
  `EchoProvider` is the no-network stand-in. Here the fallback default flips
  to `True`: a bad generation is visible the moment a human reads "[ECHO]",
  unlike a bad embedding, which degrades every downstream score invisibly.
- `prompts.py` / `pipeline.py` — one grounded-answering template, explicit
  about refusing when sources don't cover the question and about matching
  the question's language; `ask()` ties retrieval and generation together.

## Evaluation (phase 2)

`data/gold/gold_v1.jsonl` — 52 hand-labelled items, 26 English and 26
Japanese, split across 24 real Kubernetes concepts (Pods, Deployments,
Services, ConfigMaps, scheduling, and so on). Every query and reference
answer was written by reading the actual EN/JA text of the corresponding
page, not recalled from general Kubernetes knowledge — `scripts/seed_gold_v1.py`
documents that provenance and is how the set is extended (a v2 follows the
same read-the-real-page-first discipline, never adds an item from memory).

Ground truth is recorded at **page** (`parallel_id`) level rather than
`chunk_id`, so the same gold set scores both chunking strategies without
going stale when ablation 3.2 compares them.

48 items are answerable; 4 are deliberately not — plausible support
questions this corpus has no coverage for (Kubernetes pricing) or with a
false premise (Kubernetes "stock price"). The harness excludes them from
retrieval metrics, since recall/MRR/nDCG have no meaning when nothing should
be retrieved, but seeds them now so phase 5's refusal evaluation has real
cases on day one instead of starting from zero.

`scripts/run_eval.py` retrieves top-k for every answerable item and reports
recall@{1,3,5,10}, MRR, and nDCG@10, overall and broken down by query
language, then writes a timestamped JSON report to `eval_reports/` — the
same leave-a-manifest discipline as `build_corpus.py` and `build_index.py`,
so a phase-3 ablation can be diffed against this exact baseline rather than
trusted from memory. `scripts/validate_gold.py` checks the gold set's
page references still exist in a given corpus build, since the upstream
Kubernetes docs can rename or remove a page between re-pins.

Phase 2 deliberately stops at retrieval metrics. Scoring whether a
*generated* answer is faithful to its sources is what phase 4's NLI
verification exists to do — wiring an LLM into this harness before that
exists would mean eyeballing free text and calling it measurement.

## Retrieval ablations (phase 3)

Every retrieval component conforms to one `Retriever` protocol —
`search(query, top_k, lang) -> list[SearchHit]` — so each ablation below is
a different object passed to the same `eval.harness.evaluate()`, not a
separate code path. `scripts/ablate.py` runs all seven against
`gold_v1.jsonl` and prints a comparison table.

| # | ablation | what changes vs. the phase-1 baseline |
|---|---|---|
| 3.1 | baseline | nothing — dense, `structural` chunks, e5-large, top-5 |
| 3.2 | chunking | `fixed` chunks instead of `structural` (needs its own index) |
| 3.3 | embedder | `BAAI/bge-m3` instead of e5-large (needs its own index) |
| 3.4 | sparse | BM25 instead of dense — see below |
| 3.5 | hybrid | Reciprocal Rank Fusion of the baseline dense store + BM25 |
| 3.6 | rerank | over-fetch top-20, rerank to top-5 with a cross-encoder |
| 3.7 | cross-lingual | each query restricted to chunks in its own language |

**BM25 is the one ablation this sandbox can score for real.** SudachiPy
ships its dictionary in the wheel, so Japanese morphological segmentation —
mandatory for BM25 on a language with no word boundaries, same as it is for
token counting — needs no Hugging Face access and no network at all.
`src/kensho/retrieval/segment.py` / `sparse.py` are tested against real
segmentation, not a fake stand-in, and a real run against the full corpus
and gold set scores recall@1 = 0.65, recall@5 = 0.90 — genuinely strong,
because most of this gold set's questions share exact vocabulary with their
source page, which is exactly the case lexical search is good at. That
result comes with an honest limit stated up front, not discovered later:
BM25 matches literal terms, so a Japanese query cannot retrieve an
English-only passage or vice versa — there is no shared vocabulary to match
against. Dense embedding search is the only retriever here that can cross
the language boundary at all, which is what ablation 3.7 exists to measure.

Every other row — 3.1, 3.2, 3.3, the dense half of 3.5, the cross-encoder in
3.6, and 3.7 itself — needs Hugging Face Hub access this sandbox doesn't
have. `scripts/ablate.py --allow-fallback` runs the full comparison anyway,
with fake dense/rerank components, to prove the wiring works end to end;
the resulting numbers are plumbing only; a run on a machine with model
access is what produces the numbers worth reporting.

## Claim verification (phase 4)

The project's name (検証, "verification") is about this stage. A generated
answer is decomposed into claims, and each claim is checked against the
passages it was actually retrieved from — not the question, not general
knowledge. `verify/pipeline.py`'s `verify_answer()` produces a
`VerifiedAnswer` with a per-claim verdict and a `faithfulness_rate`; deciding
what to *do* with an unsupported or contradicted claim (strike it,
re-retrieve, refuse) is phase 5.

**Decomposition** has two granularities, compared rather than assumed:
`SentenceSplitDecomposer` (real, reuses the script-aware sentence splitter
already built for chunking — no model) and `LLMDecomposer` (asks the
generator to rewrite its answer as one atomic claim per line — finer-grained,
but decomposition errors are a real failure mode of this method, which is
exactly why the cruder sentence-level method exists to compare against).

**Verification** is a 4-arm comparison, cheapest to most expensive:

| arm | needs | can detect contradiction? |
|---|---|---|
| `lexical` — token-coverage overlap | nothing (real, offline) | no |
| `embedding` — cosine similarity | an `Embedder` | no |
| `nli` — mDeBERTa entailment model | Hugging Face Hub | **yes** |
| `llm-judge` — ask the generator's LLM | Groq | yes, but correlated with the generator's own failure mode |

Lexical overlap and embedding similarity share a structural blind spot: "Pods
restart automatically" and "Pods never restart automatically" share almost
all their vocabulary and topic, so both methods read a direct negation as
(nearly) supported. The NLI model is the only arm actually trained to tell
entailment from contradiction — measuring that gap, not asserting it, is
what `scripts/compare_verifiers.py` and `data/gold/faithfulness_v1.jsonl`
are for.

`faithfulness_v1.jsonl` is 48 hand-written items (36 EN, 12 JA) across 12
real Kubernetes concepts, again reused from the pages read for the retrieval
gold set. Each topic has three claims against the same real evidence
passage: one that restates it (`supported`), one with a specific fabricated
detail the evidence never states (`unsupported`), and one that directly
negates something the evidence does state (`contradicted`).

**Lexical overlap is the one arm this sandbox can score for real** — it
needs no model. A real run against the full set: **0.646 accuracy, 0.938
recall on unsupported claims, 0.0 recall on contradicted claims** — the
0.0 isn't a bug, it's the predicted structural limitation confirmed
empirically: this method has no path to ever output "contradicted" at all.
`embedding` (via `FakeEmbedder`), `nli` (Hugging Face blocked), and
`llm-judge` (no Groq key here) all need `--allow-fallback` in this sandbox,
which makes those rows plumbing-only; a run on a machine with both HF and a
Groq key is what turns the other three rows into a real comparison.

## Self-correction and refusal (phase 5)

Phase 4 only scores an answer's claims. `correct/policy.py` is what acts on
that score, with a deliberately simple, three-way policy:

- **contradicted** claims are always struck immediately — there is no retry
  path, because re-retrieving more evidence for a claim that's already
  actively wrong doesn't fix the claim, it just risks finding a passage a
  weaker verifier arm would misread as support.
- **unsupported** claims get exactly one retry: search specifically for
  *that claim's text* (not the original question) and re-verify against
  whatever comes back. The original top-k for the question sometimes just
  didn't surface the one passage that supports a specific detail; a
  claim-targeted search sometimes finds it.
- if what survives is too thin — below `refusal_threshold` (default 50%) of
  the original claims, or nothing at all — the whole answer is replaced
  with an explicit, language-matched refusal rather than shown as a
  quietly-shortened one. A mutilated answer that still reads as confident
  is arguably worse than no answer.

`scripts/correct_ask.py` runs the full stack — retrieve, generate, decompose,
verify, correct — and prints exactly which claims were kept, struck, or
recovered, so a corrected answer is never a silent edit.

`scripts/eval_refusal.py` is what the phase-2 gold set's 4 deliberately
unanswerable items were seeded for: it runs every one of the 52 gold items
(48 answerable, 4 not) through the full pipeline and reports two numbers in
both directions, because a policy that refuses everything scores perfectly
on one of them and uselessly on the other — **false refusal rate** (real
questions answered with "I don't know" for no reason) and **miss rate**
(unanswerable questions the pipeline answered anyway, the confident-looking
hallucination this project exists to catch). Both directions need a real
generator and a real verifier to mean anything; with `--allow-fallback`'s
Echo LLM the pipeline still runs end to end, but Echo's placeholder text is
uniformly unsupported by every source, so it refuses all 52 items — a
correct demonstration of the mechanism, not a result about refusal quality.

## Serving, tracing, and eval-gated CI (phase 6)

`serve/app.py` wraps the full ask → verify → correct stack in a FastAPI
service, loading every component once at startup rather than per request —
the difference between a demo that answers in two seconds and one that
reloads an embedding model on every call. Configuration is entirely
environment-driven (`serve/config.py`): set `GROQ_API_KEY` and
`KENSHO_INDEX_DIR` for the real thing, or `KENSHO_ALLOW_FALLBACK=true` alone
to run the whole service on fake/echo components with no external
dependency at all.

```
GET  /healthz   -> {status, index_size, embedder, llm, verifier}
POST /ask       -> {trace_id, answer, refused, refusal_reason, claims, sources, latency_ms}
```

Every `/ask` call writes a `Trace` — the retrieved chunks, the raw
generation, every claim's verdict and correction action, and a latency
breakdown by stage — as one JSON line to `KENSHO_TRACE_PATH`. Two reasons
this exists rather than just logging the final answer: a support engineer
asking "why did it say that" can see the whole decision without re-running
the query, and real production questions become future gold-set candidates
precisely because their retrieval and verification results were captured at
the time, not reconstructed afterward from a one-line log of the answer.

`scripts/eval_gate.py` is what CI runs on every push (see
`.github/workflows/ci.yml`). It's deliberately narrow: BM25 recall@5 against
`gold_v1.jsonl` and lexical-overlap accuracy against `faithfulness_v1.jsonl`
— the two metrics in this whole project that need no Hugging Face access,
no Groq key, and no network beyond the corpus checkout, so they're the only
ones a CI runner can be assumed to reproduce reliably and fast. Floors live
in `data/eval_floor.json`, set with headroom below the real measured values
(BM25 recall@5 = 0.896, lexical accuracy = 0.646) so the gate catches an
actual regression — a broken tokenizer, a corpus-parsing bug that empties
out chunk text — without being flaky on noise. It exits 1 and fails the
build the moment either number drops below its floor; the embedding,
reranker, NLI, and LLM-judge arms are not gated this way, and extending CI
to cover them would need a cached model and a Groq key configured as CI
secrets, which is future work, not a limitation this gate pretends to solve.

## Deployment

`docker compose up` runs the full service locally in demo mode (fake
embedder, echo LLM, zero external dependencies) — see the `Dockerfile` and
`docker-compose.yml`. `deploy/hf-space/DEPLOY.md` covers pushing this same
image to a Hugging Face Space, including why HF Spaces rather than
Render/Railway (checked against actual free-tier specs, not assumed) and
the two consequences that follow from its CPU-bound, non-persistent-disk
free tier.

## Layout

```
README.md            technical reference (this file)
WRITEUP.md            narrative case study: problem, architecture, decisions,
                      real-vs-plumbing results, limitations
Dockerfile / docker-compose.yml / docker/entrypoint.sh
                      one-command reproducible run, demo mode by default
deploy/hf-space/      Hugging Face Space config + step-by-step deploy guide
docs/architecture.svg pipeline diagram (embedded above)
src/kensho/
  schema.py          Document and Chunk; parallel_id carries JA/EN identity
  tokenizer.py       pluggable token counting, script-aware fallback
  embed.py           pluggable embedding, e5 query/passage prefixing
  store.py           Qdrant wrapper, embedded mode
  llm.py             pluggable generation provider (Groq / Echo)
  prompts.py         grounded-answering prompt template
  pipeline.py        ask() — retrieve then generate, phase 1 control arm
  corpus/
    fetch.py         sparse checkout, language pairing, corpus statistics
    parse.py         Hugo front matter, shortcodes, heading hierarchy
    chunk.py         structural and fixed strategies
  eval/
    gold.py          GoldItem schema, load/save, cross-checked invariants
    metrics.py       recall@k, MRR, nDCG@k over deduplicated page rankings
    harness.py       evaluate() — runs the gold set, aggregates by language
  retrieval/
    segment.py       BM25 tokenization: real Sudachi, char-bigram fallback
    sparse.py        BM25Store — lexical retrieval, in-memory
    rerank.py        cross-encoder reranking, pluggable (real / fake)
    hybrid.py        HybridRetriever — Reciprocal Rank Fusion of two Retrievers
  verify/
    claims.py        claim decomposition — sentence-split or LLM-based
    nli.py            the 4 verification arms + get_verifier() dispatch
    pipeline.py       verify_answer() — scores an Answer's claims
  correct/
    policy.py        correct_answer() — strike / retry / refuse
  serve/
    config.py        ServerConfig, entirely environment-driven
    trace.py         Trace / TraceWriter — one JSON line per request
    app.py           FastAPI app: GET /healthz, POST /ask
scripts/
  build_corpus.py    corpus -> chunk JSONL + manifest
  build_index.py     chunk JSONL -> Qdrant store
  ask.py             end-to-end CLI query
  seed_gold_v1.py    hand-authored gold set -> data/gold/gold_v1.jsonl
  validate_gold.py   checks gold pages still exist in a given corpus build
  run_eval.py        gold set + index -> retrieval metrics + JSON report
  ablate.py          runs all seven ablations, prints a comparison table
  seed_faithfulness_v1.py   hand-authored claims -> faithfulness_v1.jsonl
  compare_verifiers.py      runs the 4-arm verification comparison
  correct_ask.py            full pipeline: ask -> verify -> correct
  eval_refusal.py           scores refusal decisions on the gold set's
                            answerable/unanswerable split
  serve.py                  runs the FastAPI app with uvicorn
  eval_gate.py              CI regression gate: BM25 recall@5 +
                            lexical-overlap faithfulness vs. eval_floor.json
tests/               216 tests, several of them regressions (see below)
```

## Bugs worth knowing about

Two were found by building the corpus and looking at the output, and both
would have degraded results silently rather than failing loudly. Each now has
a regression test.

**Shortcodes carry content.** Deleting every `{{< ... >}}` tag is the obvious
first pass and it is wrong: `glossary_tooltip` holds the displayed technical
term in a `text=` attribute, and there are ~2,000 of them across both
languages. Stripping them wholesale removed precisely the nouns users search
for, leaving debris like `各cloud-controller-managerは複数のを実装します` — a
sentence with its object deleted. Paired tags such as `{{< note >}}` need the
opposite treatment: drop the tags, keep the prose between them, since that is
where most of the corpus's caveats live.

**Small sections inherited the wrong heading.** A section too short to stand
alone was merged into the previous chunk and took on *that* chunk's section
path. Where the neighbour was a sibling rather than an ancestor, text from
`A/B2` ended up labelled `A/B1` — so the system would have cited a heading the
text never came from. Merged chunks now re-label to the deepest common
ancestor.

## Roadmap

| phase | | status |
|---|---|---|
| 0 | Corpus and scaffold | done |
| 1 | Baseline dense retrieval + generation | done |
| 2 | Evaluation harness and hand-labelled gold set | done |
| 3 | Retrieval ablations (7) | done |
| 4 | Claim decomposition and NLI verification | done |
| 5 | Self-correction and refusal | done |
| 6 | Serving, tracing, eval-gated CI | done |
| 7 | Write-up | done |

Phase 2 is deliberately built before phase 3. Every retrieval improvement is
then measured rather than assumed, which is the difference between a project
and a tutorial.

## Licence

Project code: MIT. Corpus content: CC BY 4.0, © the Kubernetes authors —
redistributed under the terms of that licence.
