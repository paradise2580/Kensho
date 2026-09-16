# Running the real (non-fallback) numbers

Everything below needs Hugging Face Hub access (for the real embedder,
reranker, and NLI model) and a Groq API key (for real generation and the
LLM-judge verifier) — neither is available in the sandbox this project was
built in, which is why README.md and WRITEUP.md mark these results as
plumbing-only rather than measured. Nothing here changes any code; it's the
exact command sequence to turn those into real, reportable numbers.

## 0. Prerequisites

```bash
pip install -e ".[dev,corpus,retrieval,generation,verify,sparse]"
export GROQ_API_KEY=...        # see .env.example
huggingface-cli login          # only needed if you hit a rate limit anonymously;
                                # none of the models this project uses are gated
```

Confirm access before spending time on the rest:

```bash
python -c "from sentence_transformers import SentenceTransformer; \
  SentenceTransformer('intfloat/multilingual-e5-small')"
```

## 1. Build the corpus with the real tokenizer

`--strict-tokenizer` fails loudly instead of silently falling back to the
heuristic — use it for any run whose numbers you intend to report.

```bash
python scripts/build_corpus.py --strategy structural --target-tokens 512 --strict-tokenizer
python scripts/build_corpus.py --strategy fixed --target-tokens 512 --strict-tokenizer
```

## 2. Build the three indexes phase 3's ablations need

Ablation 3.2 (chunking) and 3.3 (embedder) each need their own index; 3.1,
3.4–3.7 all reuse the baseline one built here.

```bash
# baseline: structural chunks, multilingual-e5-large (used by 3.1, 3.5, 3.6, 3.7)
python scripts/build_index.py data/chunks/structural_t512_o64.jsonl

# ablation 3.2: fixed chunks, same embedder
python scripts/build_index.py data/chunks/fixed_t512_o64.jsonl --index-dir data/qdrant_fixed

# ablation 3.3: structural chunks, a different embedder
python scripts/build_index.py data/chunks/structural_t512_o64.jsonl \
  --embedder BAAI/bge-m3 --index-dir data/qdrant_bge_m3
```

Each of these is a real embedding run over 8,446 (or 5,413) chunks — on CPU
this is the slowest step, budget real time for it rather than expecting it
to finish in seconds.

## 3. Run the retrieval ablations

```bash
python scripts/ablate.py data/gold/gold_v1.jsonl
python scripts/ablate.py data/gold/gold_v2.jsonl   # broader, 5 additional topics
```

This fills in ablations 3.1, 3.2, 3.3, 3.5's dense half, 3.6, and 3.7 — the
only row already real without this (3.4, BM25) will print the same numbers
already in the README, which is a useful sanity check that nothing else
changed underneath it.

## 4. Run the 4-arm verification comparison

```bash
python scripts/compare_verifiers.py data/gold/faithfulness_v1.jsonl
python scripts/compare_verifiers.py data/gold/faithfulness_v2.jsonl
```

This is what turns the `embedding`, `nli`, and `llm-judge` rows from
plumbing into a real comparison — in particular, whether the NLI arm
actually closes the 0.0-recall-on-contradicted gap that lexical overlap
structurally cannot.

## 5. Run refusal evaluation with a real generator

```bash
python scripts/eval_refusal.py data/gold/gold_v1.jsonl
```

With a real LLM instead of Echo, this produces the actual false-refusal
rate and miss rate — the numbers the README explicitly says are not yet
measured, since Echo's placeholder text refuses all 52 items by
construction and says nothing about a real generator's refusal quality.

## 6. Try the full corrected pipeline end to end

```bash
python scripts/correct_ask.py "How does a Pod's restart policy work?"
python scripts/correct_ask.py "Pod の再起動ポリシーはどう動作しますか?"
```

## 7. (Optional) See the real margin against the CI floors

`scripts/eval_gate.py` only ever scores BM25 and lexical-overlap — it
doesn't use any of the components above — so this step isn't necessary to
"unlock" anything, but it's a quick way to confirm the floors in
`data/eval_floor.json` still have real headroom:

```bash
python scripts/eval_gate.py
```

## 8. Compare the two sparse-retrieval backends

Both need no network and no model weights this sandbox lacks — this is the
one comparison in this file runnable anywhere, including CI:

```bash
python scripts/compare_sparse_backends.py
```

Runs `BM25Store` (in-memory, ~390MB resident) and `FTS5Store` (disk-backed,
the `KENSHO_SPARSE_BACKEND=fts5` option) against both gold sets and prints
recall@{1,3,5,10} and MRR side by side. As of the exact-formula scorer in
`retrieval/fts5.py` (computing Okapi BM25 with this project's own k1/b/
epsilon rather than SQLite's built-in `bm25()`, which is hardcoded to a
different k1), the two backends measure identically:

```
data/gold/gold_v1.jsonl  (n=48 answerable / 52 total)
  metric           memory       fts5      delta
  recall@1          0.646      0.646     +0.000
  recall@3          0.854      0.854     +0.000
  recall@5          0.896      0.896     +0.000
  recall@10         0.917      0.917     +0.000
  MRR               0.753      0.753     +0.000
  top-1 hit differs on 0/52 items

data/gold/gold_v2.jsonl  (n=58 answerable / 62 total)
  metric           memory       fts5      delta
  recall@1          0.690      0.690     +0.000
  ...                                    +0.000
  top-1 hit differs on 0/62 items
```

If this ever shows a nonzero delta again — after a corpus rebuild, a
segmenter change, anything — that's a signal the two scoring paths have
drifted, not an expected source of noise; `tests/test_fts5.py`'s
`test_matches_rank_bm25_okapi_exactly` is the unit-level version of the
same claim and should be the first thing to fail.

## What to update once you have these numbers

- README.md's "Retrieval ablations (phase 3)" and "Claim verification
  (phase 4)" sections — replace "plumbing-only" language with the measured
  table.
- WRITEUP.md's "What's real and what isn't" table — move the newly-measured
  rows out of the "wired but unmeasured" paragraph.
- `scripts/eval_refusal.py`'s output — replace the Echo-LLM caveat with the
  real false-refusal-rate / miss-rate pair.

None of this requires touching `data/eval_floor.json` — those floors are
deliberately scoped to the two network-free metrics only (see
`scripts/eval_gate.py`'s docstring for why), and stay valid regardless of
what the real-model numbers turn out to be.
