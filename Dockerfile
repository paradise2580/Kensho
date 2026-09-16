# Kenshō serving image. Builds the corpus and index at container start
# (entrypoint.sh), then runs the FastAPI service — see README.md's
# "Serving" section for the environment variables that switch between the
# real backends and the offline fake/echo ones.
FROM python:3.11-slim

WORKDIR /app

# git is needed by corpus/fetch.py's sparse checkout of kubernetes/website.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY data/gold ./data/gold
COPY data/eval_floor.json ./data/eval_floor.json
COPY docker/entrypoint.sh ./docker/entrypoint.sh

# [dev] pulls in pytest/ruff too, which the image doesn't need at runtime,
# but keeping install extras identical to CI's is one less thing to drift.
# Swap in the extras your deployment actually needs — sparse+serve is the
# minimum for a fully offline demo; add corpus,retrieval,generation,verify
# for the real (non-fallback) backends.
ARG EXTRAS=sparse,serve
RUN pip install --no-cache-dir -e ".[${EXTRAS}]"

RUN chmod +x docker/entrypoint.sh

ENV KENSHO_INDEX_DIR=/app/data/qdrant \
    KENSHO_SPARSE_INDEX=/app/data/index/bm25.db \
    KENSHO_TRACE_PATH=/app/data/traces/kensho.jsonl \
    PORT=8000

# Prebuild the offline demo corpus and index into the image.
#
# Doing this at container start instead costs ~45s of every cold boot (10s
# to clone kubernetes/website, 3s to chunk, 30s to index 8,446 vectors),
# during which the service is not answering. On a platform with
# non-persistent disk — Hugging Face Spaces, which this is packaged for —
# that is every restart, not just every deploy. Baking it in trades image
# size for a container that serves its first request immediately.
#
# This index is deliberately the FAKE-embedder demo one: building the real
# index here would need Hugging Face Hub access at image-build time and
# would bake a ~35MB float32 index into the layer. entrypoint.sh's marker
# check rebuilds it if the deployment actually asks for a real embedder.
# Also bakes the FTS5 sparse index (~30MB) so `-e KENSHO_RETRIEVER=sparse
# -e KENSHO_SPARSE_BACKEND=fts5` starts serving immediately too, on the
# same "don't make a non-persistent-disk host pay a build on every
# restart" logic as the dense index above - see
# src/kensho/retrieval/fts5.py for why this backend exists at all (the
# in-memory sparse backend needs no prebuild step, RAM is the trade there).
RUN KENSHO_ALLOW_FALLBACK=true python scripts/build_corpus.py \
        --strategy structural --target-tokens 512 --tokenizer heuristic \
 && KENSHO_ALLOW_FALLBACK=true python scripts/build_index.py \
        data/chunks/structural_t512_o64.jsonl \
        --embedder fake --allow-fallback --index-dir /app/data/qdrant \
 && printf '%s' fake > /app/data/qdrant/.kensho_embedder \
 && python scripts/build_sparse_index.py \
        data/chunks/structural_t512_o64.jsonl --db-path /app/data/index/bm25.db \
 && rm -rf data/k8s-website

EXPOSE 8000

ENTRYPOINT ["docker/entrypoint.sh"]
