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
    KENSHO_TRACE_PATH=/app/data/traces/kensho.jsonl \
    PORT=8000

EXPOSE 8000

ENTRYPOINT ["docker/entrypoint.sh"]
