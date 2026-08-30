#!/bin/sh
# Builds the corpus/index once (skipped if data/qdrant already has points —
# e.g. a mounted volume from a previous run) then starts the API server.
#
# Fully offline by default (KENSHO_ALLOW_FALLBACK=true, no GROQ_API_KEY
# needed) so `docker run` works with zero configuration. Set GROQ_API_KEY
# and unset KENSHO_ALLOW_FALLBACK for the real thing — see README.md's
# Serving section for the full environment-variable list.
set -eu

KENSHO_ALLOW_FALLBACK="${KENSHO_ALLOW_FALLBACK:-true}"
export KENSHO_ALLOW_FALLBACK

CHUNKS_FILE="data/chunks/structural_t512_o64.jsonl"

if [ ! -d "${KENSHO_INDEX_DIR:-data/qdrant}" ] || [ -z "$(ls -A "${KENSHO_INDEX_DIR:-data/qdrant}" 2>/dev/null)" ]; then
  echo "[entrypoint] no existing index at ${KENSHO_INDEX_DIR:-data/qdrant} — building one"

  if [ "$KENSHO_ALLOW_FALLBACK" = "true" ] && [ -z "${KENSHO_EMBEDDER:-}" ]; then
    echo "[entrypoint] KENSHO_ALLOW_FALLBACK=true and no KENSHO_EMBEDDER set:" \
         "building an offline demo index (heuristic tokenizer, fake embedder)."
    python scripts/build_corpus.py --strategy structural --target-tokens 512 \
      --tokenizer heuristic
    python scripts/build_index.py "$CHUNKS_FILE" --embedder fake --allow-fallback \
      --index-dir "${KENSHO_INDEX_DIR:-data/qdrant}"
  else
    echo "[entrypoint] building the real index (needs Hugging Face Hub access)."
    python scripts/build_corpus.py --strategy structural --target-tokens 512 --strict-tokenizer
    python scripts/build_index.py "$CHUNKS_FILE" --index-dir "${KENSHO_INDEX_DIR:-data/qdrant}"
  fi
else
  echo "[entrypoint] found an existing index at ${KENSHO_INDEX_DIR:-data/qdrant} — skipping build"
fi

exec python scripts/serve.py
