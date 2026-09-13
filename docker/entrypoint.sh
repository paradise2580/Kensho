#!/bin/sh
# Builds the corpus/index once, then starts the API server.
#
# Fully offline by default (KENSHO_ALLOW_FALLBACK=true, no GROQ_API_KEY
# needed) so `docker run` works with zero configuration. Set GROQ_API_KEY
# and unset KENSHO_ALLOW_FALLBACK for the real thing — see README.md's
# Serving section for the full environment-variable list.
#
# Why the marker file
# -------------------
# The Dockerfile prebuilds a demo index at image-build time so a cold
# container answers immediately instead of going dark for ~45s. That
# creates a trap: "an index already exists" is NOT the same question as
# "an index built with the embedder we are about to serve with". Skipping
# the rebuild on existence alone would quietly serve fake 64-dim vectors
# to a deployment configured for the real embedder — a wrong answer that
# looks like a working one, which is the exact failure mode this whole
# project exists to avoid. So the builder records which embedder it used
# and we rebuild whenever that disagrees with what is being asked for.
set -eu

KENSHO_ALLOW_FALLBACK="${KENSHO_ALLOW_FALLBACK:-true}"
export KENSHO_ALLOW_FALLBACK

INDEX_DIR="${KENSHO_INDEX_DIR:-data/qdrant}"
CHUNKS_FILE="data/chunks/structural_t512_o64.jsonl"
MARKER="$INDEX_DIR/.kensho_embedder"

# The embedder this process would serve with. Mirrors get_embedder()'s own
# resolution order: an explicit KENSHO_EMBEDDER wins; otherwise fallback
# mode means the fake embedder, and non-fallback mode means the configured
# real default.
if [ -n "${KENSHO_EMBEDDER:-}" ]; then
  DESIRED="$KENSHO_EMBEDDER"
elif [ "$KENSHO_ALLOW_FALLBACK" = "true" ]; then
  DESIRED="fake"
else
  DESIRED="default"
fi

BUILT=""
if [ -f "$MARKER" ]; then
  BUILT="$(cat "$MARKER")"
fi

if [ -n "$BUILT" ] && [ "$BUILT" = "$DESIRED" ] && [ -n "$(ls -A "$INDEX_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] index at $INDEX_DIR was built with '$BUILT' — reusing it"
else
  if [ -n "$BUILT" ]; then
    echo "[entrypoint] index at $INDEX_DIR was built with '$BUILT' but this" \
         "deployment wants '$DESIRED' — rebuilding rather than serving" \
         "vectors from the wrong embedder"
    rm -rf "$INDEX_DIR"
  else
    echo "[entrypoint] no usable index at $INDEX_DIR — building one"
  fi

  if [ "$DESIRED" = "fake" ]; then
    echo "[entrypoint] building an offline demo index (heuristic tokenizer, fake embedder)."
    python scripts/build_corpus.py --strategy structural --target-tokens 512 \
      --tokenizer heuristic
    python scripts/build_index.py "$CHUNKS_FILE" --embedder fake --allow-fallback \
      --index-dir "$INDEX_DIR"
  else
    echo "[entrypoint] building the real index (needs Hugging Face Hub access)."
    python scripts/build_corpus.py --strategy structural --target-tokens 512 --strict-tokenizer
    if [ "$DESIRED" = "default" ]; then
      python scripts/build_index.py "$CHUNKS_FILE" --index-dir "$INDEX_DIR"
    else
      python scripts/build_index.py "$CHUNKS_FILE" --embedder "$DESIRED" --index-dir "$INDEX_DIR"
    fi
  fi

  printf '%s' "$DESIRED" > "$MARKER"
fi

exec python scripts/serve.py
