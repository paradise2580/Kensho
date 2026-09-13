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
CHUNKS_FILE="${KENSHO_CHUNKS:-data/chunks/structural_t512_o64.jsonl}"
MARKER="$INDEX_DIR/.kensho_embedder"

# Sparse (BM25) retrieval indexes chunk text in memory at startup and never
# touches the vector store, so none of the index logic below applies. It
# also loads no embedding model, which is what lets this run with real
# retrieval on a host too small for model weights.
DEFAULT_CHUNKS="data/chunks/structural_t512_o64.jsonl"

if [ "${KENSHO_RETRIEVER:-dense}" = "sparse" ]; then
  if [ -f "$CHUNKS_FILE" ]; then
    echo "[entrypoint] sparse retrieval: using existing $CHUNKS_FILE"
  elif [ "$CHUNKS_FILE" = "$DEFAULT_CHUNKS" ]; then
    echo "[entrypoint] sparse retrieval needs $CHUNKS_FILE — building the corpus"
    python scripts/build_corpus.py --strategy structural --target-tokens 512 \
      --tokenizer heuristic
  else
    # build_corpus.py writes to its own default path, so building here would
    # produce a file somewhere other than where KENSHO_CHUNKS points and then
    # fail at startup anyway — with 13s of misleading "building" output first.
    echo "[entrypoint] KENSHO_CHUNKS=$CHUNKS_FILE does not exist, and building" >&2
    echo "[entrypoint] would write to $DEFAULT_CHUNKS instead. Point KENSHO_CHUNKS" >&2
    echo "[entrypoint] at a file that exists, or unset it to use the default." >&2
    exit 1
  fi
  exec python scripts/serve.py
fi

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
