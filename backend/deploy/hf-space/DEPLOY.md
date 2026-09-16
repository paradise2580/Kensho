# Deploying Kenshō to a Hugging Face Space

This needs your own Hugging Face account and (for the real, non-demo
backends) your own Groq API key — neither is available in the sandbox this
project was built in, so this is a set of steps for you to run, not
something already done.

## Why HF Spaces, not Render/Railway/etc.

Checked during planning: HF Spaces' free tier is 2 vCPU / 16GB RAM / 50GB
**non-persistent** disk. RAM was never the constraint (Render's free tier
tops out around 512MB–2GB, too little for an embedding model plus a
reranker plus an NLI model loaded at once); CPU is — the reranker and NLI
verification stages can add real seconds of latency on 2 shared cores.
Two things follow directly from that, both already reflected in the code:

- **Non-persistent disk** means every cold start starts from the image
  again. Measured, the demo build costs ~45s (≈10s to sparse-clone
  kubernetes/website, ≈3s to chunk, ≈30s to index 8,446 vectors) — 45s of
  a Space that is up but not answering, on every restart rather than every
  deploy. So the Dockerfile now builds that demo index at *image-build*
  time and the container starts serving immediately.

  `docker/entrypoint.sh` therefore cannot decide whether to rebuild by
  asking "does an index exist" — it always will. It compares the embedder
  recorded in `data/qdrant/.kensho_embedder` against the embedder this
  deployment is configured to serve with, and rebuilds when they disagree.
  Without that check, setting `KENSHO_EMBEDDER` to a real model would
  silently keep serving the prebuilt 64-dim fake vectors: a wrong answer
  that looks like a working one.

  Real *model weights* are still not in the image and still re-download on
  a fresh container, so expect a slow first request whenever you run with
  a real embedder, reranker, or NLI verifier.
- **CPU-bound latency** is why the demo-mode default (`KENSHO_ALLOW_FALLBACK
  =true`) exists at all: it lets the Space come up and answer instantly with
  no model inference, so a visitor sees a working service immediately, with
  a clear path to turn on the real, slower-but-grounded pipeline.

## Steps

1. Create the Space: [huggingface.co/new-space](https://huggingface.co/new-space) →
   SDK: **Docker** → visibility your choice → create.
2. Clone it locally and copy this project's files in, replacing the
   Space's placeholder `README.md` with `deploy/hf-space/README.md` (the
   Spaces frontmatter has to be the literal root `README.md` of the Space
   repo — that's a Spaces requirement, not optional):

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   # copy everything from the kensho project except deploy/hf-space itself
   rsync -a --exclude='.git' --exclude='deploy/hf-space' /path/to/kensho/ ./
   cp /path/to/kensho/deploy/hf-space/README.md ./README.md
   git add -A
   git commit -m "Deploy Kenshō"
   git push
   ```

3. In the Space's **Settings → Variables and secrets**, add:
   - `KENSHO_ALLOW_FALLBACK` = `true` to start (safe, instant, zero cost —
     confirms the deploy works before spending API calls) or `false` once
     you're ready for the real thing.
   - `GROQ_API_KEY` — your key, as a **secret**, not a variable, once you
     set `KENSHO_ALLOW_FALLBACK=false`.
   - `KENSHO_EMBEDDER` = `intfloat/multilingual-e5-large` (or leave unset
     to keep the fake embedder even with a real LLM — a valid, cheaper
     partial step).
4. Rebuild the Space (Settings → Factory rebuild, or push any commit).
   Watch the build logs — `docker/entrypoint.sh` prints which mode it built
   the index in.
5. `curl https://<your-username>-<space-name>.hf.space/healthz` once it's
   live, then try `/ask` — see the main README's Serving section for the
   request shape.

## What this sandbox could not do for you

Everything above needs an authenticated `huggingface.co` account and (for
the real backends) a Groq API key. This build environment has neither and
its network access to `huggingface.co` is blocked outright — which is the
same restriction documented throughout the README wherever a result is
marked "plumbing-only" rather than measured. The Dockerfile, entrypoint
script, and this Space config are written and reviewed against the
project's actual CLI flags, but the live deploy itself is the one part of
this task that has to happen on your machine, with your credentials.
