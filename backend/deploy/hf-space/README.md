---
title: Kenshō
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Kenshō 検証 — live demo

A bilingual (Japanese / English) enterprise-support RAG assistant that
verifies every sentence it writes against the passages it retrieved, and
refuses to answer when it cannot. Full source, architecture, and
measured results: see the project's GitHub repository and `WRITEUP.md`.

`GET /healthz` and `POST /ask` are the two endpoints — see the main
README's "Serving" section for the request/response shape.

This Space runs in `KENSHO_ALLOW_FALLBACK=true` demo mode unless
`GROQ_API_KEY` (and, for the real embedder, Hugging Face Hub access) is
configured in the Space's secrets — see `deploy/hf-space/DEPLOY.md` in the
repo for exactly which secrets turn on which real component.
