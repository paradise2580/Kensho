# The console

A React single-page app that talks to the Kenshō service over the same HTTP API
any other client would use. `GET /` serves it; `POST /ask` and `GET /healthz` are
the only two endpoints it knows about.

```
src/
  api.js                 the only module that knows the service exists
  App.jsx                state, health polling, request lifecycle
  components/
    Header.jsx           brand, live health chips, theme toggle, tabs
    AskBar.jsx           question, language filter, top-k, examples
    StageTrack.jsx       the five stages, and which two are standard RAG
    AnswerCard.jsx       final text, inline citations, refusal banner, export
    ClaimList.jsx        per-claim verdict and action — the point of the project
    SourceList.jsx       retrieved chunks with normalised scores
    LatencyBar.jsx       the three measured timing buckets
    Benchmarks.jsx       measured results, and what is explicitly not measured
    Architecture.jsx     request path, decisions, engineering surface
```

## Running it

The built bundle in `dist/` is committed, so the console is already being served
by the API — start the service and open it:

```bash
uvicorn kensho.serve.app:create_app --factory
# http://127.0.0.1:8000
```

To work on the UI itself, run Vite's dev server alongside it. Vite proxies
`/ask` and `/healthz` to uvicorn on :8000, so the front end always talks to the
real service:

```bash
npm install
npm run dev        # http://127.0.0.1:5173
npm run build      # regenerate dist/
```

## Why `dist/` is committed

Normally a build artifact does not belong in version control. Here it buys
something specific: `uvicorn kensho.serve.app:create_app --factory` is a
*complete* instruction for running this project, UI included, on a machine with
Python and nothing else. No Node, no npm install, no build step, no chance that
a reviewer sees a 404 where the interface should be. The service degrades
honestly if the build is ever missing — `GET /` returns a 404 explaining how to
build it, and the API keeps serving.

The trade is that `dist/` has to be rebuilt and committed alongside any change
to `src/`. `tests/test_app.py::TestConsole` asserts the built bundle still
references `/ask` and `/healthz`, so a bundle that has drifted into being a mock
rather than a client of the service fails the build.

## Two things that are load-bearing

**There is no fixture mode.** Nothing on the console is rendered from stored
data. If the service is down, the page says so; if a request fails, the page
shows the error. A demo that quietly falls back to a canned answer would be
doing precisely what this project exists to detect.

**Colour never carries a verdict on its own.** supported/contradicted is a
green/red pair, and green/red is the pair a deuteranope cannot separate —
measured ΔE 4.1 against this surface. Every verdict badge therefore carries a
glyph and a written label; the colour is an accent on top of them, never the
channel itself.
