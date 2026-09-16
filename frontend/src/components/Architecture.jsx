const DECISIONS = [
  {
    h: 'The prompt asks for prose, and that is a correctness requirement',
    p: 'Claims are produced by splitting the answer on sentence boundaries. When the model replied with a markdown table, the whole table collapsed into one enormous "claim" covering a dozen separate assertions — a verdict on that is worthless, because it cannot say which cell was wrong, and one false cell condemns eleven true ones. The system prompt now forbids tables and lists. Verification only means something at a granularity the decomposer can actually reach, so output shape is part of the contract, not styling.',
  },
  {
    h: 'Verdict and action are two fields, not one status',
    p: 'The verifier says whether the cited source entails the claim. The correction policy then decides what to do about it. Keeping them separate is what makes "contradicted by source [2], but recovered from source [5]" expressible — collapse them and that case becomes indistinguishable from an outright failure.',
  },
  {
    h: 'Japanese is segmented morphologically, not on whitespace',
    p: 'Japanese does not delimit words with spaces, so a whitespace tokenizer hands BM25 one enormous token per sentence and retrieval quality collapses. SudachiPy does the segmentation. This is the single change that makes the bilingual half of the corpus work at all, and it is invisible in any English-only benchmark.',
  },
  {
    h: 'Chunks carry their heading path',
    p: 'A chunk lifted out of a document loses the context that made it findable. Prepending the section path — document title › section › subsection — puts the words a user would actually search for back into the indexed text, at the cost of a little redundancy across sibling chunks.',
  },
  {
    h: 'Refusal is a first-class outcome',
    p: 'When too little of the drafted answer survives verification, the service returns a refusal with a reason rather than the surviving fragments. An assistant that answers every question is not more useful than one that declines the ones its documents do not cover — it is just harder to trust.',
  },
  {
    h: 'The CI gate scores only what runs without a network',
    p: 'The build gates on BM25 recall@5 and lexical-overlap verification, because those need no model weights and no API key, so the floor is meaningful on every push from any machine. Gating on a number that requires a paid API would mean a gate that is quietly skipped.',
  },
]

const SURFACE = [
  ['Service', 'FastAPI — one process holds one loaded pipeline; models load at startup, not per request'],
  ['Boundaries', 'Retriever, LLMProvider, Embedder, Decomposer, ClaimVerifier — each a Protocol, each swappable by env var'],
  ['Console', 'React 18 + Vite, talking to the same HTTP API any other client would use'],
  ['Tracing', 'Every request appends a JSONL trace: retrieved chunks, per-claim verdicts, per-stage timings'],
  ['Tests', '230, including the full app in-process with fake components — no network, no weights'],
  ['CI', 'Tests, ruff, and an evaluation gate on recall@5 ≥ 0.896'],
  ['Packaging', 'Docker image; optional dependency groups so a sparse-only deploy pulls no torch'],
  ['Config', 'ServerConfig.from_env() — retriever, embedder, LLM, verifier, thresholds all injectable'],
]

const PATH = [
  { n: 1, t: 'Retrieve', d: 'The question is segmented (SudachiPy for Japanese), scored against the chunk index, and the top-k chunks come back with their scores and heading paths.', k: false },
  { n: 2, t: 'Generate', d: 'Those chunks go into the prompt as numbered sources. The model is told to answer only from them, to match the question\'s language, to cite inline, and to write prose.', k: false },
  { n: 3, t: 'Decompose', d: 'The draft is split into individually checkable claims. This is why stage 2 is told not to emit tables.', k: true },
  { n: 4, t: 'Verify', d: 'Each claim is checked against the source it cites, and labelled supported, contradicted, or unsupported.', k: true },
  { n: 5, t: 'Correct', d: 'Supported claims are kept. Unsupported ones are re-checked against the other retrieved chunks and recovered if any of them entails the claim. What survives neither is struck. If too little survives, the whole answer becomes a refusal.', k: true },
]

export default function Architecture() {
  return (
    <div className="stack">
      <div className="card">
        <div className="card-head">
          <h2 className="card-title">What one request does</h2>
          <span className="dim" style={{ fontSize: 11.5 }}>
            stages 1–2 are standard RAG · 3–5 are what this project adds
          </span>
        </div>
        <div className="card-body">
          <div className="notes">
            {PATH.map((s) => (
              <div className="note" key={s.n}>
                <h4 style={{ display: 'flex', gap: 9, alignItems: 'center' }}>
                  <span className="stage-num">{s.n}</span>
                  {s.t}
                  <span className={'stage-tag ' + (s.k ? 'stage-tag--kensho' : 'stage-tag--std')}>
                    {s.k ? 'kenshō' : 'standard rag'}
                  </span>
                </h4>
                <p>{s.d}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h2 className="card-title">Decisions worth defending</h2>
        </div>
        <div className="card-body">
          <div className="notes">
            {DECISIONS.map((d) => (
              <div className="note" key={d.h}>
                <h4>{d.h}</h4>
                <p>{d.p}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h2 className="card-title">Engineering surface</h2>
        </div>
        <div className="table-wrap">
          <table>
            <tbody>
              {SURFACE.map(([k, v]) => (
                <tr key={k}>
                  <td style={{ width: 130, color: 'var(--text-3)', fontWeight: 600 }}>{k}</td>
                  <td style={{ whiteSpace: 'normal' }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
