/**
 * The five stages a request passes through, and — deliberately — which two of
 * them any RAG system already has. Stages 1 and 2 are standard retrieval-
 * augmented generation. Stages 3 to 5 are what this project adds: the answer
 * is taken apart, each piece is checked against the sources it claims to come
 * from, and anything that fails is struck, re-grounded, or escalated to a
 * refusal.
 *
 * No per-stage millisecond counters here. The service reports three timing
 * buckets, not five, so a number under every stage would be invented — the
 * measured split lives in <LatencyBar /> where it is real.
 */
const STAGES = [
  {
    n: 1,
    name: 'Retrieve',
    note: 'BM25 or dense search over the bilingual chunk index',
    tag: 'standard rag',
  },
  {
    n: 2,
    name: 'Generate',
    note: 'Grounded answer with inline [n] citations',
    tag: 'standard rag',
  },
  {
    n: 3,
    name: 'Decompose',
    note: 'Split the answer into individually checkable claims',
    tag: 'kenshō',
  },
  {
    n: 4,
    name: 'Verify',
    note: 'Entail each claim against the source it cites',
    tag: 'kenshō',
  },
  {
    n: 5,
    name: 'Correct',
    note: 'Keep, strike, re-ground — or refuse outright',
    tag: 'kenshō',
  },
]

export default function StageTrack({ phase }) {
  return (
    <div className="card">
      <div className="card-head">
        <h2 className="card-title">Request path</h2>
        <span className="dim" style={{ fontSize: 11.5 }}>
          {phase === 'running'
            ? 'In flight — per-stage timings are reported on completion'
            : phase === 'done'
              ? 'Completed'
              : 'Idle'}
        </span>
      </div>
      <div className="stages">
        {STAGES.map((s) => (
          <div
            key={s.n}
            className={
              'stage' +
              (phase === 'running' ? ' stage--active' : phase === 'done' ? ' stage--done' : '')
            }
          >
            <div className="stage-top">
              <span className="stage-num">{s.n}</span>
              <span className="stage-name">{s.name}</span>
            </div>
            <div className="stage-note">{s.note}</div>
            <span
              className={
                'stage-tag ' +
                (s.tag === 'kenshō' ? 'stage-tag--kensho' : 'stage-tag--std')
              }
            >
              {s.tag}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
