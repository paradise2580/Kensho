/**
 * Where the wall-clock actually went, measured server-side.
 *
 * The three segments are the three timers the service reports — no more, no
 * fewer. It is worth looking at because the split is the honest cost of the
 * idea: verification is usually the largest segment, and a demo that hid that
 * would be selling something it doesn't deliver.
 *
 * Three categorical slots, separated by a 2px surface gap rather than strokes,
 * with a legend always present and every value directly labelled.
 */
const SEGMENTS = [
  {
    key: 'retrieval_and_generation',
    name: 'Retrieve + generate',
    color: 'var(--series-1)',
  },
  { key: 'verification', name: 'Decompose + verify', color: 'var(--series-2)' },
  { key: 'correction', name: 'Correct / refuse', color: 'var(--series-3)' },
]

function fmt(ms) {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`
  return `${Math.round(ms)}ms`
}

export default function LatencyBar({ latency }) {
  if (!latency) return null

  const total = latency.total || 1
  const rows = SEGMENTS.map((s) => ({ ...s, ms: latency[s.key] ?? 0 }))

  return (
    <div className="card">
      <div className="card-head">
        <h2 className="card-title">Latency breakdown</h2>
      </div>
      <div className="card-body">
        <div className="lat-total">
          <b>{fmt(total)}</b>
          <span className="dim" style={{ fontSize: 12 }}>
            end to end
          </span>
        </div>

        <div
          className="lat-bar"
          role="img"
          aria-label={rows.map((r) => `${r.name} ${fmt(r.ms)}`).join(', ')}
        >
          {rows.map((r) => (
            <div
              key={r.key}
              className="lat-seg"
              style={{
                width: `${(r.ms / total) * 100}%`,
                background: r.color,
              }}
            />
          ))}
        </div>

        <div className="lat-legend">
          {rows.map((r) => (
            <div className="lat-row" key={r.key}>
              <span className="lat-key" style={{ background: r.color }} />
              <span className="lat-name">{r.name}</span>
              <span className="lat-ms">{fmt(r.ms)}</span>
              <span className="dim mono" style={{ fontSize: 11, minWidth: 40, textAlign: 'right' }}>
                {((r.ms / total) * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
