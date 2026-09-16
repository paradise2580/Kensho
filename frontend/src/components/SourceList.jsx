/**
 * Retrieved chunks, in rank order, with the score that put them there.
 *
 * BM25 scores are unbounded and corpus-relative, so an absolute-width bar
 * would be meaningless. The bars are normalised to the top hit of *this*
 * result set: they answer "how far behind the winner is each of these", which
 * is the only question the number can honestly answer. The raw score sits
 * beside the bar so nothing is hidden behind the normalisation.
 *
 * One series, one hue — no legend, because the column header already says what
 * is plotted.
 */
export default function SourceList({ sources }) {
  if (!sources?.length) {
    return (
      <div className="card">
        <div className="card-head">
          <h2 className="card-title">Retrieved sources</h2>
        </div>
        <div className="empty">Nothing was retrieved for this question.</div>
      </div>
    )
  }

  const max = Math.max(...sources.map((s) => s.score), 1e-9)

  return (
    <div className="card">
      <div className="card-head">
        <h2 className="card-title">Retrieved sources</h2>
        <span className="dim" style={{ fontSize: 11.5 }}>
          {sources.length} chunks · score relative to top hit
        </span>
      </div>
      <div className="card-body">
        <div className="sources">
          {sources.map((s, i) => (
            <div className="source" key={`${s.parallel_id}-${i}`}>
              <span className="source-n">[{i + 1}]</span>
              <div>
                <div className={'source-title' + (s.lang === 'ja' ? ' jp' : '')}>
                  {s.title || <span className="dim">untitled chunk</span>}
                </div>
                <div className="source-id">{s.parallel_id}</div>
              </div>
              <div className="source-right">
                <span className="lang-chip">{s.lang || '—'}</span>
                <div className="score">
                  <div className="score-track">
                    <div
                      className="score-fill"
                      style={{ width: `${Math.max(2, (s.score / max) * 100)}%` }}
                    />
                  </div>
                  <span className="score-val">{s.score.toFixed(2)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
