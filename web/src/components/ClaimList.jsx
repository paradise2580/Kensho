/**
 * The panel the whole project exists for.
 *
 * `label` is the verifier's verdict on whether the source entails the claim;
 * `action` is what the correction policy then did about it. They are separate
 * on purpose — a contradicted claim can still be recovered if a *different*
 * retrieved chunk supports it, and that distinction is invisible if you
 * collapse verdict and action into one status.
 *
 * Colour never carries the verdict alone. supported/contradicted is a
 * green/red pair, and green/red is the one pair a deuteranope cannot separate
 * (measured ΔE 4.1 against this surface). Every badge therefore ships a
 * monospace glyph and a written label; the colour is the accent on top.
 */
const VERDICT = {
  supported: { color: 'var(--good)', glyph: '✓', text: 'supported' },
  contradicted: { color: 'var(--critical)', glyph: '✗', text: 'contradicted' },
  unsupported: { color: 'var(--warning)', glyph: '?', text: 'unsupported' },
}

const ACTION = {
  kept: { glyph: '●', text: 'kept', accent: false },
  struck: { glyph: '✕', text: 'struck from answer', accent: false },
  recovered: { glyph: '↻', text: 'recovered from another source', accent: true },
}

function Badge({ color, glyph, text, plain }) {
  return (
    <span className={plain ? 'badge badge--plain' : 'badge'} style={plain ? undefined : { '--badge-color': color }}>
      <span className="badge-glyph" style={plain ? { color: 'var(--text-3)' } : undefined}>
        {glyph}
      </span>
      {text}
    </span>
  )
}

function Tally({ claims }) {
  const counts = claims.reduce((acc, c) => {
    acc[c.label] = (acc[c.label] || 0) + 1
    return acc
  }, {})

  const order = ['supported', 'contradicted', 'unsupported'].filter((k) => counts[k])
  if (!order.length) return null

  return (
    <div className="tally">
      {order.map((k) => (
        <span className="tally-item" key={k}>
          <span className="dot" style={{ background: VERDICT[k].color }} />
          <b>{counts[k]}</b> {VERDICT[k].text}
        </span>
      ))}
    </div>
  )
}

export default function ClaimList({ claims }) {
  if (!claims?.length) {
    return (
      <div className="card">
        <div className="card-head">
          <h2 className="card-title">Claim verification</h2>
        </div>
        <div className="empty">
          No claims were extracted from this answer.
          <br />
          That happens when the generator refused outright, so there was nothing to check.
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-head">
        <h2 className="card-title">Claim verification</h2>
        <Tally claims={claims} />
      </div>
      <div className="card-body">
        <div className="claims">
          {claims.map((c, i) => {
            const v = VERDICT[c.label] || {
              color: 'var(--text-3)',
              glyph: '·',
              text: c.label,
            }
            const a = ACTION[c.action] || { glyph: '·', text: c.action, accent: false }
            return (
              <div
                key={i}
                className={'claim' + (c.action === 'struck' ? ' claim--struck' : '')}
                style={{ '--claim-color': v.color }}
              >
                <span className="claim-idx">{String(i + 1).padStart(2, '0')}</span>
                <div>
                  <div className="claim-text">{c.text}</div>
                  <div className="claim-meta">
                    <Badge color={v.color} glyph={v.glyph} text={v.text} />
                    <Badge
                      color="var(--accent)"
                      glyph={a.glyph}
                      text={a.text}
                      plain={!a.accent}
                    />
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
