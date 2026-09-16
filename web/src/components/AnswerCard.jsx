import { useState } from 'react'

/**
 * Inline citations are rendered as marks rather than left as bare text so that
 * "[3]" reads as a pointer into the source list below it, not as punctuation.
 * The split keeps the delimiters (the capture group), so nothing in the
 * model's text is dropped on the way to the screen.
 */
function withCitations(text) {
  return text.split(/(\[\d+\])/g).map((part, i) =>
    /^\[\d+\]$/.test(part) ? (
      <span className="cite" key={i}>
        {part}
      </span>
    ) : (
      part
    ),
  )
}

export default function AnswerCard({ result }) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(result.answer)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      /* clipboard blocked (insecure origin / permissions) — fail quietly */
    }
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `kensho-trace-${result.trace_id}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const struck = result.claims.filter((c) => c.action === 'struck').length

  return (
    <div className="card">
      <div className="card-head">
        <h2 className="card-title">Answer</h2>
        <div style={{ display: 'flex', gap: 7, alignItems: 'center', flexWrap: 'wrap' }}>
          <span className="dim mono" style={{ fontSize: 10.5 }}>
            {result.trace_id}
          </span>
          <button className="btn btn--ghost" onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button className="btn btn--ghost" onClick={exportJson}>
            Export trace
          </button>
        </div>
      </div>

      <div className="card-body">
        {result.refused && (
          <div className="banner banner--refused">
            <span className="banner-icon" aria-hidden="true">
              ⚠
            </span>
            <div>
              <b>Refused.</b>{' '}
              {result.refusal_reason ||
                'Too little of the drafted answer survived verification to return it safely.'}
              <div className="dim" style={{ marginTop: 3, fontSize: 12 }}>
                Refusing is a result, not a failure — it is the branch that stops an
                ungrounded answer from reaching the user.
              </div>
            </div>
          </div>
        )}

        {!result.refused && struck > 0 && (
          <div className="banner banner--refused">
            <span className="banner-icon" aria-hidden="true">
              ⚠
            </span>
            <div>
              <b>
                {struck} {struck === 1 ? 'sentence was' : 'sentences were'} removed after
                verification.
              </b>{' '}
              The text below is what survived. The original draft is in the claim list.
            </div>
          </div>
        )}

        <div className={'answer' + (/[぀-ヿ一-龯]/.test(result.answer) ? ' jp' : '')}>
          {withCitations(result.answer)}
        </div>
      </div>
    </div>
  )
}
