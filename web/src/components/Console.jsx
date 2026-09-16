import AskBar from './AskBar.jsx'
import StageTrack from './StageTrack.jsx'
import AnswerCard from './AnswerCard.jsx'
import ClaimList from './ClaimList.jsx'
import SourceList from './SourceList.jsx'
import LatencyBar from './LatencyBar.jsx'

function Pending() {
  return (
    <div className="card">
      <div className="card-body">
        <div style={{ display: 'grid', gap: 9 }}>
          <div className="skel" style={{ width: '92%' }} />
          <div className="skel" style={{ width: '78%' }} />
          <div className="skel" style={{ width: '85%' }} />
        </div>
        <p className="dim" style={{ fontSize: 12, marginTop: 16, marginBottom: 0, lineHeight: 1.6 }}>
          Retrieving, generating, then decomposing the draft and entailing every claim
          against its sources. The first verified request on a cold process is slow —
          the NLI model loads on demand, on CPU.
        </p>
      </div>
    </div>
  )
}

function Welcome({ healthState }) {
  return (
    <div className="card">
      <div className="empty">
        {healthState === 'down' ? (
          <>
            <b style={{ color: 'var(--critical)' }}>The service is not reachable.</b>
            <br />
            Start it with{' '}
            <code className="mono">uvicorn kensho.serve.app:create_app --factory</code> and
            reload.
            <br />
            <span style={{ fontSize: 12 }}>
              Nothing is shown here until it responds — a canned answer dressed up as a
              live one is the exact failure this project exists to refuse.
            </span>
          </>
        ) : (
          <>
            Ask a question to run the full pipeline.
            <br />
            <span style={{ fontSize: 12 }}>
              Every result on this page comes from the running service. There is no
              fixture mode.
            </span>
          </>
        )}
      </div>
    </div>
  )
}

export default function Console({
  question,
  setQuestion,
  lang,
  setLang,
  topK,
  setTopK,
  onSubmit,
  loading,
  result,
  error,
  history,
  onRestore,
  healthState,
}) {
  const phase = loading ? 'running' : result ? 'done' : 'idle'

  return (
    <div className="stack">
      <div className="card">
        <div className="card-body">
          <AskBar
            question={question}
            onQuestion={setQuestion}
            lang={lang}
            onLang={setLang}
            topK={topK}
            onTopK={setTopK}
            onSubmit={onSubmit}
            loading={loading}
            canSubmit={!loading && question.trim().length > 0}
          />
        </div>
      </div>

      <StageTrack phase={phase} />

      {error && (
        <div className="card">
          <div className="card-body">
            <div className="banner banner--error" style={{ marginBottom: 0 }}>
              <span className="banner-icon" aria-hidden="true">
                ✗
              </span>
              <div>
                <b>The request failed.</b>
                <div className="mono" style={{ fontSize: 12, marginTop: 4 }}>
                  {error}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid">
        <div className="stack">
          {loading ? (
            <Pending />
          ) : result ? (
            <>
              <AnswerCard result={result} />
              <ClaimList claims={result.claims} />
            </>
          ) : (
            !error && <Welcome healthState={healthState} />
          )}
        </div>

        <div className="stack">
          {result && !loading && (
            <>
              <LatencyBar latency={result.latency_ms} />
              <SourceList sources={result.sources} />
            </>
          )}

          {history.length > 0 && (
            <div className="card">
              <div className="card-head">
                <h2 className="card-title">This session</h2>
                <span className="dim" style={{ fontSize: 11.5 }}>
                  {history.length} {history.length === 1 ? 'request' : 'requests'}
                </span>
              </div>
              <div className="card-body" style={{ padding: 8 }}>
                <div className="hist">
                  {history.map((h) => (
                    <button
                      key={h.result.trace_id}
                      className="hist-item"
                      onClick={() => onRestore(h)}
                    >
                      <span className={'hist-q' + (h.jp ? ' jp' : '')}>{h.question}</span>
                      <span className="hist-meta">
                        {h.result.claims.length} claims ·{' '}
                        {h.result.claims.filter((c) => c.action === 'struck').length} struck ·{' '}
                        {Math.round(h.result.latency_ms.total)}ms
                        {h.result.refused ? ' · refused' : ''}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
