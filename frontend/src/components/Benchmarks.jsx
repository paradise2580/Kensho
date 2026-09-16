/**
 * Every number on this tab is either measured or explicitly labelled as not
 * measured. That distinction is the point: a portfolio project that reports
 * ablation results it never ran is worth less than one that runs fewer and
 * says which is which.
 *
 * The gold sets were hand-labelled by reading the actual EN/JA source pages,
 * not recalled from general Kubernetes knowledge — which is why v2 agreeing
 * with v1 across five added topics is a real check rather than a restatement.
 */

const GOLD = [
  { metric: 'recall@1', v1: '0.65', v2: '0.690' },
  { metric: 'recall@5', v1: '0.90', v2: '0.914', best: true },
  { metric: 'MRR', v1: '—', v2: '0.782' },
]

const RETRIEVAL = [
  { config: 'BM25 + SudachiPy (baseline)', measured: true, note: 'No network, no model weights' },
  { config: 'Dense — multilingual-e5-large', measured: false, note: 'Needs HF Hub' },
  { config: 'Dense — BAAI/bge-m3', measured: false, note: 'Embedder ablation' },
  { config: 'Hybrid — reciprocal rank fusion', measured: false, note: 'Dense ⊕ sparse' },
  { config: 'Cross-encoder rerank', measured: false, note: 'Top-50 → top-k' },
  { config: 'Fixed-window chunking', measured: false, note: 'vs. structural' },
  { config: 'Cross-lingual restriction', measured: false, note: 'JA query → EN corpus' },
]

const VERIFIERS = [
  { m: 'Lexical overlap', note: 'Baseline. Structurally cannot detect contradiction — a negated sentence shares nearly every token with the sentence it negates.', measured: true },
  { m: 'Embedding similarity', note: 'Same structural blind spot, higher recall on paraphrase.', measured: false },
  { m: 'NLI entailment (mDeBERTa)', note: 'The arm that can return "contradicted" at all.', measured: false },
  { m: 'LLM-as-judge', note: 'Strongest, slowest, and the only one that costs money per claim.', measured: false },
]

function Yes() {
  return (
    <span className="badge" style={{ '--badge-color': 'var(--good)' }}>
      <span className="badge-glyph">✓</span>measured
    </span>
  )
}

function No() {
  return (
    <span className="badge badge--plain">
      <span className="badge-glyph" style={{ color: 'var(--text-3)' }}>
        ○
      </span>
      wired, not run
    </span>
  )
}

export default function Benchmarks() {
  return (
    <div className="stack">
      <div className="stat-row">
        <div className="stat">
          <div className="stat-label">recall@5</div>
          <div className="stat-value">0.914</div>
          <div className="stat-note">gold_v2, 62 items</div>
        </div>
        <div className="stat">
          <div className="stat-label">CI floor</div>
          <div className="stat-value">0.896</div>
          <div className="stat-note">a regression fails the build</div>
        </div>
        <div className="stat">
          <div className="stat-label">Corpus</div>
          <div className="stat-value">8,210</div>
          <div className="stat-note">chunks · 406 EN / 325 JA</div>
        </div>
        <div className="stat">
          <div className="stat-label">Tests</div>
          <div className="stat-value">230</div>
          <div className="stat-note">gating every push</div>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h2 className="card-title">Retrieval quality</h2>
          <span className="dim" style={{ fontSize: 11.5 }}>
            BM25 sparse · SudachiPy morphological segmentation
          </span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th style={{ textAlign: 'right' }}>gold_v1 · 52 items</th>
                <th style={{ textAlign: 'right' }}>gold_v2 · 62 items</th>
              </tr>
            </thead>
            <tbody>
              {GOLD.map((r) => (
                <tr key={r.metric} className={r.best ? 'best' : undefined}>
                  <td>{r.metric}</td>
                  <td className="num">{r.v1}</td>
                  <td className="num">{r.v2}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card-body" style={{ borderTop: '1px solid var(--border)' }}>
          <p className="muted" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.65 }}>
            Both gold sets were written by reading the source pages, so a query can fail
            for a reason that is in the documents rather than in the model. v2 adds five
            topics v1 never covered; the two agreeing to within four points is the
            evidence that v1 was not a small-sample artifact.
          </p>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <div className="card-head">
            <h2 className="card-title">Retrieval configurations</h2>
            <span className="dim" style={{ fontSize: 11.5 }}>
              one <code className="mono">Retriever</code> protocol, seven implementations
            </span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Configuration</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {RETRIEVAL.map((r) => (
                  <tr key={r.config}>
                    <td style={{ whiteSpace: 'normal' }}>
                      {r.config}
                      <div className="dim" style={{ fontSize: 11 }}>
                        {r.note}
                      </div>
                    </td>
                    <td>{r.measured ? <Yes /> : <No />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <h2 className="card-title">Verification methods</h2>
          </div>
          <div className="card-body">
            <div className="notes">
              {VERIFIERS.map((v) => (
                <div className="note" key={v.m}>
                  <h4 style={{ display: 'flex', gap: 9, alignItems: 'center', flexWrap: 'wrap' }}>
                    {v.m} {v.measured ? <Yes /> : <No />}
                  </h4>
                  <p>{v.note}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-body">
          <p className="muted" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.65 }}>
            <b style={{ color: 'var(--text-1)' }}>Why some rows say "wired, not run".</b>{' '}
            The build environment this project was developed in had no Hugging Face Hub
            or Groq access, so those arms are implemented, unit-tested against fakes, and
            executable — but their numbers were never produced there, and reporting them
            as if they had been is the exact failure mode this project is about.{' '}
            <code className="mono">RUNBOOK.md</code> is the command sequence that fills
            them in on a machine with that access.
          </p>
        </div>
      </div>
    </div>
  )
}
