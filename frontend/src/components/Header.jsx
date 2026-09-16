const TABS = [
  { id: 'console', label: 'Live console' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'architecture', label: 'Architecture' },
]

/**
 * The health strip is the honest part of this component. It reports what the
 * running process actually loaded — which retriever, which embedder, which
 * LLM, which verifier — rather than what the README says it supports. If the
 * operator started the service without a Groq key, the llm chip says `echo`
 * and the demo is visibly not doing what it claims.
 */
function HealthChips({ health, state }) {
  if (state === 'loading') {
    return (
      <div className="health">
        <span className="chip">
          <span className="dot dot--wait" />
          <span className="chip-label">connecting</span>
        </span>
      </div>
    )
  }

  if (state === 'down') {
    return (
      <div className="health">
        <span className="chip">
          <span className="dot dot--down" />
          <span className="chip-label">service</span>
          <b>unreachable</b>
        </span>
      </div>
    )
  }

  const fields = [
    ['retriever', health.retriever],
    ['embedder', health.embedder],
    ['llm', health.llm],
    ['verifier', health.verifier],
    ['chunks', health.index_size >= 0 ? health.index_size.toLocaleString() : 'n/a'],
  ]

  return (
    <div className="health">
      <span className="chip">
        <span className="dot dot--live" />
        <span className="chip-label">live</span>
      </span>
      {fields.map(([label, value]) => (
        <span className="chip" key={label}>
          <span className="chip-label">{label}</span>
          <b>{value}</b>
        </span>
      ))}
    </div>
  )
}

export default function Header({ health, healthState, tab, onTab, theme, onToggleTheme }) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="brand-row">
          <div className="brand">
            <div className="brand-mark" aria-hidden="true">
              検
            </div>
            <div>
              <div className="brand-name">
                Kenshō <span className="dim jp">検証</span>
              </div>
              <div className="brand-sub">
                Bilingual JA/EN RAG with claim-level verification
              </div>
            </div>
          </div>

          <a
            className="icon-btn"
            href="https://github.com/paradise2580/Kensho"
            target="_blank"
            rel="noreferrer"
            title="Source on GitHub"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.4 7.4 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
            </svg>
            <span className="sr-only">Source on GitHub</span>
          </a>

          <button
            className="icon-btn"
            onClick={onToggleTheme}
            title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
          >
            {theme === 'dark' ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
              </svg>
            ) : (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z" />
              </svg>
            )}
            <span className="sr-only">Toggle colour theme</span>
          </button>
        </div>

        <HealthChips health={health} state={healthState} />

        <nav className="tabs" role="tablist" aria-label="Views">
          {TABS.map((t) => (
            <button
              key={t.id}
              className="tab"
              role="tab"
              aria-selected={tab === t.id}
              onClick={() => onTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  )
}
