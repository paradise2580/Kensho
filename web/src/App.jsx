import { useCallback, useEffect, useState } from 'react'
import Background from './components/Background.jsx'
import Header from './components/Header.jsx'
import Console from './components/Console.jsx'
import Benchmarks from './components/Benchmarks.jsx'
import Architecture from './components/Architecture.jsx'
import { getHealth, postAsk } from './api.js'

const THEME_KEY = 'kensho.theme'

function readStoredTheme() {
  // Storage is a per-viewer convenience here, and it throws outright in a
  // private window with site data blocked — so it is never load-bearing.
  try {
    const v = localStorage.getItem(THEME_KEY)
    return v === 'light' || v === 'dark' ? v : null
  } catch {
    return null
  }
}

export default function App() {
  const [theme, setTheme] = useState(() => readStoredTheme() || 'dark')
  const [tab, setTab] = useState('console')

  const [health, setHealth] = useState(null)
  const [healthState, setHealthState] = useState('loading')

  const [question, setQuestion] = useState('')
  const [lang, setLang] = useState('')
  const [topK, setTopK] = useState(5)

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem(THEME_KEY, theme)
    } catch {
      /* ignore */
    }
  }, [theme])

  // Poll health once at boot, then every 20s. The chips reflect what the
  // process actually loaded; if the operator forgot the API key, the llm chip
  // says so rather than the page pretending otherwise.
  useEffect(() => {
    let alive = true

    async function check() {
      try {
        const h = await getHealth()
        if (!alive) return
        setHealth(h)
        setHealthState('ok')
      } catch {
        if (!alive) return
        setHealthState('down')
      }
    }

    check()
    const id = setInterval(check, 20_000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  const submit = useCallback(async () => {
    const q = question.trim()
    if (!q || loading) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await postAsk({ question: q, lang, topK })
      setResult(res)
      setHistory((prev) => [
        { question: q, jp: /[぀-ヿ一-龯]/.test(q), lang, topK, result: res },
        ...prev,
      ])
    } catch (err) {
      // Deliberately no fallback to a stored sample. Showing a canned trace
      // when the pipeline failed would be dressing a broken system up as a
      // working one, which is precisely what this project exists to refuse.
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }, [question, lang, topK, loading])

  const restore = useCallback((entry) => {
    setQuestion(entry.question)
    setLang(entry.lang)
    setTopK(entry.topK)
    setResult(entry.result)
    setError(null)
  }, [])

  return (
    <>
      <Background />
      <Header
        health={health}
        healthState={healthState}
        tab={tab}
        onTab={setTab}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
      />
      <main className="shell">
        {tab === 'console' && (
          <Console
            question={question}
            setQuestion={setQuestion}
            lang={lang}
            setLang={setLang}
            topK={topK}
            setTopK={setTopK}
            onSubmit={submit}
            loading={loading}
            result={result}
            error={error}
            history={history}
            onRestore={restore}
            healthState={healthState}
          />
        )}
        {tab === 'benchmarks' && <Benchmarks />}
        {tab === 'architecture' && <Architecture />}
      </main>
    </>
  )
}
