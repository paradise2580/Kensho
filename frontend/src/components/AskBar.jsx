import { useRef } from 'react'

/**
 * The corpus is bilingual, so the examples are too — and the Japanese ones are
 * not translations of the English ones by accident: they exercise the
 * SudachiPy segmentation path that a whitespace tokenizer would silently get
 * wrong. Leaving `lang` on "auto" lets the retriever search both sides.
 */
// The shortcut is real on both platforms; only the glyph differs. Showing a ⌘
// to a Windows user is a small lie about what their keyboard does.
const IS_MAC =
  typeof navigator !== 'undefined' &&
  /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent || '')
const SUBMIT_HINT = IS_MAC ? '⌘ ↵' : 'Ctrl ↵'

const EXAMPLES = [
  { q: 'How does a Pod restart policy work?', lang: null },
  { q: 'What is the difference between a Deployment and a StatefulSet?', lang: null },
  { q: 'When does the kubelet evict a Pod under memory pressure?', lang: null },
  { q: 'Pod の再起動ポリシーはどう動作しますか?', lang: 'ja', jp: true },
  { q: 'Service とはなんですか?', lang: 'ja', jp: true },
]

export default function AskBar({
  question,
  onQuestion,
  lang,
  onLang,
  topK,
  onTopK,
  onSubmit,
  loading,
  canSubmit,
}) {
  const ref = useRef(null)

  function handleKeyDown(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && canSubmit) {
      e.preventDefault()
      onSubmit()
    }
  }

  function useExample(ex) {
    onQuestion(ex.q)
    onLang(ex.lang || '')
    ref.current?.focus()
  }

  return (
    <div className="ask">
      <div className="ask-field">
        <textarea
          ref={ref}
          value={question}
          onChange={(e) => onQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything the corpus covers — in English or Japanese. Every sentence of the answer is checked against the retrieved sources before you see it."
          spellCheck="false"
          aria-label="Question"
        />
      </div>

      <div className="ask-controls">
        <div className="field">
          <label htmlFor="lang">Language</label>
          <select id="lang" value={lang} onChange={(e) => onLang(e.target.value)}>
            <option value="">Auto (both)</option>
            <option value="en">English</option>
            <option value="ja">日本語</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="topk">Top-k</label>
          <input
            id="topk"
            type="range"
            min="3"
            max="10"
            step="1"
            value={topK}
            onChange={(e) => onTopK(Number(e.target.value))}
          />
          <span className="val">{topK}</span>
        </div>

        <button className="btn" onClick={onSubmit} disabled={!canSubmit}>
          {loading ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Running pipeline…
            </>
          ) : (
            <>
              Ask <span className="kbd">{SUBMIT_HINT}</span>
            </>
          )}
        </button>
      </div>

      <div className="examples">
        <span className="dim" style={{ fontSize: 11.5 }}>
          Try:
        </span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex.q}
            className={ex.jp ? 'ex jp' : 'ex'}
            onClick={() => useExample(ex)}
            disabled={loading}
          >
            {ex.q}
          </button>
        ))}
      </div>
    </div>
  )
}
