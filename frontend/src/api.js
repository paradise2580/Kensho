/**
 * The only place in the UI that knows the service exists.
 *
 * Everything goes over real HTTP to the same FastAPI app a non-browser client
 * would call — there is no mock layer and no fixture mode, so a broken contract
 * surfaces in the console exactly as it would for any other consumer.
 *
 * The timeout is generous on purpose: the NLI verifier lazily loads mDeBERTa,
 * and on a CPU-only machine the *first* verified request legitimately takes
 * tens of seconds. Timing that out would look like a failure when it is
 * actually a cold start, so we wait and say so in the UI instead.
 */

const COLD_START_TIMEOUT_MS = 180_000

class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request(path, init = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), COLD_START_TIMEOUT_MS)

  let res
  try {
    res = await fetch(path, {
      ...init,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(init.headers || {}) },
    })
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new ApiError(
        `The service did not respond within ${COLD_START_TIMEOUT_MS / 1000}s.`,
        0,
      )
    }
    throw new ApiError(
      'Could not reach the service. Is uvicorn running on this host?',
      0,
    )
  } finally {
    clearTimeout(timer)
  }

  if (!res.ok) {
    // FastAPI validation errors arrive as {detail: [...]}; surface the text
    // rather than "[object Object]", which tells the operator nothing.
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) detail = JSON.stringify(body.detail)
    } catch {
      /* body was not JSON; the status line is all we have */
    }
    throw new ApiError(detail, res.status)
  }

  return res.json()
}

export function getHealth() {
  return request('/healthz')
}

export function postAsk({ question, lang, topK }) {
  return request('/ask', {
    method: 'POST',
    body: JSON.stringify({
      question,
      lang: lang || null,
      top_k: topK ?? null,
    }),
  })
}

export { ApiError }
