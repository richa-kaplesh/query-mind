// ─── Session management ───────────────────────────────────────────────────────
//
// getSessionId()  – returns the current session id, creating one if needed.
// apiFetch()      – like fetch(), but injects X-Session-Id and handles 401.

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const SESSION_KEY = "qm_session_id";

// Single in-flight promise so that concurrent callers never fire two POSTs.
let _inflightSession: Promise<string> | null = null;

/** Return the stored session id, or POST /session to create one. */
export async function getSessionId(): Promise<string> {
  const stored = sessionStorage.getItem(SESSION_KEY);
  if (stored) return stored;

  // If a creation request is already in flight, reuse it.
  if (_inflightSession) return _inflightSession;

  _inflightSession = (async () => {
    const res = await fetch(`${API_BASE}/session`, { method: "POST" });
    if (res.status === 503) {
      throw new Error("Server is full, please try again in a minute.");
    }
    if (!res.ok) {
      throw new Error(`Failed to create session: ${res.status}`);
    }
    const data = (await res.json()) as { session_id: string };
    sessionStorage.setItem(SESSION_KEY, data.session_id);
    return data.session_id;
  })().finally(() => {
    // Clear the in-flight reference whether the call succeeded or failed.
    _inflightSession = null;
  });

  return _inflightSession;
}

/**
 * Drop-in replacement for fetch() that:
 *  1. Adds the X-Session-Id header automatically.
 *  2. On a 401, discards the stale id, obtains a fresh session, and retries
 *     the request exactly once.
 */
export async function apiFetch(
  path: string,
  init?: RequestInit
): Promise<Response> {
  const sessionId = await getSessionId();

  const buildInit = (id: string): RequestInit => ({
    ...init,
    headers: {
      ...(init?.headers ?? {}),
      "X-Session-Id": id,
    },
  });

  const res = await fetch(API_BASE + path, buildInit(sessionId));

  if (res.status === 401) {
    // Session was unknown / expired — discard it and get a fresh one.
    sessionStorage.removeItem(SESSION_KEY);
    const newId = await getSessionId();
    return fetch(API_BASE + path, buildInit(newId));
  }

  return res;
}
