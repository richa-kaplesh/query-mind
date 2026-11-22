import { useCallback, useEffect, useRef, useState } from "react";
import { TracesPanel, type Trace } from "../../Dashboard";
import { apiFetch } from "../../session";
import { API_BASE, getJson } from "../api";
import { ErrorState, Skeleton } from "../StateViews";

const PDF_QUESTION = "What is this document about, in one sentence?";
const CSV_QUESTION = "How many rows and columns does this dataset have?";

type DocStatus = { filename: string; ready: boolean } | null;

async function readDocument(): Promise<DocStatus> {
  const res = await apiFetch("/documents/status");
  if (!res.ok) return null;
  const data = (await res.json()) as { documents: Record<string, string> };
  const [name, status] = Object.entries(data.documents ?? {})[0] ?? [];
  return name ? { filename: name, ready: status === "ready" } : null;
}

async function runSample(question: string): Promise<void> {
  const res = await apiFetch("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, conversation_history: [] }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail ?? `Sample question failed (${res.status})`);
  }
  const reader = res.body?.getReader();
  if (!reader) return;
  for (;;) {
    const { done } = await reader.read();
    if (done) break;
  }
}

function SampleButton({ onDone }: { onDone: () => Promise<void> }) {
  const [doc, setDoc] = useState<DocStatus | undefined>(undefined);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    readDocument().then(setDoc).catch(() => setDoc(null));
  }, []);

  async function go() {
    if (!doc) return;
    setRunning(true);
    setError(null);
    try {
      await runSample(doc.filename.toLowerCase().endsWith(".csv") ? CSV_QUESTION : PDF_QUESTION);
      await onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sample question failed");
    } finally {
      setRunning(false);
    }
  }

  const ready = !!doc?.ready;
  return (
    <div className="lg-sample">
      <button className="lg-btn lg-btn--primary" onClick={go} disabled={!ready || running}>
        {running ? "Running sample question…" : "Run a sample question"}
      </button>
      {doc === undefined && <p className="lg-muted">Checking for a loaded document…</p>}
      {doc !== undefined && !ready && (
        <p className="lg-muted">
          {doc ? `${doc.filename} is still processing.` : "No document is loaded in this browser session yet."}
          {" "}The sample question runs against the document you uploaded in chat.
        </p>
      )}
      {ready && <p className="lg-muted">Runs against {doc!.filename}.</p>}
      {error && <p className="lg-error" role="alert">{error}</p>}
    </div>
  );
}

export default function TracesTab() {
  const [traces, setTraces] = useState<Trace[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mobileDetailOpen, setMobileDetailOpen] = useState(false);
  const [connected, setConnected] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const received = useRef(false);

  const refetch = useCallback(async () => {
    const list = await getJson<Trace[]>("/dashboard/traces");
    setTraces(list);
    if (list[0]) setSelectedId(list[0].id);
  }, []);

  useEffect(() => {
    setError(null);
    const es = new EventSource(`${API_BASE}/dashboard/stream`);
    const timer = window.setTimeout(() => {
      if (!received.current) setError("Could not reach the trace stream.");
    }, 8000);
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = e => {
      try {
        const msg = JSON.parse(e.data) as { event: string; traces?: Trace[] };
        if (msg.event === "init" || msg.event === "update") {
          const list = msg.traces ?? [];
          received.current = true;
          setTraces(list);
          setError(null);
          if (msg.event === "init" && list[0]) setSelectedId(prev => prev ?? list[0].id);
        }
      } catch {
        setConnected(false);
      }
    };
    return () => { window.clearTimeout(timer); es.close(); };
  }, [attempt]);

  if (error && traces === null) {
    return <ErrorState message={error} onRetry={() => setAttempt(a => a + 1)} />;
  }
  if (traces === null) return <div className="lg-pad"><Skeleton rows={6} /></div>;

  if (traces.length === 0) {
    return (
      <div className="lg-state">
        <p className="lg-state-title">No traces yet</p>
        <p className="lg-state-text">
          A trace is the step-by-step record of one question: every LLM call, tool call and its timing.
        </p>
        <SampleButton onDone={refetch} />
      </div>
    );
  }

  return (
    <div className="lg-traces">
      <div className="lg-live" aria-live="polite">
        <span className={`lg-live-dot ${connected ? "is-live" : ""}`} aria-hidden />
        {connected ? "Live" : "Reconnecting…"} · {traces.length} trace{traces.length === 1 ? "" : "s"}
      </div>
      <TracesPanel
        traces={traces}
        selectedId={selectedId}
        setSelectedId={setSelectedId}
        mobileDetailOpen={mobileDetailOpen}
        setMobileDetailOpen={setMobileDetailOpen}
        emptyAction={<SampleButton onDone={refetch} />}
      />
    </div>
  );
}
