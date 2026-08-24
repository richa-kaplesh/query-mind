import React, { useState, useEffect, useRef, useCallback } from "react";
import "./Dashboard.css";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Step {
  type: string;
  elapsed_ms: number;
  data: Record<string, unknown>;
}

interface Trace {
  id: string;
  filename: string;
  query: string;
  status: "running" | "success" | "error";
  created_at: string;
  total_ms: number | null;
  tool_used: string | null;
  final_answer: string | null;
  steps: Step[];
}

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

// ─── Step metadata ────────────────────────────────────────────────────────────

const STEP_META: Record<string, { label: string; color: string; icon: string }> = {
  schema_context:    { label: "Dataset Schema",           color: "#f59e0b", icon: "📋" },
  llm_call_1:        { label: "LLM Call 1 — Routing",     color: "#3b82f6", icon: "🤖" },
  llm_response_1:    { label: "LLM Response 1",           color: "#6366f1", icon: "💬" },
  tool_input:        { label: "Tool Input",               color: "#a855f7", icon: "⚙️" },
  tool_output:       { label: "Tool Output",              color: "#14b8a6", icon: "📤" },
  llm_call_2:        { label: "LLM Call 2 — Synthesis",   color: "#3b82f6", icon: "🤖" },
  llm_call_2_stream: { label: "LLM Call 2 — Streaming",   color: "#3b82f6", icon: "🤖" },
  final_answer:      { label: "Final Answer",             color: "#22c55e", icon: "✅" },
};

function stepMeta(type: string) {
  return STEP_META[type] ?? { label: type, color: "#64748b", icon: "•" };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatMs(ms: number | null) {
  if (ms === null) return "—";
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(2)}s`;
}

function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function truncate(s: string, n = 72) {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

// ─── JSON Viewer ──────────────────────────────────────────────────────────────

function JsonBlock({ data }: { data: unknown }) {
  const text = JSON.stringify(data, null, 2);
  return (
    <pre className="db-json">
      <code>{text}</code>
    </pre>
  );
}

// ─── Step Card ────────────────────────────────────────────────────────────────

function StepCard({ step, index }: { step: Step; index: number }) {
  const [open, setOpen] = useState(index === 0);
  const meta = stepMeta(step.type);

  // For llm_response_1: highlight whether tool was called
  const isToolDecision = step.type === "llm_response_1";
  const decidedTool = isToolDecision && Array.isArray((step.data as Record<string, unknown>).tool_calls)
    ? ((step.data as Record<string, unknown[]>).tool_calls as Record<string, string>[])
    : [];
  const hasTool = decidedTool.length > 0;

  // For tool_input: show code nicely
  const isToolInput = step.type === "tool_input";
  const isToolOutput = step.type === "tool_output";
  const isFinalAnswer = step.type === "final_answer";
  const isSchema = step.type === "schema_context";

  return (
    <div className="db-step-card" style={{ borderLeftColor: meta.color }}>
      <button className="db-step-header" onClick={() => setOpen(o => !o)}>
        <span className="db-step-icon">{meta.icon}</span>
        <span className="db-step-label">{meta.label}</span>
        {isToolDecision && (
          <span className={`db-tool-decision ${hasTool ? "has-tool" : "no-tool"}`}>
            {hasTool ? `→ calls ${decidedTool[0]?.name}` : "→ direct answer"}
          </span>
        )}
        <span className="db-step-elapsed">+{formatMs(step.elapsed_ms)}</span>
        <span className={`db-chevron ${open ? "open" : ""}`}>›</span>
      </button>

      {open && (
        <div className="db-step-body">
          {/* Schema: show as plain text */}
          {isSchema && (
            <pre className="db-schema-text">{String((step.data as Record<string, unknown>).schema ?? "")}</pre>
          )}

          {/* Tool input: show code block */}
          {isToolInput && (
            <div className="db-tool-section">
              <div className="db-section-title" style={{ color: meta.color }}>Tool: {String((step.data as Record<string, unknown>).tool_name)}</div>
              <pre className="db-code-block"><code>{String((step.data as Record<string, unknown>).input ?? "")}</code></pre>
            </div>
          )}

          {/* Tool output */}
          {isToolOutput && (
            <div className="db-tool-section">
              <div className="db-section-title" style={{ color: meta.color }}>Result</div>
              <pre className="db-code-block db-code-output"><code>{String((step.data as Record<string, unknown>).result ?? "")}</code></pre>
            </div>
          )}

          {/* Final answer: plain text */}
          {isFinalAnswer && (
            <div className="db-final-answer">{String((step.data as Record<string, unknown>).answer ?? "")}</div>
          )}

          {/* Everything else: JSON */}
          {!isSchema && !isToolInput && !isToolOutput && !isFinalAnswer && (
            <JsonBlock data={step.data} />
          )}
        </div>
      )}
    </div>
  );
}

// ─── Trace Detail ─────────────────────────────────────────────────────────────

function TraceDetail({ trace }: { trace: Trace }) {
  return (
    <div className="db-detail">
      {/* Header */}
      <div className="db-detail-header">
        <div className="db-detail-query">{trace.query}</div>
        <div className="db-detail-meta">
          <span className="db-tag db-tag-file">📄 {trace.filename}</span>
          {trace.tool_used && (
            <span className="db-tag db-tag-tool">⚙️ {trace.tool_used}</span>
          )}
          <span className={`db-status-badge db-status-${trace.status}`}>
            {trace.status === "running" ? "⏳ running" : trace.status === "success" ? "✓ success" : "✗ error"}
          </span>
          <span className="db-tag db-tag-time">{formatMs(trace.total_ms)}</span>
        </div>
      </div>

      {/* Steps timeline */}
      <div className="db-timeline">
        {trace.steps.length === 0 ? (
          <div className="db-empty-steps">No steps recorded yet…</div>
        ) : (
          trace.steps.map((step, i) => (
            <StepCard key={`${step.type}-${i}`} step={step} index={i} />
          ))
        )}
      </div>
    </div>
  );
}

// ─── Trace List Item ──────────────────────────────────────────────────────────

function TraceRow({
  trace,
  selected,
  onClick,
}: {
  trace: Trace;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={`db-trace-row ${selected ? "selected" : ""} db-status-border-${trace.status}`}
      onClick={onClick}
    >
      <div className="db-trace-row-top">
        <span className="db-trace-query">{truncate(trace.query, 60)}</span>
        <span className={`db-dot db-dot-${trace.status}`} />
      </div>
      <div className="db-trace-row-bottom">
        <span className="db-trace-file">{trace.filename}</span>
        <span className="db-trace-time">{formatTime(trace.created_at)}</span>
      </div>
      {trace.tool_used && (
        <div className="db-trace-tool-tag">⚙️ {trace.tool_used}</div>
      )}
    </button>
  );
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

interface DashboardProps {
  onClose: () => void;
}

export default function Dashboard({ onClose }: DashboardProps) {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [search, setSearch] = useState("");
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = new EventSource(`${API_BASE}/dashboard/stream`);
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data) as { event: string; traces?: Trace[] };
        if (event.event === "init" || event.event === "update") {
          setTraces(event.traces ?? []);
          if (event.event === "init" && event.traces && event.traces.length > 0) {
            setSelectedId(prev => prev ?? event.traces![0].id);
          }
        }
      } catch { /* ignore parse errors */ }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, []);

  const filteredTraces = traces.filter(
    t =>
      t.query.toLowerCase().includes(search.toLowerCase()) ||
      t.filename.toLowerCase().includes(search.toLowerCase())
  );

  const selectedTrace = traces.find(t => t.id === selectedId) ?? null;

  const handleClear = useCallback(async () => {
    // We can't delete from backend easily, just clear UI
    setTraces([]);
    setSelectedId(null);
  }, []);

  return (
    <div className="db-root">
      {/* ── Top bar ── */}
      <header className="db-topbar">
        <div className="db-topbar-left">
          <span className="db-logo">QM</span>
          <span className="db-title">Debug Logs</span>
          <span className={`db-live-dot ${connected ? "live" : "dead"}`} />
          <span className="db-live-label">{connected ? "live" : "disconnected"}</span>
        </div>
        <div className="db-topbar-right">
          <span className="db-count">{traces.length} trace{traces.length !== 1 ? "s" : ""}</span>
          <button className="db-close-btn" onClick={onClose} title="Close dashboard">
            ✕ Close
          </button>
        </div>
      </header>

      <div className="db-body">
        {/* ── Sidebar ── */}
        <aside className="db-sidebar">
          <div className="db-search-wrap">
            <input
              className="db-search"
              placeholder="Search traces…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          <div className="db-trace-list">
            {filteredTraces.length === 0 ? (
              <div className="db-no-traces">
                {traces.length === 0
                  ? "No traces yet.\nSend a query to see it here."
                  : "No matches for your search."}
              </div>
            ) : (
              filteredTraces.map(trace => (
                <TraceRow
                  key={trace.id}
                  trace={trace}
                  selected={trace.id === selectedId}
                  onClick={() => setSelectedId(trace.id)}
                />
              ))
            )}
          </div>
        </aside>

        {/* ── Main ── */}
        <main className="db-main">
          {selectedTrace ? (
            <TraceDetail trace={selectedTrace} />
          ) : (
            <div className="db-placeholder">
              <div className="db-placeholder-icon">🔍</div>
              <div className="db-placeholder-text">Select a trace to inspect it</div>
              <div className="db-placeholder-sub">
                Upload a CSV and ask a question — every request will appear here with full LLM payloads, tool calls, and outputs.
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
