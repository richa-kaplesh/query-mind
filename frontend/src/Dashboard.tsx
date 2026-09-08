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

// Open-ended metric shape — PDF runs carry faithfulness/answer_relevancy/…
// CSV runs carry tool_appropriateness/answer_correctness/… Both are valid.
type EvalScores = Record<string, number>;

interface EvalQuestion {
  question: string;
  answer: string;
  scores: EvalScores;
}

// Allow any JSON-shaped config from the backend
type EvalConfig = Record<string, unknown>;

interface EvalRun {
  timestamp: string;
  label: string;
  description?: string;
  type?: "pdf" | "csv";
  config?: EvalConfig;
  averages: EvalScores;
  per_question: EvalQuestion[];
}

interface TokenEntry {
  timestamp: number;
  model: string;
  purpose: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
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

function formatDateTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" }) +
    " · " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDropdownLabel(run: EvalRun): string {
  const d = new Date(run.timestamp);
  const datePart = d.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
  const timePart = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const lbl = run.label || "Unnamed Run";
  return `${lbl} — ${datePart} ${timePart}`;
}

function formatUnixTs(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleDateString([], { month: "short", day: "numeric" }) +
    " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatNum(n: number): string {
  return n.toLocaleString();
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

  const isToolDecision = step.type === "llm_response_1";
  const decidedTool = isToolDecision && Array.isArray((step.data as Record<string, unknown>).tool_calls)
    ? ((step.data as Record<string, unknown[]>).tool_calls as Record<string, string>[])
    : [];
  const hasTool = decidedTool.length > 0;

  const isToolInput  = step.type === "tool_input";
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
          {isSchema && (
            <pre className="db-schema-text">{String((step.data as Record<string, unknown>).schema ?? "")}</pre>
          )}
          {isToolInput && (
            <div className="db-tool-section">
              <div className="db-section-title" style={{ color: meta.color }}>Tool: {String((step.data as Record<string, unknown>).tool_name)}</div>
              <pre className="db-code-block"><code>{String((step.data as Record<string, unknown>).input ?? "")}</code></pre>
            </div>
          )}
          {isToolOutput && (
            <div className="db-tool-section">
              <div className="db-section-title" style={{ color: meta.color }}>Result</div>
              <pre className="db-code-block db-code-output"><code>{String((step.data as Record<string, unknown>).result ?? "")}</code></pre>
            </div>
          )}
          {isFinalAnswer && (
            <div className="db-final-answer">{String((step.data as Record<string, unknown>).answer ?? "")}</div>
          )}
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
  trace, selected, onClick,
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

// ─── Eval helpers ─────────────────────────────────────────────────────────────

// Known pretty-names — used as a lookup; falls back to prettyKey() for unknown metrics.
const METRIC_LABELS: Record<string, string> = {
  faithfulness:           "Faithfulness",
  answer_relevancy:       "Answer Relevancy",
  context_precision:      "Context Precision",
  context_recall:         "Context Recall",
  tool_appropriateness:   "Tool Appropriateness",
  answer_correctness:     "Answer Correctness",
};

/** Resolve a display label for any metric key. */
function metricLabel(k: string): string {
  return METRIC_LABELS[k] ?? prettyKey(k);
}

function scoreColor(v: number): string {
  if (v >= 0.8) return "#16a34a";
  if (v >= 0.5) return "#d97706";
  return "#dc2626";
}

function scoreBg(v: number): string {
  if (v >= 0.8) return "#f0fdf4";
  if (v >= 0.5) return "#fffbeb";
  return "#fef2f2";
}

function scoreBorder(v: number): string {
  if (v >= 0.8) return "#bbf7d0";
  if (v >= 0.5) return "#fde68a";
  return "#fecaca";
}

function ScoreBadge({ value, label }: { value: number; label: string }) {
  return (
    <div className="ev-score-badge" style={{
      background: scoreBg(value),
      borderColor: scoreBorder(value),
      color: scoreColor(value),
    }}>
      <span className="ev-score-label">{label}</span>
      <span className="ev-score-value">{value.toFixed(2)}</span>
    </div>
  );
}

function ScoreBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value * 100);
  return (
    <div className="ev-bar-row">
      <span className="ev-bar-label">{label}</span>
      <div className="ev-bar-track">
        <div
          className="ev-bar-fill"
          style={{ width: `${pct}%`, background: scoreColor(value) }}
        />
      </div>
      <span className="ev-bar-pct" style={{ color: scoreColor(value) }}>{pct}%</span>
    </div>
  );
}

// ─── Config diff helpers ───────────────────────────────────────────────────────

/**
 * Recursively flatten a config object into dotted-path → string-value pairs.
 * e.g. { chunker: { chunk_size: 500 } } → { "chunker.chunk_size": "500" }
 */
function flattenLeaves(obj: unknown, prefix = ""): Record<string, string> {
  if (obj === null || obj === undefined) return {};
  if (typeof obj !== "object" || Array.isArray(obj)) {
    return { [prefix]: String(obj) };
  }
  const result: Record<string, string> = {};
  for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      Object.assign(result, flattenLeaves(v, path));
    } else {
      result[path] = v === null || v === undefined ? "" : String(v);
    }
  }
  return result;
}

/**
 * Returns a set of dotted-path keys that differ between two configs.
 */
function diffConfigKeys(a: EvalConfig | undefined, b: EvalConfig | undefined): Set<string> {
  const fa = flattenLeaves(a);
  const fb = flattenLeaves(b);
  const allKeys = new Set([...Object.keys(fa), ...Object.keys(fb)]);
  const changed = new Set<string>();
  for (const k of allKeys) {
    if (fa[k] !== fb[k]) changed.add(k);
  }
  return changed;
}

// ─── Config section renderer ───────────────────────────────────────────────────

/**
 * Prettify a camelCase / snake_case key into a readable label.
 */
function prettyKey(k: string): string {
  return k
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, c => c.toUpperCase());
}

type ConfigLeaf = { key: string; dottedPath: string; value: string };
type ConfigSection = { title: string; rows: ConfigLeaf[]; nested: ConfigSection[] };

function buildConfigSections(
  config: EvalConfig,
  pathPrefix = ""
): ConfigSection[] {
  const sections: ConfigSection[] = [];
  for (const [k, v] of Object.entries(config)) {
    const dottedPath = pathPrefix ? `${pathPrefix}.${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      const nested = buildConfigSections(v as EvalConfig, dottedPath);
      // shallow leaf children of this object
      const rows: ConfigLeaf[] = [];
      for (const [sk, sv] of Object.entries(v as EvalConfig)) {
        if (sv === null || typeof sv !== "object" || Array.isArray(sv)) {
          rows.push({ key: prettyKey(sk), dottedPath: `${dottedPath}.${sk}`, value: String(sv ?? "") });
        }
      }
      const nestedSections = nested.filter(n => n.rows.length === 0 ? n.nested.length > 0 : true);
      sections.push({ title: prettyKey(k), rows, nested: nestedSections });
    } else {
      // top-level scalar — will be collected into a synthetic "General" section by caller
      sections.push({ title: "__scalar__", rows: [{ key: prettyKey(k), dottedPath, value: String(v ?? "") }], nested: [] });
    }
  }
  return sections;
}

function renderConfigSections(
  sections: ConfigSection[],
  diffKeys: Set<string>,
  depth = 0
): React.ReactNode {
  // Group leading scalars into a single block
  const scalars = sections.filter(s => s.title === "__scalar__");
  const groups  = sections.filter(s => s.title !== "__scalar__");

  return (
    <>
      {scalars.length > 0 && (
        <div className="ev-cfg-section" style={{ marginLeft: depth * 12 }}>
          {scalars.map(s => s.rows[0]).map(row => (
            <div
              key={row.dottedPath}
              className={`ev-cfg-row ${diffKeys.has(row.dottedPath) ? "ev-cfg-row--changed" : ""}`}
            >
              <span className="ev-cfg-key">{row.key}</span>
              <span className="ev-cfg-val">
                {row.value}
                {diffKeys.has(row.dottedPath) && <span className="ev-cfg-diff-badge">changed</span>}
              </span>
            </div>
          ))}
        </div>
      )}
      {groups.map(grp => (
        <div key={grp.title} className="ev-cfg-group" style={{ marginLeft: depth * 12 }}>
          <div className="ev-cfg-group-title">{grp.title}</div>
          {grp.rows.length > 0 && (
            <div className="ev-cfg-section">
              {grp.rows.map(row => (
                <div
                  key={row.dottedPath}
                  className={`ev-cfg-row ${diffKeys.has(row.dottedPath) ? "ev-cfg-row--changed" : ""}`}
                >
                  <span className="ev-cfg-key">{row.key}</span>
                  <span className="ev-cfg-val">
                    {row.value}
                    {diffKeys.has(row.dottedPath) && <span className="ev-cfg-diff-badge">changed</span>}
                  </span>
                </div>
              ))}
            </div>
          )}
          {grp.nested.length > 0 && renderConfigSections(grp.nested, diffKeys, depth + 1)}
        </div>
      ))}
    </>
  );
}

// ─── Investigation Notes ───────────────────────────────────────────────────────

function InvestigationNotes({ description }: { description: string }) {
  const [open, setOpen] = useState(false);
  const paragraphs = description.split(/\n\n+/).map(p => p.trim()).filter(Boolean);

  return (
    <div className="ev-notes-panel">
      <button className="ev-notes-toggle" onClick={() => setOpen(o => !o)}>
        <span className="ev-notes-toggle-left">
          <span className="ev-notes-icon">📋</span>
          <span>Investigation Notes</span>
        </span>
        <span className={`ev-run-chevron ${open ? "open" : ""}`}>›</span>
      </button>
      {open && (
        <div className="ev-notes-body">
          {paragraphs.map((p, i) => (
            <p key={i} className="ev-notes-paragraph">{p}</p>
          ))}
        </div>
      )}
    </div>
  );
}

function EvalQuestionRow({ item, index }: { item: EvalQuestion; index: number }) {
  const [open, setOpen] = useState(false);
  const s = item.scores;

  return (
    <div className="ev-q-row">
      <button className="ev-q-header" onClick={() => setOpen(o => !o)}>
        <span className="ev-q-num">Q{index + 1}</span>
        <span className="ev-q-text">{truncate(item.question, 90)}</span>
        <div className="ev-q-scores">
          {Object.entries(s).map(([k, v]) => (
            <span key={k} className="ev-q-score-dot" style={{ background: scoreColor(v) }} title={`${metricLabel(k)}: ${v.toFixed(2)}`} />
          ))}
        </div>
        <span className={`ev-run-chevron ${open ? "open" : ""}`}>›</span>
      </button>

      {open && (
        <div className="ev-q-body">
          <div className="ev-q-answer-label">Generated Answer</div>
          <div className="ev-q-answer">{item.answer}</div>
          <div className="ev-q-metric-grid">
            {Object.entries(s).map(([k, v]) => (
              <div key={k} className="ev-q-metric-cell" style={{ borderColor: scoreBorder(v), background: scoreBg(v) }}>
                <span className="ev-q-metric-label">{metricLabel(k)}</span>
                <span className="ev-q-metric-value" style={{ color: scoreColor(v) }}>{v.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Single Run View ──────────────────────────────────────────────────────────

function EvalRunView({ run, prevRun }: { run: EvalRun; prevRun?: EvalRun | null }) {
  const [cfgOpen, setCfgOpen] = useState(false);
  const avg = run.averages;

  // Compute diff keys between current and previous run's config
  const diffKeys = diffConfigKeys(run.config, prevRun?.config);
  const hasDiff = prevRun != null && diffKeys.size > 0;

  // Build nested config sections
  const cfgSections = run.config ? buildConfigSections(run.config) : [];

  return (
    <div className="ev-run-view">
      {/* ── Run identity strip ── */}
      <div className="ev-run-identity">
        <div className="ev-run-identity-left">
          <span className="ev-run-label">{run.label || "Unnamed Run"}</span>
          <span className="ev-run-ts">{formatDateTime(run.timestamp)}</span>
          {hasDiff && (
            <span className="ev-run-diff-hint">
              ↕ {diffKeys.size} config change{diffKeys.size !== 1 ? "s" : ""} vs <em>{prevRun!.label || "previous run"}</em>
            </span>
          )}
        </div>
        <div className="ev-run-identity-right">
          {Object.entries(avg).map(([k, v]) => (
            <ScoreBadge key={k} value={v} label={metricLabel(k).split(" ")[0]} />
          ))}
        </div>
      </div>

      {/* ── Investigation Notes (collapsible) ── */}
      {run.description && (
        <div className="ev-section ev-section--notes">
          <InvestigationNotes description={run.description} />
        </div>
      )}

      {/* ── Score bars ── */}
      <div className="ev-section">
        <div className="ev-section-title">Average Scores — {run.per_question.length} question{run.per_question.length !== 1 ? "s" : ""}</div>
        <div className="ev-bars">
          {Object.entries(avg).map(([k, v]) => (
            <ScoreBar key={k} value={v} label={metricLabel(k)} />
          ))}
        </div>
      </div>

      {/* ── Configuration (nested, with diff) ── */}
      {cfgSections.length > 0 && (
        <div className="ev-section">
          <button className="ev-cfg-toggle" onClick={() => setCfgOpen(o => !o)}>
            <span className="ev-cfg-toggle-left">
              <span>⚙ Configuration</span>
              {hasDiff && (
                <span className="ev-cfg-diff-count">{diffKeys.size} change{diffKeys.size !== 1 ? "s" : ""} vs prev</span>
              )}
            </span>
            <span className={`ev-run-chevron ${cfgOpen ? "open" : ""}`}>›</span>
          </button>
          {cfgOpen && (
            <div className="ev-cfg-nested-grid">
              {hasDiff && (
                <div className="ev-cfg-diff-banner">
                  <span className="ev-cfg-diff-banner-icon">↕</span>
                  Comparing against <strong>{prevRun!.label || "previous run"}</strong> — fields highlighted in amber changed
                </div>
              )}
              {renderConfigSections(cfgSections, diffKeys)}
            </div>
          )}
        </div>
      )}

      {/* ── Per-question breakdown ── */}
      <div className="ev-section">
        <div className="ev-section-title">Per-Question Breakdown</div>
        <div className="ev-questions">
          {run.per_question.map((q, qi) => (
            <EvalQuestionRow key={qi} item={q} index={qi} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Evaluations Panel (PDF / CSV tabs + dropdown) ────────────────────────────

type EvalTypeTab = "pdf" | "csv";

function EvalsPanel() {
  const [allRuns, setAllRuns] = useState<EvalRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [typeTab, setTypeTab] = useState<EvalTypeTab>("pdf");
  const [selectedPdfIdx, setSelectedPdfIdx] = useState(0);
  const [selectedCsvIdx, setSelectedCsvIdx] = useState(0);

  useEffect(() => {
    fetch(`${API_BASE}/dashboard/evals`)
      .then(r => {
        if (!r.ok) throw new Error(`Server error ${r.status}`);
        return r.json() as Promise<EvalRun[]>;
      })
      .then(data => { setAllRuns(data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) {
    return (
      <div className="ev-state">
        <div className="ev-state-icon">⏳</div>
        <div className="ev-state-text">Loading eval history…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="ev-state">
        <div className="ev-state-icon">⚠️</div>
        <div className="ev-state-text">Failed to load: {error}</div>
      </div>
    );
  }

  // Partition runs — records without a `type` field default to "pdf"
  const pdfRuns = allRuns.filter(r => !r.type || r.type === "pdf");
  const csvRuns = allRuns.filter(r => r.type === "csv");

  const activeRuns = typeTab === "pdf" ? pdfRuns : csvRuns;
  const selectedIdx = typeTab === "pdf" ? selectedPdfIdx : selectedCsvIdx;
  const setSelectedIdx = typeTab === "pdf" ? setSelectedPdfIdx : setSelectedCsvIdx;
  const selectedRun = activeRuns[selectedIdx] ?? null;
  // Previous run in the array (one position later = older, since runs are newest-first)
  const prevRun = selectedIdx + 1 < activeRuns.length ? activeRuns[selectedIdx + 1] : null;

  const emptyMsg = typeTab === "pdf"
    ? "No PDF evaluations run yet."
    : "No CSV evaluations run yet.";

  const totalRuns = pdfRuns.length + csvRuns.length;

  return (
    <div className="ev-panel">
      {/* ── Panel header ── */}
      <div className="ev-header">
        <div className="ev-header-left">
          <span className="ev-header-title">Evaluation History</span>
          <span className="ev-header-count">{totalRuns} run{totalRuns !== 1 ? "s" : ""}</span>
        </div>
        <div className="ev-header-legend">
          <span className="ev-legend-dot" style={{ background: "#16a34a" }} /> ≥0.8
          <span className="ev-legend-dot" style={{ background: "#d97706", marginLeft: 10 }} /> 0.5–0.8
          <span className="ev-legend-dot" style={{ background: "#dc2626", marginLeft: 10 }} /> &lt;0.5
        </div>
      </div>

      {/* ── Inner type tabs ── */}
      <div className="ev-type-tabs-bar">
        <div className="ev-type-tabs">
          <button
            className={`ev-type-tab ${typeTab === "pdf" ? "active" : ""}`}
            onClick={() => setTypeTab("pdf")}
          >
            📄 PDF Evaluations
            {pdfRuns.length > 0 && <span className="ev-type-tab-count">{pdfRuns.length}</span>}
          </button>
          <button
            className={`ev-type-tab ${typeTab === "csv" ? "active" : ""}`}
            onClick={() => setTypeTab("csv")}
          >
            📊 CSV Evaluations
            {csvRuns.length > 0 && <span className="ev-type-tab-count">{csvRuns.length}</span>}
          </button>
        </div>

        {/* ── Run dropdown ── */}
        {activeRuns.length > 0 && (
          <div className="ev-run-select-wrap">
            <select
              className="ev-run-select"
              value={selectedIdx}
              onChange={e => setSelectedIdx(Number(e.target.value))}
            >
              {activeRuns.map((run, i) => (
                <option key={`${run.timestamp}-${i}`} value={i}>
                  {formatDropdownLabel(run)}
                </option>
              ))}
            </select>
            <span className="ev-run-select-chevron">▾</span>
          </div>
        )}
      </div>

      {/* ── Content ── */}
      <div className="ev-tab-content">
        {activeRuns.length === 0 ? (
          <div className="ev-state">
            <div className="ev-state-icon">📊</div>
            <div className="ev-state-text">{emptyMsg}</div>
            <div className="ev-state-sub">
              Run <code className="ev-code">python eval/run_eval.py</code> from the backend directory to generate an eval report.
            </div>
          </div>
        ) : selectedRun ? (
          <EvalRunView key={selectedRun.timestamp} run={selectedRun} prevRun={prevRun} />
        ) : null}
      </div>
    </div>
  );
}

// ─── Token Usage Panel ────────────────────────────────────────────────────────

const PURPOSE_COLORS: Record<string, string> = {
  rag:       "#0e7490",
  csv_call1: "#7c3aed",
  csv_call2: "#0891b2",
};

function purposeColor(p: string): string {
  return PURPOSE_COLORS[p] ?? "#64748b";
}

function TokenPanel() {
  const [entries, setEntries] = useState<TokenEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/dashboard/tokens`)
      .then(r => {
        if (!r.ok) throw new Error(`Server error ${r.status}`);
        return r.json() as Promise<TokenEntry[]>;
      })
      .then(data => { setEntries(data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) {
    return (
      <div className="ev-state">
        <div className="ev-state-icon">⏳</div>
        <div className="ev-state-text">Loading token usage…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="ev-state">
        <div className="ev-state-icon">⚠️</div>
        <div className="ev-state-text">Failed to load: {error}</div>
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <div className="ev-state">
        <div className="ev-state-icon">🪙</div>
        <div className="ev-state-text">No token usage recorded yet.</div>
        <div className="ev-state-sub">Token usage is logged automatically on each query. Run a query to see data here.</div>
      </div>
    );
  }

  // Aggregated totals
  const totalPrompt     = entries.reduce((s, e) => s + e.prompt_tokens, 0);
  const totalCompletion = entries.reduce((s, e) => s + e.completion_tokens, 0);
  const totalAll        = entries.reduce((s, e) => s + e.total_tokens, 0);

  // Purpose breakdown
  const purposeMap: Record<string, number> = {};
  for (const e of entries) {
    purposeMap[e.purpose] = (purposeMap[e.purpose] ?? 0) + e.total_tokens;
  }
  const purposeEntries = Object.entries(purposeMap).sort((a, b) => b[1] - a[1]);
  const maxPurposeTokens = purposeEntries[0]?.[1] ?? 1;

  // Table rows: newest first
  const sorted = [...entries].sort((a, b) => b.timestamp - a.timestamp);

  return (
    <div className="tok-panel">
      {/* ── Header ── */}
      <div className="ev-header">
        <div className="ev-header-left">
          <span className="ev-header-title">Token Usage</span>
          <span className="ev-header-count">{entries.length} call{entries.length !== 1 ? "s" : ""}</span>
        </div>
      </div>

      <div className="tok-body">
        {/* ── Summary cards ── */}
        <div className="tok-summary-cards">
          <div className="tok-card">
            <span className="tok-card-value">{formatNum(totalPrompt)}</span>
            <span className="tok-card-label">Prompt Tokens</span>
          </div>
          <div className="tok-card tok-card--completion">
            <span className="tok-card-value">{formatNum(totalCompletion)}</span>
            <span className="tok-card-label">Completion Tokens</span>
          </div>
          <div className="tok-card tok-card--total">
            <span className="tok-card-value">{formatNum(totalAll)}</span>
            <span className="tok-card-label">Total Tokens</span>
          </div>
        </div>

        {/* ── Purpose breakdown ── */}
        <div className="tok-purpose-section">
          <div className="tok-section-title">Breakdown by Purpose</div>
          <div className="tok-purpose-bars">
            {purposeEntries.map(([purpose, tokens]) => (
              <div key={purpose} className="tok-purpose-bar-row">
                <span className="tok-purpose-bar-label" style={{ color: purposeColor(purpose) }}>
                  {purpose}
                </span>
                <div className="tok-purpose-bar-track">
                  <div
                    className="tok-purpose-bar-fill"
                    style={{
                      width: `${Math.round((tokens / maxPurposeTokens) * 100)}%`,
                      background: purposeColor(purpose),
                    }}
                  />
                </div>
                <span className="tok-purpose-bar-value">{formatNum(tokens)} tokens</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Chronological table ── */}
        <div className="tok-table-section">
          <div className="tok-section-title">All Calls — most recent first</div>
          <div className="tok-table-wrap">
            <table className="tok-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Model</th>
                  <th>Purpose</th>
                  <th className="tok-num">Prompt</th>
                  <th className="tok-num">Completion</th>
                  <th className="tok-num">Total</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((e, i) => (
                  <tr key={i}>
                    <td className="tok-mono tok-ts">{formatUnixTs(e.timestamp)}</td>
                    <td className="tok-mono tok-model">{e.model}</td>
                    <td>
                      <span className="tok-purpose-pill" style={{ background: `${purposeColor(e.purpose)}18`, color: purposeColor(e.purpose), borderColor: `${purposeColor(e.purpose)}40` }}>
                        {e.purpose}
                      </span>
                    </td>
                    <td className="tok-mono tok-num">{formatNum(e.prompt_tokens)}</td>
                    <td className="tok-mono tok-num">{formatNum(e.completion_tokens)}</td>
                    <td className="tok-mono tok-num tok-total">{formatNum(e.total_tokens)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

interface DashboardProps {
  onClose: () => void;
}

type Tab = "traces" | "evals" | "tokens";

export default function Dashboard({ onClose }: DashboardProps) {
  const [activeTab, setActiveTab] = useState<Tab>("traces");
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
      } catch { /* ignore */ }
    };
    return () => { es.close(); esRef.current = null; };
  }, []);

  const filteredTraces = traces.filter(
    t =>
      t.query.toLowerCase().includes(search.toLowerCase()) ||
      t.filename.toLowerCase().includes(search.toLowerCase())
  );

  const selectedTrace = traces.find(t => t.id === selectedId) ?? null;

  const handleClear = useCallback(() => {
    setTraces([]);
    setSelectedId(null);
  }, []);

  // suppress unused warning
  void handleClear;

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

        {/* Tab switcher */}
        <div className="db-tabs">
          <button
            className={`db-tab ${activeTab === "traces" ? "active" : ""}`}
            onClick={() => setActiveTab("traces")}
          >
            Traces
            <span className="db-tab-count">{traces.length}</span>
          </button>
          <button
            className={`db-tab ${activeTab === "evals" ? "active" : ""}`}
            onClick={() => setActiveTab("evals")}
          >
            Evaluations
          </button>
          <button
            className={`db-tab ${activeTab === "tokens" ? "active" : ""}`}
            onClick={() => setActiveTab("tokens")}
          >
            Token Usage
          </button>
        </div>

        <div className="db-topbar-right">
          {activeTab === "traces" && (
            <span className="db-count">{traces.length} trace{traces.length !== 1 ? "s" : ""}</span>
          )}
          <button className="db-close-btn" onClick={onClose} title="Close dashboard">
            ✕ Close
          </button>
        </div>
      </header>

      {/* ── Traces tab ── */}
      {activeTab === "traces" && (
        <div className="db-body">
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
      )}

      {/* ── Evaluations tab ── */}
      {activeTab === "evals" && (
        <div className="db-body db-body--evals">
          <main className="db-main db-main--evals">
            <EvalsPanel />
          </main>
        </div>
      )}

      {/* ── Token Usage tab ── */}
      {activeTab === "tokens" && (
        <div className="db-body db-body--evals">
          <main className="db-main db-main--evals">
            <TokenPanel />
          </main>
        </div>
      )}
    </div>
  );
}
