import React, { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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

type EvalScores = Record<string, number>;

interface EvalQuestion {
  question: string;
  answer: string;
  scores: EvalScores;
}

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

// ─── Step metadata (colours reference CSS variables) ──────────────────────────

const STEP_META: Record<string, { label: string; cssVar: string; icon: React.ReactNode }> = {
  schema_context:    { label: "Dataset Schema",         cssVar: "var(--step-schema)",   icon: <SchemaIcon /> },
  llm_call_1:        { label: "LLM Call 1 — Routing",   cssVar: "var(--step-llm)",      icon: <LlmIcon /> },
  llm_response_1:    { label: "LLM Response 1",         cssVar: "var(--step-llm-resp)", icon: <ResponseIcon /> },
  tool_input:        { label: "Tool Input",             cssVar: "var(--step-tool-in)",  icon: <ToolIcon /> },
  tool_output:       { label: "Tool Output",            cssVar: "var(--step-tool-out)", icon: <OutputIcon /> },
  llm_call_2:        { label: "LLM Call 2 — Synthesis", cssVar: "var(--step-llm)",      icon: <LlmIcon /> },
  llm_call_2_stream: { label: "LLM Call 2 — Streaming", cssVar: "var(--step-llm)",      icon: <LlmIcon /> },
  final_answer:      { label: "Final Answer",           cssVar: "var(--step-answer)",   icon: <AnswerIcon /> },
};

function stepMeta(type: string) {
  return STEP_META[type] ?? { label: type, cssVar: "var(--step-default)", icon: <DefaultStepIcon /> };
}

// ─── Score helpers (CSS variable strings — theme-aware) ───────────────────────

const METRIC_LABELS: Record<string, string> = {
  faithfulness:          "Faithfulness",
  answer_relevancy:      "Answer Relevancy",
  context_precision:     "Context Precision",
  context_recall:        "Context Recall",
  tool_appropriateness:  "Tool Appropriateness",
  answer_correctness:    "Answer Correctness",
};

function metricLabel(k: string): string {
  return METRIC_LABELS[k] ?? prettyKey(k);
}

function scoreState(v: number): "good" | "ok" | "bad" {
  if (v >= 0.8) return "good";
  if (v >= 0.5) return "ok";
  return "bad";
}

function scoreColor(v: number)   { return `var(--score-${scoreState(v)})`; }
function scoreBg(v: number)      { return `var(--score-${scoreState(v)}-bg)`; }
function scoreBorder(v: number)  { return `var(--score-${scoreState(v)}-brd)`; }

// ─── Purpose colours ──────────────────────────────────────────────────────────

const PURPOSE_CSS: Record<string, string> = {
  rag:       "var(--purpose-rag)",
  csv_call1: "var(--purpose-csv1)",
  csv_call2: "var(--purpose-csv2)",
};

function purposeColor(p: string): string {
  return PURPOSE_CSS[p] ?? "var(--purpose-default)";
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

function formatRelative(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60)  return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function formatDropdownLabel(run: EvalRun): string {
  const d = new Date(run.timestamp);
  const datePart = d.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
  const timePart = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const lbl = run.label || "Unnamed Run";
  const avg = Object.values(run.averages);
  const score = avg.length > 0 ? (avg.reduce((s, v) => s + v, 0) / avg.length * 100).toFixed(0) + "%" : "";
  return `${lbl} — ${datePart} ${timePart}${score ? " · " + score : ""}`;
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

function prettyKey(k: string): string {
  return k.replace(/_/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").replace(/\b\w/g, c => c.toUpperCase());
}

// ─── SVG Icons ────────────────────────────────────────────────────────────────

function SchemaIcon()      { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>; }
function LlmIcon()         { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>; }
function ResponseIcon()    { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>; }
function ToolIcon()        { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>; }
function OutputIcon()      { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>; }
function AnswerIcon()      { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>; }
function DefaultStepIcon() { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="4"/></svg>; }
function FileIcon()        { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>; }
function ToolTagIcon()     { return <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>; }
function ChevronIcon()     { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="9 18 15 12 9 6"/></svg>; }
function ChevronLeftIcon() { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="15 18 9 12 15 6"/></svg>; }
function CopyIcon()        { return <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>; }
function CheckIcon()       { return <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>; }
function ExpandIcon()      { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>; }
function CollapseIcon()    { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="10" y1="14" x2="3" y2="21"/><line x1="21" y1="3" x2="14" y2="10"/></svg>; }
function SearchIcon()      { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>; }
function PdfIcon()         { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>; }
function CsvIcon()         { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>; }
function NotesIcon()       { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>; }
function ConfigIcon()      { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 010 14.14M4.93 4.93a10 10 0 000 14.14"/></svg>; }
function PlaceholderIcon() { return <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>; }
function SpinnerIcon()     { return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" style={{ animation: "db-spin 0.8s linear infinite", transformOrigin: "center" }}/></svg>; }

// ─── Code block with copy button ──────────────────────────────────────────────

function CodeBlock({ text, className = "" }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="db-code-wrap">
      <pre className={`db-code-block ${className}`}><code>{text}</code></pre>
      <button
        className={`db-code-copy-btn ${copied ? "db-code-copy-btn--copied" : ""}`}
        onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1800); }}
        aria-label="Copy code"
      >
        {copied ? <><CheckIcon /> Copied</> : <><CopyIcon /> Copy</>}
      </button>
    </div>
  );
}

// ─── JSON Viewer ──────────────────────────────────────────────────────────────

function JsonBlock({ data }: { data: unknown }) {
  return <CodeBlock text={JSON.stringify(data, null, 2)} />;
}

// ─── Waterfall ────────────────────────────────────────────────────────────────

function Waterfall({ steps, totalMs, hoveredType, onHover }: {
  steps: Step[];
  totalMs: number;
  hoveredType: string | null;
  onHover: (t: string | null) => void;
}) {
  if (totalMs <= 0 || steps.length === 0) return null;

  return (
    <div>
      <div className="db-waterfall">
        {steps.map((step, i) => {
          const meta = stepMeta(step.type);
          const pct = Math.max(1, (step.elapsed_ms / totalMs) * 100);
          const isHov = hoveredType === step.type;
          const isDim = hoveredType !== null && !isHov;
          return (
            <div
              key={i}
              className={`db-waterfall-seg ${isDim ? "db-waterfall-seg--dim" : ""} ${isHov ? "db-waterfall-seg--hi" : ""}`}
              style={{ flex: `0 0 ${pct}%`, background: meta.cssVar }}
              title={`${meta.label}: ${formatMs(step.elapsed_ms)}`}
              onMouseEnter={() => onHover(step.type)}
              onMouseLeave={() => onHover(null)}
            />
          );
        })}
      </div>
      <div className="db-waterfall-legend">
        {steps.map((step, i) => {
          const meta = stepMeta(step.type);
          return (
            <span key={i} className="db-waterfall-legend-item">
              <span className="db-waterfall-legend-dot" style={{ background: meta.cssVar }} />
              {meta.label} {formatMs(step.elapsed_ms)}
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ─── Step Card ────────────────────────────────────────────────────────────────

function StepCard({ step, index, forceOpen }: { step: Step; index: number; forceOpen?: boolean }) {
  const [open, setOpen] = useState(index === 0);
  const meta = stepMeta(step.type);
  const effectiveOpen = forceOpen !== undefined ? forceOpen : open;

  const isToolDecision = step.type === "llm_response_1";
  const decidedTool = isToolDecision && Array.isArray((step.data as Record<string, unknown>).tool_calls)
    ? ((step.data as Record<string, unknown[]>).tool_calls as Record<string, string>[])
    : [];
  const hasTool = decidedTool.length > 0;

  const isToolInput   = step.type === "tool_input";
  const isToolOutput  = step.type === "tool_output";
  const isFinalAnswer = step.type === "final_answer";
  const isSchema      = step.type === "schema_context";

  return (
    <div className="db-step-card" style={{ borderLeftColor: meta.cssVar }}>
      <button
        className="db-step-header"
        onClick={() => setOpen(o => !o)}
        aria-expanded={effectiveOpen}
        aria-label={meta.label}
      >
        <span className="db-step-dot" style={{ background: meta.cssVar }} />
        <span className="db-step-icon">{meta.icon}</span>
        <span className="db-step-label">{meta.label}</span>
        {isToolDecision && (
          <span className={`db-tool-decision ${hasTool ? "has-tool" : "no-tool"}`}>
            {hasTool ? `→ calls ${decidedTool[0]?.name ?? "tool"}` : "→ direct answer"}
          </span>
        )}
        <span className="db-step-elapsed">+{formatMs(step.elapsed_ms)}</span>
        <span className={`db-chevron ${effectiveOpen ? "open" : ""}`}><ChevronIcon /></span>
      </button>

      <div className={`db-step-body-wrap ${effectiveOpen ? "open" : ""}`}>
        <div className="db-step-body-inner">
          <div className="db-step-body">
            {isSchema && (
              <pre className="db-schema-text">{String((step.data as Record<string, unknown>).schema ?? "")}</pre>
            )}
            {isToolInput && (
              <div className="db-tool-section">
                <div className="db-section-title">Tool: {String((step.data as Record<string, unknown>).tool_name)}</div>
                <CodeBlock text={String((step.data as Record<string, unknown>).input ?? "")} />
              </div>
            )}
            {isToolOutput && (
              <div className="db-tool-section">
                <div className="db-section-title">Result</div>
                <CodeBlock text={String((step.data as Record<string, unknown>).result ?? "")} className="db-code-output" />
              </div>
            )}
            {isFinalAnswer && (
              <div className="db-final-answer">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {String((step.data as Record<string, unknown>).answer ?? "")}
                </ReactMarkdown>
              </div>
            )}
            {!isSchema && !isToolInput && !isToolOutput && !isFinalAnswer && (
              <JsonBlock data={step.data} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Trace Detail ─────────────────────────────────────────────────────────────

function TraceDetail({ trace }: { trace: Trace }) {
  const [hoveredType, setHoveredType] = useState<string | null>(null);
  const [allOpen, setAllOpen]         = useState<boolean | undefined>(undefined);

  return (
    <div className="db-detail">
      <div className="db-detail-header">
        <div className="db-detail-query">{trace.query}</div>
        <div className="db-detail-meta">
          <span className="db-tag db-tag-file"><FileIcon /> {trace.filename}</span>
          {trace.tool_used && (
            <span className="db-tag db-tag-tool"><ToolTagIcon /> {trace.tool_used}</span>
          )}
          <span className={`db-status-badge db-status-${trace.status}`}>
            {trace.status === "running" ? "running" : trace.status === "success" ? "✓ success" : "✗ error"}
          </span>
          <span className="db-tag db-tag-time">{formatMs(trace.total_ms)}</span>
          <button
            className="db-expand-all-btn"
            onClick={() => setAllOpen(a => a === true ? false : true)}
            aria-label="Toggle all steps"
          >
            {allOpen ? <><CollapseIcon /> Collapse all</> : <><ExpandIcon /> Expand all</>}
          </button>
        </div>
        {trace.total_ms && trace.total_ms > 0 && (
          <Waterfall
            steps={trace.steps}
            totalMs={trace.total_ms}
            hoveredType={hoveredType}
            onHover={setHoveredType}
          />
        )}
      </div>
      <div className="db-timeline">
        {trace.steps.length === 0 ? (
          <div className="db-empty-steps">No steps recorded yet…</div>
        ) : (
          trace.steps.map((step, i) => (
            <StepCard
              key={`${step.type}-${i}`}
              step={step}
              index={i}
              forceOpen={allOpen}
            />
          ))
        )}
      </div>
    </div>
  );
}

// ─── Trace Row ────────────────────────────────────────────────────────────────

function TraceRow({
  trace, selected, onClick, isNew,
}: {
  trace: Trace;
  selected: boolean;
  onClick: () => void;
  isNew?: boolean;
}) {
  return (
    <button
      className={`db-trace-row ${selected ? "selected" : ""} db-status-border-${trace.status} ${isNew ? "db-trace-row--new" : ""}`}
      onClick={onClick}
      aria-pressed={selected}
      aria-label={trace.query}
    >
      <div className="db-trace-row-top">
        <span className="db-trace-query">{trace.query}</span>
        <span className={`db-dot db-dot-${trace.status}`} aria-label={trace.status} />
      </div>
      <div className="db-trace-row-bottom">
        <span className="db-trace-file" title={trace.filename}>{trace.filename}</span>
        <span className="db-trace-dur">{formatMs(trace.total_ms)}</span>
        <span className="db-trace-time" title={formatTime(trace.created_at)}>
          {formatRelative(trace.created_at)}
        </span>
      </div>
      {trace.tool_used && (
        <div className="db-trace-tool-tag"><ToolTagIcon /> {trace.tool_used}</div>
      )}
    </button>
  );
}

// ─── Traces Panel ─────────────────────────────────────────────────────────────

const TOOL_OPTIONS = ["all", "pandas_sandbox", "get_csv_stats", "search_documents"];

function TracesPanel({
  traces,
  selectedId,
  setSelectedId,
  mobileDetailOpen,
  setMobileDetailOpen,
  onBackToChat,
}: {
  traces: Trace[];
  selectedId: string | null;
  setSelectedId: (id: string) => void;
  mobileDetailOpen: boolean;
  setMobileDetailOpen: (v: boolean) => void;
  onBackToChat: () => void;
}) {
  const [search,     setSearch]     = useState("");
  const [toolFilter, setToolFilter] = useState("all");
  const [fileFilter, setFileFilter] = useState("all");
  const [newestId,   setNewestId]   = useState<string | null>(null);

  // Track newest trace arrival for highlight animation
  const prevLenRef = useRef(traces.length);
  useEffect(() => {
    if (traces.length > prevLenRef.current) {
      setNewestId(traces[0]?.id ?? null);
      const t = setTimeout(() => setNewestId(null), 2500);
      prevLenRef.current = traces.length;
      return () => clearTimeout(t);
    }
    prevLenRef.current = traces.length;
  }, [traces]);

  // Unique file names for filter dropdown
  const uniqueFiles = Array.from(new Set(traces.map(t => t.filename)));

  const filtered = traces.filter(t => {
    const q = search.toLowerCase();
    const matchSearch = !q || t.query.toLowerCase().includes(q) || t.filename.toLowerCase().includes(q);
    const matchTool   = toolFilter === "all" || t.tool_used === toolFilter;
    const matchFile   = fileFilter === "all" || t.filename === fileFilter;
    return matchSearch && matchTool && matchFile;
  });

  const hasFilters = search !== "" || toolFilter !== "all" || fileFilter !== "all";

  // Auto-select newest on desktop when nothing selected
  useEffect(() => {
    if (!selectedId && filtered.length > 0 && window.innerWidth > 700) {
      setSelectedId(filtered[0].id);
    }
  }, [filtered, selectedId, setSelectedId]);

  // j/k keyboard nav
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "TEXTAREA") return;
      const idx = filtered.findIndex(t => t.id === selectedId);
      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        const next = filtered[Math.min(idx + 1, filtered.length - 1)];
        if (next) setSelectedId(next.id);
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        const prev = filtered[Math.max(idx - 1, 0)];
        if (prev) setSelectedId(prev.id);
      } else if (e.key === "Enter" && selectedId) {
        setMobileDetailOpen(true);
      } else if (e.key === "Escape") {
        setMobileDetailOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [filtered, selectedId, setSelectedId, setMobileDetailOpen]);

  const selectedTrace = traces.find(t => t.id === selectedId) ?? null;

  return (
    <div className={`db-body ${mobileDetailOpen ? "db-body--detail-open" : ""}`}>
      {/* Sidebar */}
      <aside className="db-sidebar">
        <div className="db-filter-area">
          <div className="db-filter-row">
            <input
              className="db-search"
              placeholder="Search traces…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              aria-label="Search traces"
            />
          </div>
          <div className="db-filter-row">
            <select
              className="db-filter-select"
              value={toolFilter}
              onChange={e => setToolFilter(e.target.value)}
              aria-label="Filter by tool"
              style={{ flex: 1 }}
            >
              {TOOL_OPTIONS.map(t => <option key={t} value={t}>{t === "all" ? "All tools" : t}</option>)}
            </select>
            <select
              className="db-filter-select"
              value={fileFilter}
              onChange={e => setFileFilter(e.target.value)}
              aria-label="Filter by file"
              style={{ flex: 1 }}
            >
              <option value="all">All files</option>
              {uniqueFiles.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
            {hasFilters && (
              <button
                className="db-filter-clear"
                onClick={() => { setSearch(""); setToolFilter("all"); setFileFilter("all"); }}
                aria-label="Clear filters"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        <div className="db-trace-list" role="listbox" aria-label="Traces">
          {filtered.length === 0 ? (
            <div className="db-no-traces">
              {traces.length === 0
                ? "No traces yet.\nSend a query in chat to see it here."
                : "No traces match your search or filters."}
            </div>
          ) : (
            filtered.map(trace => (
              <TraceRow
                key={trace.id}
                trace={trace}
                selected={trace.id === selectedId}
                isNew={trace.id === newestId}
                onClick={() => { setSelectedId(trace.id); setMobileDetailOpen(true); }}
              />
            ))
          )}
        </div>
      </aside>

      {/* Detail */}
      <main className="db-main">
        <button
          className="db-back-btn"
          onClick={() => setMobileDetailOpen(false)}
          aria-label="Back to trace list"
          style={{ margin: "12px 16px 0", display: mobileDetailOpen ? undefined : "none" }}
        >
          <ChevronLeftIcon /> Back
        </button>
        {selectedTrace ? (
          <TraceDetail trace={selectedTrace} />
        ) : (
          <div className="db-placeholder">
            <div className="db-placeholder-icon"><PlaceholderIcon /></div>
            <div className="db-placeholder-text">Select a trace to inspect it</div>
            <div className="db-placeholder-sub">
              Upload a file and ask a question — every request will appear here with full LLM payloads, tool calls, and timing.
            </div>
            <button className="db-placeholder-action" onClick={onBackToChat}>
              <ChevronLeftIcon /> Back to chat
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

// ─── Sparkline ────────────────────────────────────────────────────────────────

function Sparkline({ values, selectedIdx, color }: { values: number[]; selectedIdx: number; color: string }) {
  if (values.length < 2) return null;
  const W = 80, H = 24, PAD = 2;
  const min = Math.min(...values), max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => {
    const x = PAD + (i / (values.length - 1)) * (W - PAD * 2);
    const y = H - PAD - ((v - min) / range) * (H - PAD * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const selPt = pts[selectedIdx]?.split(",") ?? ["0", "0"];
  return (
    <svg className="ev-sparkline" width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
      <polyline className="ev-sparkline-line" points={pts.join(" ")} stroke={color} strokeOpacity="0.5" />
      <circle className="ev-sparkline-dot" cx={selPt[0]} cy={selPt[1]} r="3" fill={color} stroke="var(--db-surface)" />
    </svg>
  );
}

// ─── Metric Card ──────────────────────────────────────────────────────────────

function MetricCard({ metricKey, value, prevValue, allValues, selectedIdx }: {
  metricKey: string;
  value: number;
  prevValue?: number;
  allValues?: number[];
  selectedIdx?: number;
}) {
  const color  = scoreColor(value);
  const bg     = scoreBg(value);
  const border = scoreBorder(value);
  const delta  = prevValue !== undefined ? value - prevValue : null;

  return (
    <div className="ev-metric-card" style={{ borderColor: border, background: bg }}>
      <div className="ev-metric-name">{metricLabel(metricKey)}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <div className="ev-metric-value" style={{ color }}>{value.toFixed(2)}</div>
        {allValues && allValues.length > 1 && selectedIdx !== undefined && (
          <Sparkline values={allValues} selectedIdx={selectedIdx} color={scoreColor(value)} />
        )}
      </div>
      {delta !== null && (
        <div className={`ev-metric-delta ${delta > 0.001 ? "ev-metric-delta--up" : delta < -0.001 ? "ev-metric-delta--down" : "ev-metric-delta--flat"}`}>
          {delta > 0.001 ? `▲ +${delta.toFixed(3)}` : delta < -0.001 ? `▼ ${delta.toFixed(3)}` : "— no change"}
        </div>
      )}
    </div>
  );
}

// ─── Score Badge ──────────────────────────────────────────────────────────────

function ScoreBadge({ value, label }: { value: number; label: string }) {
  return (
    <div className="ev-score-badge" style={{ background: scoreBg(value), borderColor: scoreBorder(value), color: scoreColor(value) }}>
      <span className="ev-score-label">{label}</span>
      <span className="ev-score-value">{value.toFixed(2)}</span>
    </div>
  );
}

// ─── Config diff helpers ───────────────────────────────────────────────────────

function flattenLeaves(obj: unknown, prefix = ""): Record<string, string> {
  if (obj === null || obj === undefined) return {};
  if (typeof obj !== "object" || Array.isArray(obj)) return { [prefix]: String(obj) };
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

function diffConfigKeys(a: EvalConfig | undefined, b: EvalConfig | undefined): Set<string> {
  const fa = flattenLeaves(a), fb = flattenLeaves(b);
  const all = new Set([...Object.keys(fa), ...Object.keys(fb)]);
  const changed = new Set<string>();
  for (const k of all) { if (fa[k] !== fb[k]) changed.add(k); }
  return changed;
}

type ConfigLeaf    = { key: string; dottedPath: string; value: string };
type ConfigSection = { title: string; rows: ConfigLeaf[]; nested: ConfigSection[] };

function buildConfigSections(config: EvalConfig, pathPrefix = ""): ConfigSection[] {
  const sections: ConfigSection[] = [];
  for (const [k, v] of Object.entries(config)) {
    const dottedPath = pathPrefix ? `${pathPrefix}.${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      const nested = buildConfigSections(v as EvalConfig, dottedPath);
      const rows: ConfigLeaf[] = [];
      for (const [sk, sv] of Object.entries(v as EvalConfig)) {
        if (sv === null || typeof sv !== "object" || Array.isArray(sv)) {
          rows.push({ key: prettyKey(sk), dottedPath: `${dottedPath}.${sk}`, value: String(sv ?? "") });
        }
      }
      sections.push({ title: prettyKey(k), rows, nested: nested.filter(n => n.rows.length > 0 || n.nested.length > 0) });
    } else {
      sections.push({ title: "__scalar__", rows: [{ key: prettyKey(k), dottedPath, value: String(v ?? "") }], nested: [] });
    }
  }
  return sections;
}

function renderConfigSections(sections: ConfigSection[], diffKeys: Set<string>, depth = 0): React.ReactNode {
  const scalars = sections.filter(s => s.title === "__scalar__");
  const groups  = sections.filter(s => s.title !== "__scalar__");
  return (
    <>
      {scalars.length > 0 && (
        <div className="ev-cfg-section" style={{ marginLeft: depth * 12 }}>
          {scalars.map(s => s.rows[0]).map(row => (
            <div key={row.dottedPath} className={`ev-cfg-row ${diffKeys.has(row.dottedPath) ? "ev-cfg-row--changed" : ""}`}>
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
                <div key={row.dottedPath} className={`ev-cfg-row ${diffKeys.has(row.dottedPath) ? "ev-cfg-row--changed" : ""}`}>
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

// ─── Investigation Notes parser ───────────────────────────────────────────────

/** Cleans up typography: -- → —, multiple spaces, trailing periods. */
function cleanBullet(s: string): string {
  return s
    .replace(/\s*--\s*/g, " — ")     // em dash
    .replace(/\s{2,}/g, " ")          // collapse spaces
    .replace(/\.\s*$/, "")            // drop trailing period (we add it back consistently)
    .trim();
}

/** Cap first letter. */
function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Parses a free-prose description into a context header + bullet array.
 *
 * Strategy (in order):
 *  1. Split on "(N)" numbered markers — most common format in eval notes.
 *     Everything before (1) becomes the context line.
 *     Trailing sentences after the last numbered bullet are split further.
 *  2. If no (N) markers, split on double-newlines.
 *  3. Otherwise fall back to sentence splitting.
 */
function parseNotes(text: string): { context: string; bullets: string[] } {
  const raw = text.trim();

  // ── Strategy 1: numbered markers ──────────────────────────────────────────
  const markerRe = /\s*\(\d+\)\s*/g;
  const markers  = [...raw.matchAll(markerRe)];

  if (markers.length >= 1) {
    const firstIdx = markers[0].index!;
    const context  = raw.slice(0, firstIdx).replace(/:\s*$/, "").trim(); // strip trailing colon
    const bullets: string[] = [];

    for (let i = 0; i < markers.length; i++) {
      const start = markers[i].index! + markers[i][0].length;
      const end   = i + 1 < markers.length ? markers[i + 1].index! : raw.length;
      let chunk   = raw.slice(start, end).trim();

      // Last numbered bullet may have appended sentences — split them off
      if (i === markers.length - 1) {
        const tailSplit = chunk.split(/\.\s+(?=Also |Confirmed |Additionally |Furthermore |Note )/);
        if (tailSplit.length > 1) {
          bullets.push(cap(cleanBullet(tailSplit[0])));
          for (let j = 1; j < tailSplit.length; j++) {
            const extra = tailSplit[j].trim();
            if (extra) bullets.push(cap(cleanBullet(extra)));
          }
          continue;
        }
      }
      bullets.push(cap(cleanBullet(chunk)));
    }

    return { context, bullets };
  }

  // ── Strategy 2: paragraph blocks ──────────────────────────────────────────
  const paras = raw.split(/\n\n+/).map(p => p.trim()).filter(Boolean);
  if (paras.length > 1) {
    return { context: "", bullets: paras.map(p => cap(cleanBullet(p))) };
  }

  // ── Strategy 3: sentence split ────────────────────────────────────────────
  const sentences = raw.match(/[^.!?]+[.!?]+(?:\s|$)/g);
  if (sentences && sentences.length > 1) {
    return { context: "", bullets: sentences.map(s => cap(cleanBullet(s))) };
  }

  return { context: "", bullets: [cap(cleanBullet(raw))] };
}

// ─── Investigation Notes component ────────────────────────────────────────────

function InvestigationNotes({ description }: { description: string }) {
  const [open, setOpen] = useState(true);
  const { context, bullets } = parseNotes(description);
  const preview = description.replace(/\n/g, " ").slice(0, 100).trim() + (description.length > 100 ? "…" : "");

  return (
    <div className="ev-notes-card">
      <button
        className="ev-notes-header"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        aria-label="Toggle investigation notes"
      >
        <span className="ev-notes-icon-wrap" aria-hidden><NotesIcon /></span>
        <span className="ev-notes-title-col">
          <span className="ev-notes-title">Investigation Notes</span>
          {!open && <span className="ev-notes-subtitle">{preview}</span>}
        </span>
        <span className={`ev-notes-chevron ${open ? "open" : ""}`} aria-hidden><ChevronIcon /></span>
      </button>

      {open && (
        <div className="ev-notes-body">
          {/* Context / intro line */}
          {context && <p className="ev-notes-context">{context}:</p>}

          {/* Bullet list */}
          <ul className="ev-notes-bullets">
            {bullets.map((b, i) => (
              <li key={i} className="ev-notes-bullet">{b}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ─── Question Row ─────────────────────────────────────────────────────────────

const Q_PAGE = 20;

function EvalQuestionRow({ item, index }: { item: EvalQuestion; index: number }) {
  const [open, setOpen] = useState(false);
  const s = item.scores;
  return (
    <div className="ev-q-row">
      <button className="ev-q-header" onClick={() => setOpen(o => !o)} aria-expanded={open}>
        <span className="ev-q-num">Q{index + 1}</span>
        <span className="ev-q-text">{truncate(item.question, 90)}</span>
        <div className="ev-q-scores">
          {Object.entries(s).map(([k, v]) => (
            <span key={k} className="ev-q-score-dot" style={{ background: scoreColor(v) }} title={`${metricLabel(k)}: ${v.toFixed(2)}`} />
          ))}
        </div>
        <span className={`ev-run-chevron ${open ? "open" : ""}`}><ChevronIcon /></span>
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

// ─── Eval Run View ────────────────────────────────────────────────────────────

function EvalRunView({ run, prevRun, allRunsOfType }: {
  run: EvalRun;
  prevRun?: EvalRun | null;
  allRunsOfType?: EvalRun[];
}) {
  const [cfgOpen, setCfgOpen] = useState(false);
  const [qSearch, setQSearch] = useState("");
  const [failOnly, setFailOnly] = useState(false);
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(1);

  const avg      = run.averages;
  const diffKeys = diffConfigKeys(run.config, prevRun?.config);
  const hasDiff  = prevRun != null && diffKeys.size > 0;
  const cfgSections = run.config ? buildConfigSections(run.config) : [];

  // Changed-key chips: show old → new
  const prevFlat = flattenLeaves(prevRun?.config);
  const curFlat  = flattenLeaves(run.config);

  // Sparkline values per metric across all runs of this type
  const metricsKeys = Object.keys(avg);
  const selectedRunIdx = allRunsOfType
    ? allRunsOfType.findIndex(r => r.timestamp === run.timestamp)
    : -1;

  // Question filtering + sort + pagination
  const metricCols = Object.keys(run.per_question[0]?.scores ?? {});

  function toggleSort(k: string) {
    if (sortKey === k) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortKey(k); setSortDir("desc"); }
  }

  const questions = run.per_question
    .filter(q => !qSearch || q.question.toLowerCase().includes(qSearch.toLowerCase()))
    .filter(q => !failOnly || Object.values(q.scores).some(v => v < 0.5))
    .sort((a, b) => {
      if (!sortKey) return 0;
      const av = a.scores[sortKey] ?? 0;
      const bv = b.scores[sortKey] ?? 0;
      return sortDir === "desc" ? bv - av : av - bv;
    });

  const visible = questions.slice(0, page * Q_PAGE);
  const remaining = questions.length - visible.length;

  return (
    <div className="ev-run-view">
    <div className="ev-panel-body">
      {/* Run identity */}
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

      {/* Investigation Notes */}
      {run.description && (
        <div className="ev-section ev-section--notes">
          <InvestigationNotes description={run.description} />
        </div>
      )}

      {/* Metric cards with deltas + sparklines */}
      <div className="ev-section">
        <div className="ev-section-title">Average Scores — {run.per_question.length} question{run.per_question.length !== 1 ? "s" : ""}</div>
        <div className="ev-metric-cards">
          {metricsKeys.map(k => {
            const allVals = allRunsOfType?.map(r => r.averages[k]).filter(v => v !== undefined) as number[] | undefined;
            return (
              <MetricCard
                key={k}
                metricKey={k}
                value={avg[k]}
                prevValue={prevRun?.averages[k]}
                allValues={allVals}
                selectedIdx={selectedRunIdx >= 0 ? selectedRunIdx : undefined}
              />
            );
          })}
        </div>
      </div>

      {/* Config with diff chips */}
      {cfgSections.length > 0 && (
        <div className="ev-section">
          <button className="ev-cfg-toggle" onClick={() => setCfgOpen(o => !o)} aria-expanded={cfgOpen}>
            <span className="ev-cfg-toggle-left">
              <ConfigIcon />
              <span>Configuration</span>
              {hasDiff && <span className="ev-cfg-diff-count">{diffKeys.size} change{diffKeys.size !== 1 ? "s" : ""} vs prev</span>}
            </span>
            <span className={`ev-run-chevron ${cfgOpen ? "open" : ""}`}><ChevronIcon /></span>
          </button>
          {cfgOpen && (
            <div className="ev-cfg-nested-grid">
              {hasDiff && (
                <>
                  <div className="ev-cfg-diff-banner">
                    ↕ Comparing against <strong>{prevRun!.label || "previous run"}</strong> — fields highlighted in amber changed
                  </div>
                  <div className="ev-cfg-diff-chips">
                    {Array.from(diffKeys).slice(0, 8).map(k => (
                      <span key={k} className="ev-cfg-diff-chip">
                        <span className="ev-cfg-diff-chip-key">{k.split(".").pop()}</span>
                        <span className="ev-cfg-diff-chip-old">{prevFlat[k] ?? "—"}</span>
                        <span className="ev-cfg-diff-chip-arrow">→</span>
                        <span className="ev-cfg-diff-chip-new">{curFlat[k] ?? "—"}</span>
                      </span>
                    ))}
                  </div>
                </>
              )}
              {renderConfigSections(cfgSections, diffKeys)}
            </div>
          )}
        </div>
      )}

      {/* Per-question breakdown */}
      <div className="ev-section">
        <div className="ev-section-title">Per-Question Breakdown</div>
        <div className="ev-questions-controls">
          <input
            className="ev-q-search"
            placeholder="Search questions…"
            value={qSearch}
            onChange={e => { setQSearch(e.target.value); setPage(1); }}
            aria-label="Search questions"
          />
          <button
            className={`ev-q-toggle-btn ${failOnly ? "active" : ""}`}
            onClick={() => { setFailOnly(f => !f); setPage(1); }}
          >
            {failOnly ? "Show all" : "Failing only"}
          </button>
          {metricCols.length > 0 && (
            <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ fontSize: 10, color: "var(--db-text-3)", fontWeight: 600, textTransform: "uppercase" }}>Sort:</span>
              {metricCols.map(k => (
                <button
                  key={k}
                  className={`ev-q-sort-btn ${sortKey === k ? "active" : ""}`}
                  onClick={() => toggleSort(k)}
                >
                  {metricLabel(k).split(" ")[0]}
                  {sortKey === k && (sortDir === "desc" ? " ▼" : " ▲")}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="ev-questions">
          {visible.map((q, qi) => <EvalQuestionRow key={qi} item={q} index={qi} />)}
        </div>
        {remaining > 0 && (
          <button className="ev-load-more-btn" onClick={() => setPage(p => p + 1)}>
            Load more ({remaining} remaining)
          </button>
        )}
        {visible.length === 0 && (
          <div className="db-no-traces" style={{ padding: "24px 0" }}>
            {failOnly ? "No failing questions in this run." : "No questions match your search."}
          </div>
        )}
      </div>
    </div>
    </div>
  );
}

// ─── Evaluations Panel ────────────────────────────────────────────────────────

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
      .then(r => { if (!r.ok) throw new Error(`Server error ${r.status}`); return r.json() as Promise<EvalRun[]>; })
      .then(data => { setAllRuns(data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="ev-state"><div className="ev-state-icon"><SpinnerIcon /></div><div className="ev-state-text">Loading eval history…</div></div>;
  if (error)   return <div className="ev-state"><div className="ev-state-icon" style={{ color: "var(--db-red)" }}><SearchIcon /></div><div className="ev-state-text">Failed to load: {error}</div></div>;

  const pdfRuns   = allRuns.filter(r => !r.type || r.type === "pdf");
  const csvRuns   = allRuns.filter(r => r.type === "csv");
  const activeRuns = typeTab === "pdf" ? pdfRuns : csvRuns;
  const selectedIdx = typeTab === "pdf" ? selectedPdfIdx : selectedCsvIdx;
  const setSelectedIdx = typeTab === "pdf" ? setSelectedPdfIdx : setSelectedCsvIdx;
  const selectedRun = activeRuns[selectedIdx] ?? null;
  const prevRun = selectedIdx + 1 < activeRuns.length ? activeRuns[selectedIdx + 1] : null;
  const totalRuns = pdfRuns.length + csvRuns.length;
  const emptyMsg = typeTab === "pdf" ? "No PDF evaluations run yet." : "No CSV evaluations run yet.";

  return (
    <div className="ev-panel">
      <div className="ev-header">
        <div className="ev-header-left">
          <span className="ev-header-title">Evaluation History</span>
          <span className="ev-header-count">{totalRuns} run{totalRuns !== 1 ? "s" : ""}</span>
        </div>
        <div className="ev-header-legend">
          <span className="ev-legend-dot" style={{ background: "var(--score-good)" }} /> ≥0.8
          <span className="ev-legend-dot" style={{ background: "var(--score-ok)", marginLeft: 10 }} /> 0.5–0.8
          <span className="ev-legend-dot" style={{ background: "var(--score-bad)", marginLeft: 10 }} /> &lt;0.5
        </div>
      </div>

      <div className="ev-type-tabs-bar">
        <div className="ev-type-tabs">
          <button className={`ev-type-tab ${typeTab === "pdf" ? "active" : ""}`} onClick={() => setTypeTab("pdf")}>
            <PdfIcon /> PDF
            {pdfRuns.length > 0 && <span className="ev-type-tab-count">{pdfRuns.length}</span>}
          </button>
          <button className={`ev-type-tab ${typeTab === "csv" ? "active" : ""}`} onClick={() => setTypeTab("csv")}>
            <CsvIcon /> CSV
            {csvRuns.length > 0 && <span className="ev-type-tab-count">{csvRuns.length}</span>}
          </button>
        </div>
        {activeRuns.length > 0 && (
          <div className="ev-run-select-wrap">
            <select
              className="ev-run-select"
              value={selectedIdx}
              onChange={e => setSelectedIdx(Number(e.target.value))}
              aria-label="Select evaluation run"
            >
              {activeRuns.map((run, i) => <option key={`${run.timestamp}-${i}`} value={i}>{formatDropdownLabel(run)}</option>)}
            </select>
            <span className="ev-run-select-chevron" aria-hidden>▾</span>
          </div>
        )}
      </div>

      <div className="ev-tab-content">
        {activeRuns.length === 0 ? (
          <div className="ev-state">
            <div className="ev-state-icon" style={{ color: "var(--db-text-3)" }}>{typeTab === "pdf" ? <PdfIcon /> : <CsvIcon />}</div>
            <div className="ev-state-text">{emptyMsg}</div>
            <div className="ev-state-sub">
              Run <code className="ev-code">python eval/run_eval.py</code> from the backend directory to generate a report.
            </div>
          </div>
        ) : selectedRun ? (
          <EvalRunView
            key={selectedRun.timestamp}
            run={selectedRun}
            prevRun={prevRun}
            allRunsOfType={activeRuns}
          />
        ) : null}
      </div>
    </div>
  );
}

// ─── Token Usage Panel ────────────────────────────────────────────────────────

const TOK_PAGE = 50;

function TokenPanel() {
  const [entries, setEntries]     = useState<TokenEntry[]>([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);
  const [sortCol, setSortCol]     = useState<keyof TokenEntry>("timestamp");
  const [sortDir, setSortDir]     = useState<"asc" | "desc">("desc");
  const [page, setPage]           = useState(1);

  useEffect(() => {
    fetch(`${API_BASE}/dashboard/tokens`)
      .then(r => { if (!r.ok) throw new Error(`Server error ${r.status}`); return r.json() as Promise<TokenEntry[]>; })
      .then(data => { setEntries(data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="ev-state"><div className="ev-state-icon"><SpinnerIcon /></div><div className="ev-state-text">Loading token usage…</div></div>;
  if (error)   return <div className="ev-state"><div className="ev-state-icon" style={{ color: "var(--db-red)" }}><SearchIcon /></div><div className="ev-state-text">Failed to load: {error}</div></div>;
  if (entries.length === 0) return (
    <div className="ev-state">
      <div className="ev-state-icon" style={{ color: "var(--db-text-3)" }}><SearchIcon /></div>
      <div className="ev-state-text">No token usage recorded yet.</div>
      <div className="ev-state-sub">Token usage is logged automatically on each query. Run a query to see data here.</div>
    </div>
  );

  const totalPrompt     = entries.reduce((s, e) => s + e.prompt_tokens, 0);
  const totalCompletion = entries.reduce((s, e) => s + e.completion_tokens, 0);
  const totalAll        = entries.reduce((s, e) => s + e.total_tokens, 0);
  const avgPerCall      = Math.round(totalAll / entries.length);

  const purposeMap: Record<string, number> = {};
  for (const e of entries) purposeMap[e.purpose] = (purposeMap[e.purpose] ?? 0) + e.total_tokens;
  const purposeEntries = Object.entries(purposeMap).sort((a, b) => b[1] - a[1]);
  const maxTokens = purposeEntries[0]?.[1] ?? 1;

  function toggleSort(col: keyof TokenEntry) {
    if (sortCol === col) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortCol(col); setSortDir("desc"); }
  }

  const sorted = [...entries].sort((a, b) => {
    const av = a[sortCol], bv = b[sortCol];
    const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
    return sortDir === "desc" ? -cmp : cmp;
  });

  const visible = sorted.slice(0, page * TOK_PAGE);
  const remaining = sorted.length - visible.length;

  function SortTh({ col, children }: { col: keyof TokenEntry; children: React.ReactNode }) {
    const active = sortCol === col;
    return (
      <th className={`sortable ${active ? "sorted" : ""}`} onClick={() => toggleSort(col)} aria-sort={active ? (sortDir === "asc" ? "ascending" : "descending") : "none"}>
        {children}
        <span className="tok-sort-icon">{active ? (sortDir === "desc" ? "▼" : "▲") : "⇅"}</span>
      </th>
    );
  }

  return (
    <div className="tok-panel">
      <div className="ev-header">
        <div className="ev-header-left">
          <span className="ev-header-title">Token Usage</span>
          <span className="ev-header-count">{entries.length} call{entries.length !== 1 ? "s" : ""}</span>
        </div>
      </div>
      <div className="ev-panel-body">
      <div className="tok-body">
        {/* Summary cards */}
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
          <div className="tok-card tok-card--calls">
            <span className="tok-card-value">{entries.length}</span>
            <span className="tok-card-label">API Calls</span>
          </div>
          <div className="tok-card tok-card--avg">
            <span className="tok-card-value">{formatNum(avgPerCall)}</span>
            <span className="tok-card-label">Avg / Call</span>
          </div>
        </div>

        {/* Purpose breakdown */}
        <div className="tok-purpose-section">
          <div className="tok-section-title">Breakdown by Purpose</div>
          <div className="tok-purpose-bars">
            {purposeEntries.map(([purpose, tokens]) => (
              <div key={purpose} className="tok-purpose-bar-row" title={`${formatNum(tokens)} tokens (${(tokens / totalAll * 100).toFixed(1)}%)`}>
                <span className="tok-purpose-bar-label" style={{ color: purposeColor(purpose) }}>{purpose}</span>
                <div className="tok-purpose-bar-track">
                  <div className="tok-purpose-bar-fill" style={{ width: `${Math.round((tokens / maxTokens) * 100)}%`, background: purposeColor(purpose) }} />
                </div>
                <span className="tok-purpose-bar-pct">{(tokens / totalAll * 100).toFixed(0)}%</span>
                <span className="tok-purpose-bar-value">{formatNum(tokens)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Table */}
        <div className="tok-table-section">
          <div className="tok-section-title">All Calls</div>
          <div className="tok-table-wrap">
            <table className="tok-table">
              <thead>
                <tr>
                  <SortTh col="timestamp">Timestamp</SortTh>
                  <th>Model</th>
                  <SortTh col="purpose">Purpose</SortTh>
                  <SortTh col="prompt_tokens">Prompt</SortTh>
                  <SortTh col="completion_tokens">Completion</SortTh>
                  <SortTh col="total_tokens">Total</SortTh>
                </tr>
              </thead>
              <tbody>
                {visible.map((e, i) => (
                  <tr key={i}>
                    <td className="tok-mono tok-ts">{formatUnixTs(e.timestamp)}</td>
                    <td className="tok-mono tok-model">{e.model}</td>
                    <td>
                      <span className="tok-purpose-pill" style={{ background: `${purposeColor(e.purpose)}18`, color: purposeColor(e.purpose), borderColor: `${purposeColor(e.purpose)}40` }}>
                        {e.purpose}
                      </span>
                    </td>
                    <td className="tok-num">{formatNum(e.prompt_tokens)}</td>
                    <td className="tok-num">{formatNum(e.completion_tokens)}</td>
                    <td className="tok-num tok-total">{formatNum(e.total_tokens)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {remaining > 0 && (
              <button className="tok-load-more-btn" onClick={() => setPage(p => p + 1)}>
                Load {Math.min(remaining, TOK_PAGE)} more ({remaining} remaining)
              </button>
            )}
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}

// ─── Dashboard Shell ──────────────────────────────────────────────────────────

interface DashboardProps { onClose: () => void; }

type Tab = "traces" | "evals" | "tokens";

export default function Dashboard({ onClose }: DashboardProps) {
  const [activeTab,        setActiveTab]        = useState<Tab>("traces");
  const [traces,           setTraces]           = useState<Trace[]>([]);
  const [selectedId,       setSelectedId]       = useState<string | null>(null);
  const [connected,        setConnected]        = useState(false);
  const [mobileDetailOpen, setMobileDetailOpen] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  // Reset mobile detail whenever tab changes
  useEffect(() => { setMobileDetailOpen(false); }, [activeTab]);

  useEffect(() => {
    const es = new EventSource(`${API_BASE}/dashboard/stream`);
    esRef.current = es;
    es.onopen  = () => setConnected(true);
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

  const handleSelectId = useCallback((id: string) => setSelectedId(id), []);

  return (
    <div className="db-root">
      {/* ── Topbar ── */}
      <header className="db-topbar">
        <div className="db-topbar-left">
          <div className="db-logo" aria-hidden>QM</div>
          <span className="db-title">Debug Logs</span>
          <span className={`db-live-dot ${connected ? "live" : "dead"}`} aria-label={connected ? "Live" : "Disconnected"} />
          <span className="db-live-label">{connected ? "Live" : "Reconnecting…"}</span>
        </div>

        {/* Segmented tab control */}
        <nav className="db-tabs" role="tablist" aria-label="Dashboard sections">
          <button
            role="tab"
            className={`db-tab ${activeTab === "traces" ? "active" : ""}`}
            onClick={() => setActiveTab("traces")}
            aria-selected={activeTab === "traces"}
          >
            Traces
            {traces.length > 0 && <span className="db-tab-count">{traces.length}</span>}
          </button>
          <button
            role="tab"
            className={`db-tab ${activeTab === "evals" ? "active" : ""}`}
            onClick={() => setActiveTab("evals")}
            aria-selected={activeTab === "evals"}
          >
            Evaluations
          </button>
          <button
            role="tab"
            className={`db-tab ${activeTab === "tokens" ? "active" : ""}`}
            onClick={() => setActiveTab("tokens")}
            aria-selected={activeTab === "tokens"}
          >
            Token Usage
          </button>
        </nav>

        <div className="db-topbar-right">
          {activeTab === "traces" && traces.length > 0 && (
            <span className="db-count">{traces.length} trace{traces.length !== 1 ? "s" : ""}</span>
          )}
          <button className="db-back-btn" onClick={onClose} aria-label="Back to chat">
            <ChevronLeftIcon /> Back to chat
          </button>
        </div>
      </header>

      {/* ── Tab panels ── */}
      {activeTab === "traces" && (
        <TracesPanel
          traces={traces}
          selectedId={selectedId}
          setSelectedId={handleSelectId}
          mobileDetailOpen={mobileDetailOpen}
          setMobileDetailOpen={setMobileDetailOpen}
          onBackToChat={onClose}
        />
      )}

      {activeTab === "evals" && (
        <div className="db-body db-body--evals">
          <main className="db-main db-main--evals"><EvalsPanel /></main>
        </div>
      )}

      {activeTab === "tokens" && (
        <div className="db-body db-body--evals">
          <main className="db-main db-main--evals"><TokenPanel /></main>
        </div>
      )}
    </div>
  );
}
