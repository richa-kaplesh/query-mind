import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./Dashboard.css";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Step {
  type: string;
  elapsed_ms: number;
  data: Record<string, unknown>;
}

export interface Trace {
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

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatMs(ms: number | null) {
  if (ms === null) return "—";
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(2)}s`;
}

function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatRelative(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60)  return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
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
function PlaceholderIcon() { return <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>; }

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

export function TracesPanel({
  traces,
  selectedId,
  setSelectedId,
  mobileDetailOpen,
  setMobileDetailOpen,
  emptyAction,
}: {
  traces: Trace[];
  selectedId: string | null;
  setSelectedId: (id: string) => void;
  mobileDetailOpen: boolean;
  setMobileDetailOpen: (v: boolean) => void;
  emptyAction?: React.ReactNode;
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
            {emptyAction}
          </div>
        )}
      </main>
    </div>
  );
}

