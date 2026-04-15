import { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

// ─── Types ────────────────────────────────────────────────────────────────────
interface Source {
  source: string;
  page: number;
  text: string;
  rerank_score: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  loading?: boolean;
}

interface DocumentStatus {
  [filename: string]: "processing" | "ready";
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

function uid() {
  return Math.random().toString(36).slice(2);
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatusDot({ status }: { status: "processing" | "ready" }) {
  return (
    <span className={`status-dot ${status}`}>
      {status === "processing" ? (
        <span className="spinner" />
      ) : (
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
          <path d="M2 5l2 2 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )}
    </span>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const shortName = source.source.replace(/^.*[\\/]/, "");
  const score = Math.round(source.rerank_score * 100);
  const pageNum = source.page != null ? Number(source.page) : null;

  return (
    <div className="source-card" style={{ "--i": index } as React.CSSProperties}>
      <div className="source-header" onClick={() => setExpanded(!expanded)}>
        <div className="source-meta">
          <span className="source-icon">
            <svg width="12" height="14" viewBox="0 0 12 14" fill="none">
              <rect x="1" y="1" width="10" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.2"/>
              <path d="M3.5 4.5h5M3.5 7h5M3.5 9.5h3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
            </svg>
          </span>
          <span className="source-filename">{shortName}</span>
          <span className="source-page">{pageNum != null ? `p. ${pageNum}` : "p. —"}</span>
        </div>
        <div className="source-right">
          <span className="source-score">{score}%</span>
          <span className={`source-chevron ${expanded ? "open" : ""}`}>
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <path d="M2 3.5l3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </span>
        </div>
      </div>
      {expanded && (
        <div className="source-text">{source.text}</div>
      )}
    </div>
  );
}

function DropZone({ onUpload, uploading }: { onUpload: (file: File) => void; uploading: boolean }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handle = (file: File) => {
    if (file.type === "application/pdf") onUpload(file);
  };

  return (
    <div
      className={`dropzone ${dragging ? "drag-over" : ""} ${uploading ? "uploading" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handle(f); }}
    >
      <input ref={inputRef} type="file" accept=".pdf" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) handle(f); e.target.value = ""; }} />
      <div className="dropzone-inner">
        {uploading ? (
          <>
            <div className="dz-spinner" />
            <span className="dz-label">Uploading…</span>
          </>
        ) : (
          <>
            <svg className="dz-icon" width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M12 16V8m0 0l-3 3m3-3l3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3 16.5V18a2 2 0 002 2h14a2 2 0 002-2v-1.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            <span className="dz-label">Drop PDF or <u>browse</u></span>
          </>
        )}
      </div>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="skeleton-block">
      <div className="skel skel-line" style={{ width: "82%" }} />
      <div className="skel skel-line" style={{ width: "67%" }} />
      <div className="skel skel-line" style={{ width: "74%" }} />
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [documents, setDocuments] = useState<DocumentStatus>({});
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll documents
  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API}/documents`);
      const data = await res.json();
      setDocuments(data.documents ?? {});
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    fetchDocuments();
    pollRef.current = setInterval(async () => {
      const res = await fetch(`${API}/documents`).catch(() => null);
      if (!res) return;
      const data = await res.json();
      const docs: DocumentStatus = data.documents ?? {};
      setDocuments(docs);
      const allReady = Object.values(docs).every((s) => s === "ready");
      if (allReady && pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [fetchDocuments]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error("Upload failed");
      await fetchDocuments();
      // restart polling
      if (!pollRef.current) {
        pollRef.current = setInterval(async () => {
          const r = await fetch(`${API}/documents`).catch(() => null);
          if (!r) return;
          const d = await r.json();
          const docs: DocumentStatus = d.documents ?? {};
          setDocuments(docs);
          if (Object.values(docs).every((s) => s === "ready") && pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
        }, 3000);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async () => {
    const q = input.trim();
    if (!q || querying) return;

    const userMsg: Message = { id: uid(), role: "user", content: q };
    const loadingMsg: Message = { id: uid(), role: "assistant", content: "", loading: true };

    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    setInput("");
    setQuerying(true);
    setError(null);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, conversation_history: [] }),
      });
      if (!res.ok) throw new Error("Query failed");
      const data = await res.json();
      setMessages((prev) =>
        prev.map((m) =>
          m.loading ? { ...m, loading: false, content: data.answer, sources: data.sources } : m
        )
      );
    } catch (e: unknown) {
      setMessages((prev) => prev.filter((m) => !m.loading));
      setError(e instanceof Error ? e.message : "Query failed");
    } finally {
      setQuerying(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleQuery(); }
  };

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
  };

  const docEntries = Object.entries(documents);
  const readyCount = docEntries.filter(([, s]) => s === "ready").length;

  return (
    <div className="app">
      {/* ── Sidebar ───────────────────────────────────── */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-head">
          <div className="logo">
            <span className="logo-mark">Q</span>
            <span className="logo-text">QueryMind</span>
          </div>
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="Toggle sidebar">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        <div className="sidebar-section">
          <p className="section-label">Documents</p>
          {docEntries.length > 0 && (
            <p className="doc-count">{readyCount}/{docEntries.length} ready</p>
          )}
          <div className="doc-list">
            {docEntries.length === 0 ? (
              <p className="empty-hint">No documents yet.</p>
            ) : (
              docEntries.map(([name, status]) => (
                <div key={name} className="doc-item">
                  <StatusDot status={status} />
                  <span className="doc-name" title={name}>{name}</span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="sidebar-section">
          <p className="section-label">Upload</p>
          <DropZone onUpload={handleUpload} uploading={uploading} />
        </div>

        <div className="sidebar-footer">
          <p className="footer-hint">PDF documents only · RAG powered</p>
        </div>
      </aside>

      {/* ── Main ──────────────────────────────────────── */}
      <main className="main">
        {/* Header */}
        <header className="topbar">
          {!sidebarOpen && (
            <button className="sidebar-toggle inline" onClick={() => setSidebarOpen(true)}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </button>
          )}
          <div className="topbar-center">
            <span className="topbar-title">Research Assistant</span>
            {docEntries.length > 0 && (
              <span className="topbar-badge">{docEntries.length} doc{docEntries.length !== 1 ? "s" : ""}</span>
            )}
          </div>
        </header>

        {/* Error banner */}
        {error && (
          <div className="error-banner">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.3"/>
              <path d="M7 4.5v3M7 9.5v.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
            </svg>
            {error}
            <button onClick={() => setError(null)} className="error-dismiss">✕</button>
          </div>
        )}

        {/* Chat */}
        <div className="chat-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">
                <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                  <circle cx="20" cy="20" r="18" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 3"/>
                  <path d="M13 20h14M20 13v14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <h2 className="empty-title">Ask across your documents</h2>
              <p className="empty-sub">Upload PDFs in the sidebar, then ask any question. Answers will cite the exact source and page.</p>
              {docEntries.length > 0 && (
                <div className="sample-questions">
                  {["Summarize the key findings", "What methodology was used?", "List all recommendations"].map((q) => (
                    <button key={q} className="sample-q" onClick={() => setInput(q)}>
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="messages">
              {messages.map((msg) => (
                <div key={msg.id} className={`message-row ${msg.role}`}>
                  <div className="message-bubble">
                    <div className="avatar">{msg.role === "user" ? "U" : "Q"}</div>
                    <div className="message-content">
                      {msg.loading ? (
                        <Skeleton />
                      ) : (
                        <>
                          <p className="message-text">{msg.content}</p>
                          {msg.sources && msg.sources.length > 0 && (
                            <div className="sources-section">
                              <p className="sources-label">
                                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                                  <circle cx="6" cy="6" r="5" stroke="currentColor" strokeWidth="1.2"/>
                                  <path d="M6 4v3.5M6 8.5v.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                                </svg>
                                {msg.sources.length} source{msg.sources.length !== 1 ? "s" : ""}
                              </p>
                              <div className="sources-list">
                                {msg.sources.map((src, i) => (
                                  <SourceCard key={i} source={src} index={i} />
                                ))}
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="input-area">
          <div className={`input-box ${querying ? "disabled" : ""}`}>
            <textarea
              ref={textareaRef}
              className="chat-input"
              placeholder={docEntries.length === 0 ? "Upload documents first…" : "Ask a question…"}
              value={input}
              onChange={handleTextareaInput}
              onKeyDown={handleKeyDown}
              disabled={querying || docEntries.length === 0}
              rows={1}
            />
            <button
              className={`send-btn ${(!input.trim() || querying) ? "dim" : ""}`}
              onClick={handleQuery}
              disabled={!input.trim() || querying || docEntries.length === 0}
              aria-label="Send"
            >
              {querying ? (
                <span className="btn-spinner" />
              ) : (
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M14 8H2m0 0l5-5M2 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
            </button>
          </div>
          <p className="input-hint">Enter to send · Shift+Enter for new line</p>
        </div>
      </main>
    </div>
  );
}