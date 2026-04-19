import { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";
import { useStream } from "./useStream";
import type { Source } from "./useStream";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  /** true only while this message is actively streaming */
  streaming?: boolean;
  timestamp: string;
}

interface DocumentStatus {
  [filename: string]: "processing" | "ready";
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const API = "http://localhost:8000";
function uid() { return Math.random().toString(36).slice(2); }
function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatusPip({ status }: { status: "processing" | "ready" }) {
  return <span className={`status-pip ${status}`} />;
}

/** Blinking text cursor shown at end of streaming bubble */
function StreamCursor() {
  return <span className="stream-cursor" aria-hidden="true" />;
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [open, setOpen] = useState(false);
  const name = source.source.replace(/^.*[\\/]/, "");
  const score = Math.round(source.rerank_score * 100);
  const page = source.page != null ? Number(source.page) : null;

  return (
    <div
      className="source-item"
      style={{ animationDelay: `${index * 60}ms` }}
      onClick={() => setOpen(!open)}
    >
      <div className="source-item-left">
        <div className="source-row1">
          <span className="source-name">{name}</span>
          <span className="source-pg">{page != null ? `p. ${page}` : "p. —"}</span>
        </div>
        <div className={`source-excerpt ${open ? "expanded" : ""}`}>{source.text}</div>
      </div>
      <div className="source-score">
        <span className="score-num">{score}%</span>
        <div className="score-bar"><div className="score-fill" style={{ width: `${score}%` }} /></div>
        <span className={`source-chevron ${open ? "open" : ""}`}>
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M2 4l3 3 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </div>
    </div>
  );
}

function DropZone({ onUpload, uploading }: { onUpload: (f: File) => void; uploading: boolean }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const handle = (f: File) => { if (f.type === "application/pdf") onUpload(f); };

  return (
    <div
      className={`upload-zone ${dragging ? "drag-over" : ""} ${uploading ? "uploading" : ""}`}
      onClick={() => !uploading && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handle(f); }}
    >
      <input
        ref={inputRef} type="file" accept=".pdf" hidden
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handle(f); e.target.value = ""; }}
      />
      <div className="upload-zone-icon">
        {uploading ? <span className="dz-spinner" /> : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        )}
      </div>
      <div className="upload-zone-label">
        {uploading ? "Uploading…" : <>Drop PDF or <span>browse files</span></>}
      </div>
      <div className="upload-zone-hint">PDF only · max 50MB</div>
    </div>
  );
}

/** Shown only until the first token arrives */
function Skeleton() {
  return (
    <div className="skeleton">
      <div className="skel-line" style={{ width: "76%" }} />
      <div className="skel-line" style={{ width: "60%" }} />
      <div className="skel-line" style={{ width: "68%" }} />
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [documents, setDocuments] = useState<DocumentStatus>({});
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Tracks which assistant message is currently receiving tokens
  const streamingMsgIdRef = useRef<string | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Streaming hook ──────────────────────────────────────────────────────────
  const { streamedAnswer, sources, isStreaming, startStream, reset } = useStream(API);

  // Push incoming tokens into the live message on every render
  useEffect(() => {
    const id = streamingMsgIdRef.current;
    if (!id) return;
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, content: streamedAnswer, streaming: isStreaming } : m
      )
    );
  }, [streamedAnswer, isStreaming]);

  // Once [DONE] is received, attach sources and clear the streaming flag
  useEffect(() => {
    const id = streamingMsgIdRef.current;
    if (!id || isStreaming) return;
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, sources, streaming: false } : m
      )
    );
    streamingMsgIdRef.current = null;
  }, [isStreaming, sources]);

  // ── Document polling ────────────────────────────────────────────────────────
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
      if (Object.values(docs).every((s) => s === "ready") && pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [fetchDocuments]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Upload ──────────────────────────────────────────────────────────────────
  const handleUpload = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error("Upload failed");
      await fetchDocuments();
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

  // ── Query ───────────────────────────────────────────────────────────────────
  const handleQuery = async () => {
    const q = input.trim();
    if (!q || isStreaming) return;

    const userMsg: Message = { id: uid(), role: "user", content: q, timestamp: now() };
    const assistantId = uid();
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
      timestamp: now(),
    };

    // Register this ID before startStream fires any state updates
    streamingMsgIdRef.current = assistantId;
    reset();

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setError(null);
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    try {
      await startStream(q);
    } catch (e: unknown) {
      // Remove the broken assistant message and surface the error
      setMessages((prev) => prev.filter((m) => m.id !== assistantId));
      streamingMsgIdRef.current = null;
      setError(e instanceof Error ? e.message : "Stream failed");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleQuery(); }
  };

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  };

  const docEntries = Object.entries(documents);
  const readyCount = docEntries.filter(([, s]) => s === "ready").length;

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="app">

      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-logo">
          <div className="logo-icon">Q</div>
          <div>
            <div className="logo-name">QueryMind</div>
            <div className="logo-sub">RAG · v2.0</div>
          </div>
          <button className="sidebar-toggle-btn" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className="sidebar-section">
          <div className="section-title">
            Documents
            {docEntries.length > 0 && (
              <span className="doc-count-badge">{readyCount}/{docEntries.length}</span>
            )}
          </div>
          <div className="doc-list">
            {docEntries.length === 0 ? (
              <div className="doc-empty">No documents yet</div>
            ) : (
              docEntries.map(([name, status]) => (
                <div key={name} className="doc-item">
                  <div className="doc-icon">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                      <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <span className="doc-name" title={name}>{name}</span>
                  <StatusPip status={status} />
                </div>
              ))
            )}
          </div>
        </div>

        <DropZone onUpload={handleUpload} uploading={uploading} />

        <div className="sidebar-footer">
          <span className="footer-dot" />
          <span className="footer-label">Connected · claude-3.5-sonnet</span>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────────────────────────── */}
      <main className="main">

        <header className="topbar">
          {!sidebarOpen && (
            <button className="topbar-menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
              <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
                <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </button>
          )}
          <span className="topbar-title">Research Assistant</span>
          {docEntries.length > 0 && (
            <div className="topbar-pill">
              <div className="topbar-pill-dot" />
              {docEntries.length} doc{docEntries.length !== 1 ? "s" : ""} indexed
            </div>
          )}
          <div className="topbar-actions">
            <div className="model-tag">
              <div className="model-tag-icon">
                <svg viewBox="0 0 24 24" fill="currentColor" width="8" height="8"><circle cx="12" cy="12" r="6" /></svg>
              </div>
              sonnet-3.5
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                <path d="M2 4l3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
            </div>
            <button className="topbar-btn" aria-label="Settings">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
              </svg>
            </button>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
              <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.3" />
              <path d="M7 4.5v3M7 9.5v.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
            {error}
            <button onClick={() => setError(null)} className="error-dismiss">✕</button>
          </div>
        )}

        <div className="chat-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-ring">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                </svg>
              </div>
              <h2 className="empty-title">Ask across your documents</h2>
              <p className="empty-sub">Upload PDFs in the sidebar, then ask any question. Answers are grounded in your sources with page-level citations.</p>
              {docEntries.length > 0 && (
                <div className="chips">
                  {["Summarize the key findings", "What methodology was used?", "List all recommendations", "What are the main risks?"].map((q) => (
                    <button key={q} className="chip" onClick={() => setInput(q)}>
                      {q} <span className="chip-arrow">↗</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="messages">
              {messages.map((msg) => (
                <div key={msg.id} className={`msg-row ${msg.role}`}>
                  <div className={`avatar ${msg.role === "user" ? "user-av" : "ai-av"}`}>
                    {msg.role === "user" ? "U" : "Q"}
                  </div>
                  <div className="msg-body">
                    {/* Skeleton until first token; then switch to live bubble */}
                    {msg.streaming && msg.content === "" ? (
                      <Skeleton />
                    ) : (
                      <>
                        <div className="msg-bubble">
                          {msg.content}
                          {msg.streaming && <StreamCursor />}
                        </div>
                        <div className="msg-time">
                          {msg.timestamp}
                          {!msg.streaming && msg.sources && msg.sources.length > 0 && (
                            <> · {msg.sources.length} source{msg.sources.length !== 1 ? "s" : ""} cited</>
                          )}
                        </div>
                        {/* Sources only appear after streaming ends ([DONE] received) */}
                        {!msg.streaming && msg.sources && msg.sources.length > 0 && (
                          <div className="sources">
                            <div className="sources-header">
                              <span className="sources-label">SOURCES</span>
                              <div className="sources-line" />
                            </div>
                            {msg.sources.map((src, i) => (
                              <SourceCard key={i} source={src} index={i} />
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        <div className="input-area">
          <div className={`input-wrap ${isStreaming ? "disabled" : ""}`}>
            <div className="input-top">
              <textarea
                ref={textareaRef}
                className="chat-textarea"
                placeholder={docEntries.length === 0 ? "Upload documents to get started…" : "Ask anything across your documents…"}
                value={input}
                onChange={handleTextareaInput}
                onKeyDown={handleKeyDown}
                disabled={isStreaming || docEntries.length === 0}
                rows={1}
              />
              <button
                className={`send-btn ${(!input.trim() || isStreaming) ? "dim" : ""}`}
                onClick={handleQuery}
                disabled={!input.trim() || isStreaming || docEntries.length === 0}
                aria-label="Send"
              >
                {isStreaming ? (
                  <span className="btn-spinner" />
                ) : (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" strokeLinejoin="round" strokeLinecap="round" />
                  </svg>
                )}
              </button>
            </div>
            <div className="input-bottom">
              <div className="input-bottom-left">
                <button className="input-tool-btn">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                  </svg>
                  Attach
                </button>
                <button className="input-tool-btn">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3M12 17h.01" />
                  </svg>
                  Instructions
                </button>
              </div>
              <span className="input-hint">↵ send · ⇧↵ newline</span>
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}