import { useState, useRef, useEffect, useCallback } from "react";
import { useStream, useQuery } from "./hooks/useStream";
import "./App.css";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Document {
  id: string;
  filename: string;
  status: "ready" | "processing" | "failed";
  uploadedAt: Date;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  toolUsed?: string | null;
  attachments?: string[];
}

// ─── Config ───────────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const POLL_INTERVAL_MS = 3000;

const SUGGESTED_QUERIES = [
  "What are the key findings?",
  "Summarize the main points",
  "What are the recommendations?",
  "Explain the methodology",
];

// ─── SVG Icons ────────────────────────────────────────────────────────────────

const PaperclipIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
  </svg>
);

const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
  </svg>
);

const TrashIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
    <path d="M10 11v6M14 11v6" />
    <path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2" />
  </svg>
);

const SpinnerIcon = ({ size = 16 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
    <circle cx="12" cy="12" r="10" strokeOpacity="0.2" />
    <path d="M12 2a10 10 0 0110 10" strokeLinecap="round" className="spin-path" />
  </svg>
);

const FileIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </svg>
);

const PlusIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const ChevronIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

// ─── App Component ────────────────────────────────────────────────────────────

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const errorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [docsOpen, setDocsOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const attachBtnRef = useRef<HTMLButtonElement>(null);

  // ─── Auto-dismiss errors after 5 s ────────────────────────────────────────
  useEffect(() => {
    if (!error) return;
    if (errorTimerRef.current) clearTimeout(errorTimerRef.current);
    errorTimerRef.current = setTimeout(() => setError(null), 5000);
    return () => { if (errorTimerRef.current) clearTimeout(errorTimerRef.current); };
  }, [error]);

  // Stream hook — handles /query/stream (token-by-token SSE)
  const { streamedAnswer, isStreaming, startStream, reset: resetStream } = useStream(API_BASE);
  // Query hook — hits /query to get tool_used for the badge
  const { toolUsed, sendQuery, reset: resetQuery } = useQuery(API_BASE);

  const reset = useCallback(() => { resetStream(); resetQuery(); }, [resetStream, resetQuery]);

  // ─── Backend health check on mount ──────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/documents`, { signal: AbortSignal.timeout(4000) })
      .then((r) => { if (!r.ok) throw new Error(); setBackendDown(false); })
      .catch(() => setBackendDown(true));
  }, []);

  // ─── Poll /documents ───────────────────────────────────────────────────────
  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (!res.ok) return;
      const data: { documents: Record<string, "ready" | "processing" | "failed"> } = await res.json();

      setDocuments((prev) => {
        const serverFilenames = new Set(Object.keys(data.documents));
        const updated = prev.map((doc) => {
          const s = data.documents[doc.filename];
          return s ? { ...doc, status: s } : doc;
        });
        const localFilenames = new Set(prev.map((d) => d.filename));
        const newDocs: Document[] = [];
        for (const [filename, status] of Object.entries(data.documents)) {
          if (!localFilenames.has(filename)) {
            newDocs.push({ id: `server_${filename}`, filename, status, uploadedAt: new Date() });
          }
        }
        const filtered = updated.filter((doc) => serverFilenames.has(doc.filename));
        return [...filtered, ...newDocs];
      });
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    const hasProcessing = documents.some((d) => d.status === "processing");
    if (hasProcessing && !pollTimerRef.current) {
      pollTimerRef.current = setInterval(fetchDocuments, POLL_INTERVAL_MS);
    } else if (!hasProcessing && pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    return () => { if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null; } };
  }, [documents, fetchDocuments]);

  useEffect(() => { fetchDocuments(); }, [fetchDocuments]);

  // ─── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, streamedAnswer, isStreaming]);

  // ─── Auto-resize textarea ──────────────────────────────────────────────────
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }
  }, [inputValue]);

  // ─── Upload files ──────────────────────────────────────────────────────────
  const uploadFiles = useCallback(async (files: File[]) => {
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));

      let res: Response;
      try {
        res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      } catch {
        // fetch() itself threw — backend is unreachable
        setBackendDown(true);
        throw new Error(`Cannot reach backend at ${API_BASE} — is the server running?`);
      }

      setBackendDown(false);

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: `Server returned ${res.status}` }));
        throw new Error(errBody.detail || `Upload failed: ${res.status}`);
      }

      const data = await res.json();
      const newDocs: Document[] = (data.files || []).map((filename: string) => ({
        id: `${Date.now()}_${filename}`,
        filename,
        status: "processing" as const,
        uploadedAt: new Date(),
      }));
      setDocuments((prev) => {
        const existing = new Set(prev.map((d) => d.filename));
        return [...prev, ...newDocs.filter((d) => !existing.has(d.filename))];
      });
      setTimeout(fetchDocuments, 600);
      setPendingFiles([]);
    } catch (err) {
      setError((err as Error).message);
      throw err; // re-throw so handleSubmit stops on failure
    } finally {
      setUploading(false);
    }
  }, [fetchDocuments]);

  const handleFileSelect = (files: FileList | null) => {
    if (!files || !files.length) return;
    const arr = Array.from(files);
    setPendingFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      return [...prev, ...arr.filter((f) => !names.has(f.name))];
    });
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removePendingFile = (name: string) => {
    setPendingFiles((prev) => prev.filter((f) => f.name !== name));
  };

  // ─── Submit ────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    const question = inputValue.trim();
    if ((!question && !pendingFiles.length) || isStreaming) return;

    // Snapshot pending files before any state changes
    const filesToUpload = [...pendingFiles];
    const attachedNames = filesToUpload.map((f) => f.name);

    // Upload first — stop everything if it fails (uploadFiles re-throws)
    if (filesToUpload.length) {
      try {
        await uploadFiles(filesToUpload);
      } catch {
        // Error already shown in banner; do not proceed to query
        return;
      }
    }

    if (!question) return;

    const userMsg: Message = {
      id: `user_${Date.now()}`,
      role: "user",
      content: question,
      timestamp: new Date(),
      attachments: attachedNames.length ? attachedNames : undefined,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    reset();

    const history = messages.map((m) => ({ role: m.role as "user" | "assistant", content: m.content }));
    try {
      // Fire both in parallel: stream gives tokens, query gives tool_used
      await Promise.all([
        startStream(question, history),
        sendQuery(question, history),
      ]);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  // ─── Commit streamed answer ────────────────────────────────────────────────
  useEffect(() => {
    if (!isStreaming && streamedAnswer) {
      setMessages((prev) => [
        ...prev,
        {
          id: `ai_${Date.now()}`,
          role: "assistant",
          content: streamedAnswer,
          timestamp: new Date(),
          toolUsed: toolUsed ?? null,
        },
      ]);
    }
  }, [isStreaming, streamedAnswer, toolUsed]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // ─── Delete document ───────────────────────────────────────────────────────
  const handleDeleteDoc = async (doc: Document) => {
    if (!confirm(`Remove "${doc.filename}"?`)) return;
    try {
      const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(doc.filename)}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Delete failed");
      setDocuments((prev) => prev.filter((d) => d.id !== doc.id));
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const toggleSource = (key: string) => {
    setExpandedSources((prev) => {
      const n = new Set(prev);
      n.has(key) ? n.delete(key) : n.add(key);
      return n;
    });
  };

  const handleClearChat = () => {
    if (messages.length === 0) return;
    if (confirm("Clear conversation?")) { setMessages([]); reset(); }
  };

  // ─── Drag-over whole page ──────────────────────────────────────────────────
  const [isDragging, setIsDragging] = useState(false);
  const dragCounter = useRef(0);

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current++;
    setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) setIsDragging(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  // ─── Derived ──────────────────────────────────────────────────────────────
  const canSend = (inputValue.trim().length > 0 || pendingFiles.length > 0) && !isStreaming;
  const processingDocs = documents.filter((d) => d.status === "processing");

  return (
    <div
      className={`app ${isDragging ? "drag-over" : ""}`}
      onDragEnter={handleDragEnter}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* ─── Drag overlay ─────────────────────────────────────────────────── */}
      {isDragging && (
        <div className="drag-overlay">
          <div className="drag-overlay-inner">
            <PaperclipIcon />
            <span>Drop files to attach</span>
          </div>
        </div>
      )}

      {/* ─── Hidden file input ─────────────────────────────────────────────── */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.txt,.md,.doc,.docx"
        style={{ display: "none" }}
        onChange={(e) => handleFileSelect(e.target.files)}
      />

      {/* ─── Topbar ───────────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-logo">QM</div>
          <span className="brand-name">QueryMind</span>
        </div>

        <div className="topbar-center">
          {processingDocs.length > 0 && (
            <div className="processing-pill">
              <span className="processing-dot" />
              Indexing {processingDocs.length} file{processingDocs.length > 1 ? "s" : ""}…
            </div>
          )}
        </div>

        <div className="topbar-right">
          {documents.length > 0 && (
            <button
              className={`docs-toggle ${docsOpen ? "active" : ""}`}
              onClick={() => setDocsOpen((o) => !o)}
              title="Uploaded documents"
            >
              <FileIcon />
              <span>{documents.length}</span>
            </button>
          )}
          <button
            className="topbar-action-btn"
            onClick={handleClearChat}
            disabled={messages.length === 0}
            title="Clear conversation"
          >
            <TrashIcon />
          </button>
        </div>
      </header>

      {/* ─── Docs tray ────────────────────────────────────────────────────── */}
      {docsOpen && documents.length > 0 && (
        <div className="docs-tray">
          {documents.map((doc) => (
            <div key={doc.id} className={`doc-chip ${doc.status}`}>
              <div className="doc-chip-pip" />
              <FileIcon />
              <span className="doc-chip-name" title={doc.filename}>{doc.filename}</span>
              <button
                className="doc-chip-remove"
                onClick={() => handleDeleteDoc(doc)}
                title="Remove"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ─── Backend offline banner (persistent, no auto-dismiss) ─────────── */}
      {backendDown && (
        <div className="error-banner backend-down-banner">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
          Backend unreachable at <code style={{fontFamily:"monospace", margin:"0 4px"}}>{API_BASE}</code> — make sure the FastAPI server is running.
          <button className="error-dismiss" onClick={() => setBackendDown(false)}>×</button>
        </div>
      )}

      {/* ─── Error banner (auto-dismisses after 5 s) ──────────────────────── */}
      {error && (
        <div className="error-banner">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
          {error}
          <button className="error-dismiss" onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* ─── Chat area ────────────────────────────────────────────────────── */}
      <div className="chat-area" ref={chatAreaRef}>
        {messages.length === 0 && !isStreaming ? (
          <div className="empty-state">
            <div className="empty-orb">
              <div className="orb-ring" />
              <div className="orb-core">QM</div>
            </div>
            <h1 className="empty-title">What can I help you with?</h1>
            <p className="empty-sub">
              Ask anything, or attach a document to get grounded answers with sources.
            </p>
            <div className="chips">
              {SUGGESTED_QUERIES.map((q, i) => (
                <button key={i} className="chip" onClick={() => setInputValue(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`msg-row ${msg.role}`}>
                {msg.role === "assistant" && (
                  <div className="avatar ai-av">QM</div>
                )}
                <div className="msg-body">
                  {/* Attachments shown above the bubble */}
                  {msg.attachments && msg.attachments.length > 0 && (
                    <div className="msg-attachments">
                      {msg.attachments.map((name) => (
                        <div key={name} className="msg-attach-chip">
                          <FileIcon />
                          <span>{name}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="msg-bubble">{msg.content}</div>

                  {/* Tool badge — shows which backend tool was selected */}
                  {msg.role === "assistant" && msg.toolUsed && (
                    <div className="tool-badge">
                      {msg.toolUsed === "search_documents" ? (
                        <><FileIcon /> Document Search</>
                      ) : msg.toolUsed === "search_web" ? (
                        <>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 010 20M12 2a15.3 15.3 0 000 20"/></svg>
                          Web Search
                        </>
                      ) : msg.toolUsed === "get_csv_stats" ? (
                        <>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/></svg>
                          CSV Analysis
                        </>
                      ) : (
                        msg.toolUsed
                      )}
                    </div>
                  )}

                  <div className="msg-time">
                    {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </div>
                </div>
                {msg.role === "user" && (
                  <div className="avatar user-av">U</div>
                )}
              </div>
            ))}

            {/* Streaming */}
            {isStreaming && (
              <div className="msg-row assistant">
                <div className="avatar ai-av">QM</div>
                <div className="msg-body">
                  {streamedAnswer ? (
                    <div className="msg-bubble">
                      {streamedAnswer}
                      <span className="stream-cursor" />
                    </div>
                  ) : (
                    <div className="thinking">
                      <span className="thinking-dot" />
                      <span className="thinking-dot" />
                      <span className="thinking-dot" />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ─── Input area ───────────────────────────────────────────────────── */}
      <div className="input-area">
        {/* Pending file chips */}
        {pendingFiles.length > 0 && (
          <div className="pending-files">
            {pendingFiles.map((f) => (
              <div key={f.name} className="pending-chip">
                {uploading ? <SpinnerIcon size={12} /> : <FileIcon />}
                <span>{f.name}</span>
                {!uploading && (
                  <button className="pending-chip-remove" onClick={() => removePendingFile(f.name)}>×</button>
                )}
              </div>
            ))}
          </div>
        )}

        <div className={`input-box ${isStreaming ? "streaming" : ""}`}>
          <button
            ref={attachBtnRef}
            className="attach-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Attach files (PDF, TXT, MD, DOC)"
            disabled={isStreaming}
          >
            <PaperclipIcon />
          </button>

          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder="Message QueryMind…"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isStreaming}
            rows={1}
          />

          <button
            className={`send-btn ${canSend ? "active" : ""}`}
            onClick={handleSubmit}
            disabled={!canSend}
            aria-label="Send"
          >
            {isStreaming ? <SpinnerIcon size={15} /> : <SendIcon />}
          </button>
        </div>

        <div className="input-footer">
          <span className="input-hint">Enter to send · Shift+Enter for new line · Attach PDF, TXT, MD, DOC</span>
        </div>
      </div>
    </div>
  );
}