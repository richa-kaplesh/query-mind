import { useState, useRef, useEffect } from "react";
import { useStream, type Source } from "./hooks/useStream";
import "./App.css";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Document {
  id: string;
  filename: string;
  status: "ready" | "processing";
  uploadedAt: Date;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: Source[];
}

// ─── Config ───────────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const SUGGESTED_QUERIES = [
  "What are the key findings?",
  "Summarize the main points",
  "What are the recommendations?",
  "Explain the methodology",
];

// ─── App Component ────────────────────────────────────────────────────────────

export default function App() {
  // State
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set());

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Custom hook for streaming
  const { streamedAnswer, sources, isStreaming, startStream, reset } = useStream(API_BASE);

  // ─── Auto-scroll chat ─────────────────────────────────────────────────────
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, streamedAnswer]);

  // ─── Auto-resize textarea ─────────────────────────────────────────────────
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [inputValue]);

  // ─── Document Upload ──────────────────────────────────────────────────────
  const handleFileSelect = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      Array.from(files).forEach((file) => {
        formData.append("files", file);
      });

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(errData.detail || `Upload failed: ${res.status}`);
      }

      const data = await res.json();
      
      // Add uploaded documents to state
      const newDocs: Document[] = (data.files || []).map((filename: string) => ({
        id: `${Date.now()}_${filename}`,
        filename,
        status: "processing" as const,
        uploadedAt: new Date(),
      }));

      setDocuments((prev) => [...prev, ...newDocs]);

      // Simulate processing completion (in real app, poll backend)
      setTimeout(() => {
        setDocuments((prev) =>
          prev.map((doc) =>
            newDocs.find((nd) => nd.id === doc.id)
              ? { ...doc, status: "ready" as const }
              : doc
          )
        );
      }, 2000);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  // ─── Query Submission ─────────────────────────────────────────────────────
  const handleSubmit = async () => {
    const question = inputValue.trim();
    if (!question || isStreaming) return;

    // Add user message
    const userMsg: Message = {
      id: `user_${Date.now()}`,
      role: "user",
      content: question,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    reset();

    try {
      await startStream(question);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  // ─── Add AI response when streaming completes ─────────────────────────────
  useEffect(() => {
    if (!isStreaming && streamedAnswer) {
      const aiMsg: Message = {
        id: `ai_${Date.now()}`,
        role: "assistant",
        content: streamedAnswer,
        timestamp: new Date(),
        sources: sources.length > 0 ? sources : undefined,
      };
      setMessages((prev) => [...prev, aiMsg]);
    }
  }, [isStreaming, streamedAnswer, sources]);

  // ─── Keyboard handling ────────────────────────────────────────────────────
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // ─── Drag & Drop ──────────────────────────────────────────────────────────
  const [isDragging, setIsDragging] = useState(false);

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.currentTarget === e.target) setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  // ─── Clear conversation ───────────────────────────────────────────────────
  const handleClearChat = () => {
    if (confirm("Clear conversation?")) {
      setMessages([]);
      reset();
    }
  };

  // ─── Delete document ──────────────────────────────────────────────────────
  const handleDeleteDoc = async (docId: string) => {
    const doc = documents.find((d) => d.id === docId);
    if (!doc) return;

    if (!confirm(`Delete ${doc.filename}?`)) return;

    try {
      const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(doc.filename)}`, {
        method: "DELETE",
      });

      if (!res.ok) throw new Error("Failed to delete document");

      setDocuments((prev) => prev.filter((d) => d.id !== docId));
    } catch (err) {
      setError((err as Error).message);
    }
  };

  // ─── Toggle source expansion ──────────────────────────────────────────────
  const toggleSourceExpansion = (index: number) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  const canSend = inputValue.trim().length > 0 && !isStreaming;
  const hasDocuments = documents.length > 0;

  return (
    <div className="app">
      {/* ─── Sidebar ─────────────────────────────────────────────────────── */}
      <aside className={`sidebar ${sidebarOpen ? "" : "closed"}`}>
        <div className="sidebar-logo">
          <div className="logo-icon">QM</div>
          <div>
            <div className="logo-name">QueryMind</div>
            <div className="logo-sub">v2.0</div>
          </div>
          <button
            className="sidebar-toggle-btn"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            ←
          </button>
        </div>

        <div className="sidebar-section">
          <div className="section-title">
            Documents
            {documents.length > 0 && (
              <span className="doc-count-badge">{documents.length}</span>
            )}
          </div>

          {documents.length === 0 ? (
            <div className="doc-empty">No documents yet</div>
          ) : (
            <div className="doc-list">
              {documents.map((doc) => (
                <div key={doc.id} className="doc-item">
                  <div className="doc-icon">📄</div>
                  <div className="doc-name" title={doc.filename}>
                    {doc.filename}
                  </div>
                  <div className={`status-pip ${doc.status}`} />
                  <button
                    className="doc-delete-btn"
                    onClick={() => handleDeleteDoc(doc.id)}
                    aria-label="Delete document"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div
          className={`upload-zone ${isDragging ? "drag-over" : ""} ${uploading ? "uploading" : ""}`}
          onClick={() => !uploading && fileInputRef.current?.click()}
          onDragEnter={handleDragEnter}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md,.doc,.docx"
            style={{ display: "none" }}
            onChange={(e) => handleFileSelect(e.target.files)}
          />
          <div className="upload-zone-icon">
            {uploading ? <div className="dz-spinner" /> : "📎"}
          </div>
          <div className="upload-zone-label">
            {uploading ? "Uploading..." : <><span>Click</span> or drag files</>}
          </div>
          <div className="upload-zone-hint">PDF, TXT, MD, DOC</div>
        </div>

        <div className="sidebar-footer">
          <div className="footer-dot" />
          <div className="footer-label">Connected</div>
        </div>
      </aside>

      {/* ─── Main ────────────────────────────────────────────────────────── */}
      <main className="main">
        {/* Topbar */}
        <div className="topbar">
          {!sidebarOpen && (
            <button
              className="topbar-menu-btn"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
            >
              ☰
            </button>
          )}
          <div className="topbar-title">Query Assistant</div>
          {hasDocuments && (
            <div className="topbar-pill">
              <div className="topbar-pill-dot" />
              {documents.length} doc{documents.length !== 1 ? "s" : ""}
            </div>
          )}
          <div className="topbar-actions">
            <div className="model-tag" title="AI Model">
              <div className="model-tag-icon">✨</div>
              Claude Sonnet
            </div>
            <button
              className="topbar-btn"
              onClick={handleClearChat}
              disabled={messages.length === 0}
              aria-label="Clear conversation"
              title="Clear conversation"
            >
              🗑️
            </button>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="error-banner">
            ⚠️ {error}
            <button className="error-dismiss" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        )}

        {/* Chat area */}
        <div className="chat-area" ref={chatAreaRef}>
          {messages.length === 0 && !isStreaming ? (
            <div className="empty-state">
              <div className="empty-ring">💬</div>
              <div className="empty-title">Ready to answer your questions</div>
              <div className="empty-sub">
                Upload documents and ask questions about their content. Get answers
                with source references.
              </div>
              {hasDocuments && (
                <div className="chips">
                  {SUGGESTED_QUERIES.map((query, i) => (
                    <button
                      key={i}
                      className="chip"
                      onClick={() => setInputValue(query)}
                    >
                      {query}
                      <span className="chip-arrow">→</span>
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
                    {msg.role === "user" ? "U" : "QM"}
                  </div>
                  <div className="msg-body">
                    <div className="msg-bubble">{msg.content}</div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="sources">
                        <div className="sources-header">
                          <div className="sources-label">SOURCES</div>
                          <div className="sources-line" />
                        </div>
                        {msg.sources.map((source, idx) => {
                          const isExpanded = expandedSources.has(idx);
                          return (
                            <div
                              key={idx}
                              className="source-item"
                              onClick={() => toggleSourceExpansion(idx)}
                            >
                              <div className="source-item-left">
                                <div className="source-row1">
                                  <div className="source-name">{source.source}</div>
                                  <div className="source-pg">p.{source.page}</div>
                                </div>
                                <div className={`source-excerpt ${isExpanded ? "expanded" : ""}`}>
                                  {source.text}
                                </div>
                              </div>
                              <div className="source-score">
                                <div className="score-num">
                                  {(source.rerank_score * 100).toFixed(0)}%
                                </div>
                                <div className="score-bar">
                                  <div
                                    className="score-fill"
                                    style={{ width: `${source.rerank_score * 100}%` }}
                                  />
                                </div>
                              </div>
                              <div className={`source-chevron ${isExpanded ? "open" : ""}`}>
                                ⌄
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                    <div className="msg-time">
                      {msg.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </div>
                  </div>
                </div>
              ))}

              {/* Streaming message */}
              {isStreaming && (
                <div className="msg-row ai">
                  <div className="avatar ai-av">QM</div>
                  <div className="msg-body">
                    {streamedAnswer ? (
                      <div className="msg-bubble">
                        {streamedAnswer}
                        <span className="stream-cursor" />
                      </div>
                    ) : (
                      <div className="skeleton">
                        <div className="skel-line" style={{ width: "90%" }} />
                        <div className="skel-line" style={{ width: "75%" }} />
                        <div className="skel-line" style={{ width: "85%" }} />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="input-area">
          <div className={`input-wrap ${!hasDocuments ? "disabled" : ""}`}>
            <div className="input-top">
              <textarea
                ref={textareaRef}
                className="chat-textarea"
                placeholder={
                  hasDocuments
                    ? "Ask a question about your documents..."
                    : "Upload documents to start"
                }
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={!hasDocuments || isStreaming}
                rows={1}
              />
              <button
                className={`send-btn ${!canSend ? "dim" : ""}`}
                onClick={handleSubmit}
                disabled={!canSend}
                aria-label="Send message"
              >
                {isStreaming ? <div className="btn-spinner" /> : "→"}
              </button>
            </div>
            <div className="input-bottom">
              <div className="input-bottom-left">
                <div className="input-hint">
                  {hasDocuments
                    ? "Enter to send · Shift+Enter for new line"
                    : "Upload documents first"}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}