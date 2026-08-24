import { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useStream } from "./hooks/useStream";
import type { Source } from "./hooks/useStream";
import "./App.css";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  toolUsed?: string | null;
  sources?: Source[];
  isUploadEvent?: boolean;
  fileName?: string;       // generic: works for CSV and PDF
  fileType?: "csv" | "pdf";
}

// ─── Config ───────────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

// Poll interval while a doc is still "processing" (ms)
const STATUS_POLL_MS = 1500;

const SUGGESTED_QUERIES = [
  "What are the column names and data types?",
  "Show me summary statistics",
  "Are there any missing values?",
  "What insights can you find?",
];

const SAMPLE_FILES = [
  {
    name: "titanic.csv",
    label: "Titanic Dataset",
    description: "891 passengers · survival analysis",
    type: "text/csv",
    path: "/samples/titanic.csv",
    tag: "CSV",
  },
  {
    name: "sample_report.pdf",
    label: "AI Adoption Report",
    description: "3-page research report · 2024",
    type: "application/pdf",
    path: "/samples/sample_report.pdf",
    tag: "PDF",
  },
] as const;

// ─── SVG Icons ────────────────────────────────────────────────────────────────

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

const PlusIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const LogsIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="9" rx="1"/>
    <rect x="14" y="3" width="7" height="5" rx="1"/>
    <rect x="14" y="12" width="7" height="9" rx="1"/>
    <rect x="3" y="16" width="7" height="5" rx="1"/>
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

const UploadCloudIcon = () => (
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="16 16 12 12 8 16" />
    <line x1="12" y1="12" x2="12" y2="21" />
    <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3" />
  </svg>
);

const CsvIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <path d="M3 9h18M3 15h18M9 3v18" />
  </svg>
);

const PdfIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="9" y1="13" x2="15" y2="13" />
    <line x1="9" y1="17" x2="15" y2="17" />
  </svg>
);

const PandasIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="3" width="7" height="18" rx="1" />
    <rect x="14" y="3" width="7" height="18" rx="1" />
    <line x1="3" y1="12" x2="10" y2="12" />
    <line x1="14" y1="12" x2="21" y2="12" />
  </svg>
);

const ChartIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="20" x2="18" y2="10" />
    <line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
    <line x1="2" y1="20" x2="22" y2="20" />
  </svg>
);

const ChevronIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

const CheckCircleIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </svg>
);

// ─── Tool Badge ────────────────────────────────────────────────────────────────

function ToolBadge({ tool }: { tool: string }) {
  let icon: React.ReactNode;
  let label: string;

  switch (tool) {
    case "pandas_sandbox":
      icon = <PandasIcon />;
      label = "Pandas Sandbox";
      break;
    case "get_csv_stats":
      icon = <ChartIcon />;
      label = "CSV Stats";
      break;
    case "search_documents":
      icon = <FileIcon />;
      label = "Document Search";
      break;
    default:
      icon = <ChartIcon />;
      label = tool;
  }

  return (
    <div className="tool-badge">
      {icon}
      <span>{label}</span>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const navigate = useNavigate();

  // ── Conversation state ───────────────────────────────────────────────────
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [successToast, setSuccessToast] = useState<string | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const [isDragging, setIsDragging] = useState(false);

  // ── Upload / document state ─────────────────────────────────────────────
  const [uploading, setUploading] = useState(false);
  const [docStatus, setDocStatus] = useState<"idle" | "processing" | "ready" | "failed">("idle");
  const [docName, setDocName] = useState<string | null>(null);
  const [docType, setDocType] = useState<"csv" | "pdf" | null>(null);

  // ── Concurrent upload + query queue ────────────────────────────────────
  // Stores a question typed while doc is still processing, auto-submitted on ready
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Refs ─────────────────────────────────────────────────────────────────
  const dragCounter = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const errorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Stream hook ──────────────────────────────────────────────────────────
  const {
    streamedAnswer, streamedSources, streamedTool, isStreaming,
    startStream, reset: resetStream,
  } = useStream(API_BASE);

  // ── Auto-dismiss error / toast ────────────────────────────────────────
  useEffect(() => {
    if (!error) return;
    if (errorTimerRef.current) clearTimeout(errorTimerRef.current);
    errorTimerRef.current = setTimeout(() => setError(null), 5000);
    return () => { if (errorTimerRef.current) clearTimeout(errorTimerRef.current); };
  }, [error]);

  useEffect(() => {
    if (!successToast) return;
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setSuccessToast(null), 4000);
    return () => { if (toastTimerRef.current) clearTimeout(toastTimerRef.current); };
  }, [successToast]);

  // ── Backend health check ──────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(4000) })
      .then((r) => { if (!r.ok) throw new Error(); setBackendDown(false); })
      .catch(() => setBackendDown(true));
  }, []);

  // ── Auto-scroll ────────────────────────────────────────────────────────
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, streamedAnswer, isStreaming]);

  // ── Auto-resize textarea ──────────────────────────────────────────────
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }
  }, [inputValue]);

  // ── Commit streamed answer to messages ────────────────────────────────
  useEffect(() => {
    if (!isStreaming && streamedAnswer) {
      setMessages((prev) => [
        ...prev,
        {
          id: `ai_${Date.now()}`,
          role: "assistant",
          content: streamedAnswer,
          timestamp: new Date(),
          toolUsed: streamedTool ?? null,
          sources: streamedSources.length > 0 ? streamedSources : undefined,
        },
      ]);
    }
  }, [isStreaming, streamedAnswer, streamedTool, streamedSources]);

  // ── Document status polling ───────────────────────────────────────────
  // When a doc is in "processing", poll every STATUS_POLL_MS.
  // When it flips to "ready", stop polling and auto-fire any pendingQuery.
  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const startPolling = useCallback((targetFilename: string) => {
    stopPolling();
    pollTimerRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/documents/status`);
        if (!res.ok) return;
        const data = await res.json() as { documents: Record<string, string> };
        const status = data.documents[targetFilename];

        if (status === "ready") {
          stopPolling();
          setDocStatus("ready");
          setSuccessToast(`"${targetFilename}" ready — you can now ask questions!`);

          // Auto-fire queued query
          setPendingQuery((queued) => {
            if (queued) {
              // Fire after a tick so state is settled
              setTimeout(() => fireQuery(queued), 50);
            }
            return null;
          });
        } else if (status === "failed") {
          stopPolling();
          setDocStatus("failed");
          setError(`Processing failed for "${targetFilename}". Try re-uploading.`);
        }
      } catch {
        // silently ignore poll failures
      }
    }, STATUS_POLL_MS);
  }, [stopPolling]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup poll on unmount
  useEffect(() => () => stopPolling(), [stopPolling]);

  // ── Upload ────────────────────────────────────────────────────────────
  const uploadFile = useCallback(async (file: File) => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (ext !== "csv" && ext !== "pdf") {
      setError("Please upload a .csv or .pdf file.");
      return;
    }

    setUploading(true);
    setError(null);
    setDocStatus("processing");
    setDocName(file.name);
    setDocType(ext as "csv" | "pdf");
    stopPolling();

    try {
      const formData = new FormData();
      formData.append("files", file);

      let res: Response;
      try {
        res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      } catch {
        setBackendDown(true);
        throw new Error(`Cannot reach backend at ${API_BASE} — is the server running?`);
      }

      setBackendDown(false);

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: `Server returned ${res.status}` }));
        throw new Error(errBody.detail || `Upload failed: ${res.status}`);
      }

      // Inject upload event card into chat immediately
      const uploadMsg: Message = {
        id: `upload_${Date.now()}`,
        role: "user",
        content: "",
        timestamp: new Date(),
        isUploadEvent: true,
        fileName: file.name,
        fileType: ext as "csv" | "pdf",
      };
      setMessages((prev) => [...prev, uploadMsg]);

      // Start polling for status transition (processing → ready)
      startPolling(file.name);

    } catch (err) {
      setDocStatus("failed");
      setError((err as Error).message);
    } finally {
      setUploading(false);
    }
  }, [stopPolling, startPolling]);

  const handleFileSelect = (files: FileList | null) => {
    if (!files || !files.length) return;
    uploadFile(files[0]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Submit / fireQuery ────────────────────────────────────────────────
  const fireQuery = useCallback(async (question: string) => {
    const userMsg: Message = {
      id: `user_${Date.now()}`,
      role: "user",
      content: question,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    resetStream();

    const history = messages
      .filter((m) => !m.isUploadEvent && m.content)
      .map((m) => ({ role: m.role as "user" | "assistant", content: m.content }));

    try {
      await startStream(question, history);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [messages, resetStream, startStream]);

  const handleSubmit = async () => {
    const question = inputValue.trim();
    if (!question || isStreaming) return;

    if (docStatus === "processing") {
      // Queue the query — it fires automatically when doc becomes ready
      setPendingQuery(question);
      setInputValue("");
      // Show inline feedback via the input placeholder (handled in render)
      return;
    }

    if (docStatus !== "ready") return;
    await fireQuery(question);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // ── Conversation management ───────────────────────────────────────────

  const resetClientState = useCallback(() => {
    setMessages([]);
    setInputValue("");
    setPendingQuery(null);
    setDocStatus("idle");
    setDocName(null);
    setDocType(null);
    setError(null);
    setSuccessToast(null);
    stopPolling();
    resetStream();
  }, [stopPolling, resetStream]);

  const callServerReset = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/reset`, { method: "POST" });
    } catch {
      // Best-effort; backend may be down
    }
  }, []);

  /** New Conversation: reset state + backend, start fresh */
  const handleNewConversation = useCallback(async () => {
    if (messages.length === 0 && docStatus === "idle") return;
    if (!confirm("Start a new conversation? Current chat will be cleared.")) return;
    await callServerReset();
    resetClientState();
  }, [messages.length, docStatus, callServerReset, resetClientState]);

  /** Delete Conversation: clear chat, return to upload state */
  const handleDeleteConversation = useCallback(async () => {
    if (messages.length === 0) return;
    if (!confirm("Delete this conversation?")) return;
    await callServerReset();
    resetClientState();
  }, [messages.length, callServerReset, resetClientState]);

  const handleChangeFile = () => fileInputRef.current?.click();

  const toggleSource = (key: string) => {
    setExpandedSources((prev) => {
      const n = new Set(prev);
      n.has(key) ? n.delete(key) : n.add(key);
      return n;
    });
  };

  // ── Drag events ───────────────────────────────────────────────────────
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
    const files = e.dataTransfer.files;
    if (files && files.length > 0) uploadFile(files[0]);
  };

  // ── Derived ───────────────────────────────────────────────────────────
  const isDocReady = docStatus === "ready";
  const isDocProcessing = docStatus === "processing";

  // Send is enabled when: we have text, not currently streaming, and doc is ready
  const canSend = inputValue.trim().length > 0 && !isStreaming && isDocReady;
  // Queue is allowed when: we have text, not streaming, doc is still processing
  const canQueue = inputValue.trim().length > 0 && !isStreaming && isDocProcessing;

  const showUploadZone = docStatus === "idle";
  const showEmptyState = isDocReady && messages.length === 0 && !isStreaming;

  const docIcon = docType === "pdf" ? <PdfIcon /> : <CsvIcon />;
  const docLabel = docType === "pdf" ? "PDF" : "CSV";

  // Input placeholder logic
  const inputPlaceholder = (() => {
    if (isDocProcessing && pendingQuery)
      return `Queued: "${pendingQuery}" — waiting for document…`;
    if (isDocProcessing)
      return "Type a question — it will be sent once the document is ready";
    if (isDocReady)
      return "Ask anything about your document…";
    return "Upload a CSV or PDF to start chatting";
  })();

  // ─────────────────────────────────────────────────────────────────────
  return (
    <div
      className={`app ${isDragging ? "drag-over" : ""}`}
      onDragEnter={handleDragEnter}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragging && (
        <div className="drag-overlay">
          <div className="drag-overlay-inner">
            <UploadCloudIcon />
            <span>Drop your CSV or PDF to {isDocReady ? "replace dataset" : "get started"}</span>
          </div>
        </div>
      )}

      {/* Hidden file input — accepts CSV and PDF */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.pdf"
        style={{ display: "none" }}
        onChange={(e) => handleFileSelect(e.target.files)}
      />

      {/* ── Topbar ──────────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-logo">QM</div>
          <span className="brand-name">QueryMind</span>
        </div>

        <div className="topbar-center">
          {docName && (isDocReady || isDocProcessing) && (
            <div className={`csv-active-chip ${isDocProcessing ? "csv-active-chip--processing" : ""}`}>
              {isDocProcessing
                ? <SpinnerIcon size={12} />
                : <span className="csv-active-dot" />}
              {docIcon}
              <span className="csv-active-name" title={docName}>{docName}</span>
              {isDocProcessing && (
                <span className="csv-active-status">processing…</span>
              )}
            </div>
          )}
        </div>

        <div className="topbar-right">
          {/* Navigate to /logs page */}
          <button
            className="topbar-action-btn"
            onClick={() => navigate("/logs")}
            title="Open debug logs"
          >
            <LogsIcon />
            <span>Logs</span>
          </button>

          {/* Change file */}
          {(isDocReady || isDocProcessing) && (
            <button className="topbar-action-btn change-csv-btn" onClick={handleChangeFile} title={`Upload a different ${docLabel}`}>
              {docIcon}
              <span>Change {docLabel}</span>
            </button>
          )}

          {/* New Conversation */}
          <button
            className="topbar-action-btn"
            onClick={handleNewConversation}
            disabled={messages.length === 0 && docStatus === "idle"}
            title="New conversation"
          >
            <PlusIcon />
            <span>New</span>
          </button>

          {/* Delete Conversation */}
          <button
            className="topbar-action-btn topbar-action-btn--danger"
            onClick={handleDeleteConversation}
            disabled={messages.length === 0}
            title="Delete this conversation"
          >
            <span>+</span> New Conversation
          </button>
        </div>
      </header>

      {/* ── Backend offline banner ──────────────────────────────────────── */}
      {backendDown && (
        <div className="error-banner backend-down-banner">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" /></svg>
          Backend unreachable at <code style={{ fontFamily: "monospace", margin: "0 4px" }}>{API_BASE}</code> — make sure the FastAPI server is running.
          <button className="error-dismiss" onClick={() => setBackendDown(false)}>×</button>
        </div>
      )}

      {/* ── Error banner ──────────────────────────────────────────────── */}
      {error && (
        <div className="error-banner">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" /></svg>
          {error}
          <button className="error-dismiss" onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* ── Success toast ─────────────────────────────────────────────── */}
      {successToast && (
        <div className="success-toast">
          <CheckCircleIcon />
          <span>{successToast}</span>
        </div>
      )}

      {/* ── Chat area ─────────────────────────────────────────────────── */}
      <div className="chat-area" ref={chatAreaRef}>

        {/* Onboarding: no file uploaded yet */}
        {showUploadZone && (
          <div className="onboarding-state">
            <div className="onboarding-inner">
              <div className="empty-orb">
                <div className="orb-ring" />
                <div className="orb-core">QM</div>
              </div>
              <h1 className="empty-title">Analyse your data with AI</h1>
              <p className="empty-sub">
                Upload a CSV or PDF to unlock intelligent analysis, statistics, and natural-language queries powered by your data.
              </p>

              <div
                className={`upload-zone ${isDragging ? "upload-zone--active" : ""} ${uploading ? "upload-zone--uploading" : ""}`}
                onClick={() => !uploading && fileInputRef.current?.click()}
                onDragEnter={handleDragEnter}
                onDragOver={(e) => e.preventDefault()}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="upload-zone-icon">
                  {uploading ? <SpinnerIcon size={48} /> : <UploadCloudIcon />}
                </div>
                <div className="upload-zone-text">
                  {uploading ? (
                    <>
                      <span className="upload-zone-title">Uploading…</span>
                      <span className="upload-zone-sub">Extracting schema in the background</span>
                    </>
                  ) : (
                    <>
                      <span className="upload-zone-title">Drop your dataset here</span>
                      <span className="upload-zone-sub">or <u>browse files</u> to upload</span>
                    </>
                  )}
                </div>
                <div className="upload-zone-hint">Supports .csv and .pdf files</div>
              </div>
            </div>
          </div>
        )}

        {/* File uploaded, no messages yet */}
        {showEmptyState && (
          <div className="empty-state">
            <div className="empty-orb">
              <div className="orb-ring" />
              <div className="orb-core">QM</div>
            </div>
            <h1 className="empty-title">Ready to analyse <span className="csv-name-highlight">{docName}</span></h1>
            <p className="empty-sub">Ask anything about your data — QueryMind will compute stats and answer from your document.</p>
            <div className="chips">
              {SUGGESTED_QUERIES.map((q, i) => (
                <button key={i} className="chip" onClick={() => setInputValue(q)}>{q}</button>
              ))}
            </div>

            {/* ── Sample-file quick-load cards ── */}
            <div className="sample-files-section">
              <div className="sample-files-label">
                <FlashIcon />
                <span>Try a sample file</span>
              </div>
              <div className="sample-files-row">
                {SAMPLE_FILES.map((sf) => (
                  <button
                    key={sf.name}
                    className="sample-file-card"
                    onClick={() => loadSampleFile(sf)}
                    title={`Load ${sf.label} into pending files`}
                  >
                    <div className="sample-file-icon">
                      {sf.tag === "CSV" ? <CsvIcon /> : <PdfIcon />}
                    </div>
                    <div className="sample-file-info">
                      <span className="sample-file-name">{sf.label}</span>
                      <span className="sample-file-desc">{sf.description}</span>
                    </div>
                    <span className={`sample-file-tag tag-${sf.tag.toLowerCase()}`}>
                      {sf.tag}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Message thread — shown once there are messages OR while streaming */}
        {(messages.length > 0 || isStreaming) && !showUploadZone && (
          <div className="messages">
            {messages.map((msg) => {

              // Upload receipt card
              if (msg.isUploadEvent && msg.fileName) {
                return (
                  <div key={msg.id} className="upload-receipt-row">
                    <div className={`upload-receipt-card status-${cardStatus}`}>
                      <div className="upload-receipt-header">
                        <CheckCircleIcon />
                        <span>{msg.fileType === "pdf" ? "PDF uploaded & indexed" : "CSV uploaded & schema extracted"}</span>
                        <span className="upload-receipt-badge">
                          <span className="upload-receipt-dot" />
                          {isDocProcessing && docName === msg.fileName ? "Processing…" : "Ready"}
                        </span>
                      </div>
                      <div className="upload-receipt-files">
                        <div className="upload-receipt-file">
                          {msg.fileType === "pdf" ? <PdfIcon /> : <CsvIcon />}
                          <span>{msg.fileName}</span>
                        </div>
                      </div>
                      <div className="upload-receipt-note">
                        {msg.fileType === "pdf"
                          ? "QueryMind has indexed your document. You can ask questions while it processes."
                          : "QueryMind has indexed your dataset. You can now ask questions about the content."}
                      </div>
                    </div>
                    <div className="upload-receipt-time">
                      {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </div>
                  </div>
                );
              }

              // Pending query indicator (shown as a ghost bubble)
              if ((msg as Message & { isPending?: boolean }).isPending) {
                return (
                  <div key={msg.id} className="msg-row user">
                    <div className="msg-body">
                      <div className="msg-bubble msg-bubble--pending">
                        <SpinnerIcon size={12} />
                        <span>{msg.content}</span>
                        <span className="pending-label">queued — waiting for document…</span>
                      </div>
                    </div>
                    <div className="avatar user-av">U</div>
                  </div>
                );
              }

              // Regular message bubble
              return (
                <div key={msg.id} className={`msg-row ${msg.role}`}>
                  {msg.role === "assistant" && (
                    <div className="avatar ai-av">QM</div>
                  )}
                  <div className="msg-body">
                    <div className="msg-bubble">
                      {msg.role === "assistant" ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      ) : (
                        msg.content
                      )}
                    </div>

                    {msg.role === "assistant" && msg.toolUsed && (
                      <ToolBadge tool={msg.toolUsed} />
                    )}

                    {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                      <div className="sources-panel">
                        <button className="sources-toggle" onClick={() => toggleSource(msg.id)}>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                          {msg.sources.length} source{msg.sources.length > 1 ? "s" : ""}
                          <span className={`sources-chevron ${expandedSources.has(msg.id) ? "open" : ""}`}>
                            <ChevronIcon />
                          </span>
                        </button>
                        {expandedSources.has(msg.id) && (
                          <div className="sources-list">
                            {msg.sources.map((src, i) => (
                              <div key={i} className="source-item">
                                <div className="source-header">
                                  <FileIcon />
                                  <span className="source-name">{src.source}</span>
                                  {src.page !== "N/A" && (
                                    <span className="source-page">p.{src.page}</span>
                                  )}
                                  <span className="source-score">{(src.rerank_score * 100).toFixed(0)}%</span>
                                </div>
                                <p className="source-text">{src.text.slice(0, 220)}{src.text.length > 220 ? "…" : ""}</p>
                              </div>
                            ))}
                          </div>
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
              );
            })}

            {/* Streaming bubble */}
            {isStreaming && (
              <div className="msg-row assistant">
                <div className="avatar ai-av">QM</div>
                <div className="msg-body">
                  {streamedTool && (
                    <div className="tool-badge tool-badge--live">
                      <SpinnerIcon size={11} />
                      <span>
                        {streamedTool === "pandas_sandbox"
                          ? "Running Pandas Sandbox…"
                          : streamedTool === "get_csv_stats"
                          ? "Computing CSV Stats…"
                          : `Using ${streamedTool}…`}
                      </span>
                    </div>
                  )}
                  {streamedAnswer ? (
                    <div className="msg-bubble">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {streamedAnswer}
                      </ReactMarkdown>
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

      {/* ── Input area ──────────────────────────────────────────────────── */}
      <div className="input-area">
        <div className={`input-box ${isStreaming ? "streaming" : ""} ${docStatus === "idle" ? "input-box--locked" : ""} ${isDocProcessing ? "input-box--processing" : ""}`}>

          {/* Processing indicator inside the input box */}
          {isDocProcessing && (
            <div className="input-processing-badge" title="Document is being processed">
              <SpinnerIcon size={12} />
              <span>processing document…</span>
            </div>
          )}

          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder={inputPlaceholder}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isStreaming || (docStatus !== "ready" && docStatus !== "processing")}
            rows={1}
          />

          <button
            className={`send-btn ${(canSend || canQueue) ? "active" : ""}`}
            onClick={handleSubmit}
            disabled={!canSend && !canQueue}
            aria-label={isDocProcessing ? "Queue query" : "Send"}
            title={isDocProcessing ? "Queue query — will send when document is ready" : "Send"}
          >
            {isStreaming
              ? <SpinnerIcon size={15} />
              : isDocProcessing
              ? <span style={{ fontSize: 14 }}>⏳</span>
              : <SendIcon />}
          </button>
        </div>

        <div className="input-footer">
          {isDocReady ? (
            <span className="input-hint">Enter to send · Shift+Enter for new line · Drag &amp; drop to replace file</span>
          ) : isDocProcessing ? (
            <span className="input-hint input-hint--processing">
              {pendingQuery
                ? `Query queued: "${pendingQuery.slice(0, 40)}${pendingQuery.length > 40 ? "…" : ""}" — will fire when ready`
                : "Type a question now — it will be sent automatically when the document is ready"}
            </span>
          ) : (
            <span className="input-hint input-hint--locked">Upload a CSV or PDF file to unlock the chat</span>
          )}
        </div>
      </div>
    </div>
  );
}