import "./EmptyState.css";

export interface Sample {
  file: string;
  path: string;
  mime: string;
  type: "csv" | "pdf";
  question: string;
  tag: string;
  primary: boolean;
}

export const SAMPLES: Sample[] = [
  {
    file: "titanic.csv",
    path: "/samples/titanic.csv",
    mime: "text/csv",
    type: "csv",
    question: "Which passenger class had the highest survival rate?",
    tag: "pandas",
    primary: true,
  },
  {
    file: "report.pdf",
    path: "/samples/report.pdf",
    mime: "application/pdf",
    type: "pdf",
    question: "What are the top barriers to AI adoption?",
    tag: "cited pages",
    primary: false,
  },
];

const CsvIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <path d="M3 9h18M3 15h18M9 3v18" />
  </svg>
);

const PdfIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="9" y1="13" x2="15" y2="13" />
    <line x1="9" y1="17" x2="15" y2="17" />
  </svg>
);

const SearchIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="7" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const RankIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="4" y1="6" x2="20" y2="6" />
    <line x1="4" y1="12" x2="15" y2="12" />
    <line x1="4" y1="18" x2="10" y2="18" />
  </svg>
);

const SourceIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 19.5A2.5 2.5 0 016.5 17H20" />
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" />
  </svg>
);

interface EmptyStateProps {
  busy: boolean;
  loadingFile: string | null;
  onRunSample: (sample: Sample) => void;
  onUpload: () => void;
  suggestions: string[] | null;
  onAsk: (question: string) => void;
}

export default function EmptyState({ busy, loadingFile, onRunSample, onUpload, suggestions, onAsk }: EmptyStateProps) {
  return (
    <div className="es">
      <h1 className="es-headline">Ask your documents. Check every answer.</h1>
      <p className="es-sub">PDF answers cite the exact page. CSV questions run real pandas code in a sandbox.</p>

      <div className="es-samples">
        <span className="es-label">Try a sample</span>
        <div className="es-grid">
          {SAMPLES.map((s) => (
            <div key={s.file} className={`es-card ${s.primary ? "es-card--primary" : ""}`}>
              <div className="es-card-head">
                <span className="es-card-icon">{s.type === "pdf" ? <PdfIcon /> : <CsvIcon />}</span>
                <span className="es-card-file" title={s.file}>{s.file}</span>
                <span className="es-tag">{s.tag}</span>
              </div>
              <p className="es-card-q">{s.question}</p>
              <button
                type="button"
                className={`es-btn ${s.primary ? "es-btn--primary" : "es-btn--secondary"}`}
                onClick={() => onRunSample(s)}
                disabled={busy}
              >
                {loadingFile === s.file ? "Loading…" : "Run this question"}
              </button>
            </div>
          ))}
        </div>
      </div>

      <button type="button" className="es-upload-link" onClick={onUpload} disabled={busy}>
        or upload your own CSV or PDF (up to 50 MB)
      </button>

      {suggestions && (
        <div className="suggested-chips">
          <span className="suggested-label">Try asking:</span>
          <div className="chips">
            {suggestions.map((q) => (
              <button key={q} className="chip" onClick={() => onAsk(q)}>{q}</button>
            ))}
          </div>
        </div>
      )}

      <div className="es-features">
        <span className="es-feature"><SearchIcon />Hybrid search (BM25 + vectors)</span>
        <span className="es-feature"><RankIcon />Reranked results</span>
        <span className="es-feature"><SourceIcon />Answers with sources</span>
      </div>
    </div>
  );
}
