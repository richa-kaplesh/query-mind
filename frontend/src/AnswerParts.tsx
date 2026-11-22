import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "./hooks/useStream";

const VISIBLE_SOURCES = 2;
const CITE_PREFIX = "#cite-";

export function citeCardId(msgId: string, n: number) {
  return `cite-${msgId}-${n}`;
}

function linkCitations(content: string, sourceCount: number) {
  if (!sourceCount) return content;
  return content.replace(/\[(\d+)\](?!\()/g, (whole, num: string) => {
    const n = Number(num);
    return n >= 1 && n <= sourceCount ? `[${n}](${CITE_PREFIX}${n})` : whole;
  });
}

export function AnswerMarkdown({
  content,
  sources,
  onCite,
}: {
  content: string;
  sources?: Source[];
  onCite?: (n: number) => void;
}) {
  const count = sources?.length ?? 0;
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children, ...rest }) => {
          if (href && href.startsWith(CITE_PREFIX)) {
            const n = Number(href.slice(CITE_PREFIX.length));
            const src = sources?.[n - 1];
            return (
              <button
                type="button"
                className="cite-chip"
                onClick={() => onCite?.(n)}
                title={src ? `${src.filename}${src.page != null ? `, p. ${src.page}` : ""}` : undefined}
                aria-label={`Show source ${n}`}
              >
                [{n}]
              </button>
            );
          }
          return (
            <a href={href} target="_blank" rel="noreferrer" {...rest}>
              {children}
            </a>
          );
        },
      }}
    >
      {linkCitations(content, count)}
    </ReactMarkdown>
  );
}

export function SourceCards({
  msgId,
  sources,
  expanded,
  highlighted,
  onToggle,
}: {
  msgId: string;
  sources: Source[];
  expanded: boolean;
  highlighted: number | null;
  onToggle: () => void;
}) {
  const shown = expanded ? sources : sources.slice(0, VISIBLE_SOURCES);
  const hidden = sources.length - VISIBLE_SOURCES;
  return (
    <div className="cite-panel">
      <ol className="cite-list">
        {shown.map((src) => (
          <li
            key={src.n}
            id={citeCardId(msgId, src.n)}
            className={`cite-card${highlighted === src.n ? " cite-card--active" : ""}`}
          >
            <div className="cite-card-head">
              <span className="cite-card-num">{src.n}</span>
              <span className="cite-card-file" title={src.filename}>{src.filename}</span>
              {src.page != null && <span className="cite-card-page">p. {src.page}</span>}
            </div>
            {src.section && <div className="cite-card-section">{src.section}</div>}
            {src.snippet && <p className="cite-card-snippet">{src.snippet}</p>}
          </li>
        ))}
      </ol>
      {hidden > 0 && (
        <button type="button" className="cite-more" onClick={onToggle} aria-expanded={expanded}>
          {expanded ? "Show fewer sources" : `Show ${hidden} more source${hidden > 1 ? "s" : ""}`}
        </button>
      )}
    </div>
  );
}

export function needsExpand(n: number) {
  return n > VISIBLE_SOURCES;
}

export function CodeRan({ code }: { code: string[] }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="code-ran" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="code-ran-summary">Code that ran</summary>
      {open && (
        <pre className="code-ran-pre">
          <code>{code.join("\n\n")}</code>
        </pre>
      )}
    </details>
  );
}
