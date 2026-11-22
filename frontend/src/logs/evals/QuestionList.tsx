import { useMemo, useState } from "react";
import type { Metric, RunQuestion } from "../types";
import { deltaTone, formatDelta, isBelowThreshold } from "../thresholds";
import { meanScore, prettyKey, questionColumns } from "./evalMath";

const MEAN = "__mean";
const PASSED = "passed";

function PassCell({ passed }: { passed: boolean }) {
  return (
    <span className={`lg-pass lg-pass--${passed ? "yes" : "no"}`}>
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        {passed ? <polyline points="20 6 9 17 4 12" /> : <path d="M18 6L6 18M6 6l12 12" />}
      </svg>
      {passed ? "Pass" : "Fail"}
    </span>
  );
}

export default function QuestionList({ questions, baselineQuestions, metrics }: {
  questions: RunQuestion[];
  baselineQuestions: RunQuestion[] | null;
  metrics: Metric[];
}) {
  const [sortBy, setSortBy] = useState<string>(MEAN);
  const columns = useMemo(() => questionColumns(questions, metrics.map(m => m.key)), [questions, metrics]);
  const labelOf = (k: string) => metrics.find(m => m.key === k)?.label ?? prettyKey(k);
  const higherOf = (k: string) => metrics.find(m => m.key === k)?.higher_is_better ?? true;
  const passMode = columns.includes(PASSED);
  const baseById = useMemo(
    () => new Map((baselineQuestions ?? []).map(q => [q.id, q])),
    [baselineQuestions],
  );

  const rows = useMemo(() => {
    const keyOf = (q: RunQuestion) => (sortBy === MEAN ? meanScore(q) : q.scores[sortBy] ?? null);
    return [...questions].sort((a, b) => {
      const va = keyOf(a);
      const vb = keyOf(b);
      if (va === null && vb === null) return a.id.localeCompare(b.id);
      if (va === null) return 1;
      if (vb === null) return -1;
      return va - vb || a.id.localeCompare(b.id);
    });
  }, [questions, sortBy]);

  if (questions.length === 0) {
    return <p className="lg-muted">This run has no per-question scores.</p>;
  }

  return (
    <div className="lg-card lg-questions">
      <div className="lg-questions-head">
        <h2 className="lg-section-title">Questions <span className="lg-muted">weakest first</span></h2>
        <label className="lg-picker-field lg-picker-field--inline">
          <span className="lg-picker-label">Sort by</span>
          <select className="lg-select" value={sortBy} onChange={e => setSortBy(e.target.value)}>
            <option value={MEAN}>Mean score</option>
            {columns.map(k => <option key={k} value={k}>{labelOf(k)}</option>)}
          </select>
        </label>
      </div>
      <div className="lg-table-wrap">
        <table className="lg-table">
          <thead>
            <tr>
              <th scope="col" className="lg-col-q">Question</th>
              {columns.map(k => <th key={k} scope="col" className="lg-num">{labelOf(k)}</th>)}
              {baselineQuestions && <th scope="col" className="lg-num">{passMode ? "Δ" : "Δ mean"}</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map(q => {
              const base = baseById.get(q.id);
              const m = meanScore(q);
              const bm = base ? meanScore(base) : null;
              const d = m !== null && bm !== null ? m - bm : null;
              return (
                <tr key={q.id}>
                  <td className="lg-col-q">
                    <span className="lg-qid">{q.id}</span>
                    <span className="lg-qtext" title={q.text}>{q.text}</span>
                  </td>
                  {columns.map(k => {
                    const v = q.scores[k];
                    if (v === undefined) return <td key={k} className="lg-num lg-muted">—</td>;
                    if (k === PASSED) {
                      return <td key={k} className="lg-num"><PassCell passed={v >= 1} /></td>;
                    }
                    const low = isBelowThreshold(v, higherOf(k));
                    return (
                      <td key={k} className={`lg-num ${low ? "lg-low" : ""}`}>
                        {v.toFixed(2)}
                      </td>
                    );
                  })}
                  {baselineQuestions && passMode && (
                    <td className="lg-num">
                      {(() => {
                        const now = q.scores[PASSED];
                        const before = base?.scores[PASSED];
                        if (now === undefined || before === undefined || (now >= 1) === (before >= 1)) return null;
                        return now >= 1
                          ? <span className="lg-delta lg-delta--success">new pass</span>
                          : <span className="lg-delta lg-delta--danger">new fail</span>;
                      })()}
                    </td>
                  )}
                  {baselineQuestions && !passMode && (
                    <td className="lg-num">
                      {d === null ? (
                        <span className="lg-muted">—</span>
                      ) : (
                        <span className={`lg-delta lg-delta--${deltaTone(d, true)}`}>{formatDelta(d)}</span>
                      )}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
