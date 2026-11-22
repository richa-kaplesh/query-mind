import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useJson } from "../api";
import type { RunDetail, RunSummary } from "../types";
import { EmptyState, ErrorState, Skeleton } from "../StateViews";
import { comparable, defaultBaseline, pickMetricCards, runsOfSet } from "./evalMath";
import RunPicker from "./RunPicker";
import MetricCard from "./MetricCard";
import FixCard, { ChangeTags } from "./FixCard";
import QuestionList from "./QuestionList";

export default function EvaluationsTab() {
  const list = useJson<{ runs: RunSummary[] }>("/api/evals/runs");
  const [params, setParams] = useSearchParams();
  const [showAllMetrics, setShowAllMetrics] = useState(false);
  const runs = list.data?.runs ?? [];

  const run = runs.find(r => r.id === params.get("run")) ?? runs[0] ?? null;
  const baseParam = params.get("base");
  const baselineId = useMemo(() => {
    if (!run) return null;
    const picked = runs.find(r => r.id === baseParam);
    if (picked && picked.id !== run.id && comparable(picked, run)) return picked.id;
    return defaultBaseline(runs, run);
  }, [runs, run, baseParam]);
  const baseline = runs.find(r => r.id === baselineId) ?? null;

  const detail = useJson<RunDetail>(run ? `/api/evals/runs/${run.id}` : null);
  const baseDetail = useJson<RunDetail>(baseline ? `/api/evals/runs/${baseline.id}` : null);

  function update(next: Record<string, string | null>) {
    const p = new URLSearchParams(params);
    Object.entries(next).forEach(([k, v]) => (v === null ? p.delete(k) : p.set(k, v)));
    setParams(p, { replace: true });
  }

  if (list.loading) return <div className="lg-evals"><Skeleton cards={4} rows={4} /></div>;
  if (list.error) return <div className="lg-evals"><ErrorState message={list.error} onRetry={list.reload} /></div>;
  if (!run) {
    return (
      <div className="lg-evals">
        <EmptyState title="No evaluation runs yet">
          <p className="lg-state-text">Run an eval from the backend folder; each run writes one file to eval/runs/.</p>
          <code className="lg-code">cd backend &amp;&amp; python eval/run_pdf_eval.py</code>
        </EmptyState>
      </div>
    );
  }

  const setRuns = runsOfSet(runs, run);
  const selectedIdx = setRuns.findIndex(r => r.id === run.id);
  const showDeltas = baseline !== null && setRuns.length > 1;
  const cards = pickMetricCards(run.metrics, showDeltas ? baseline!.metrics : null);
  const visibleMetrics = showAllMetrics ? [...cards.shown, ...cards.rest] : cards.shown;
  const d = detail.data && detail.data.id === run.id ? detail.data : null;
  const bd = baseDetail.data && baseline && baseDetail.data.id === baseline.id ? baseDetail.data : null;

  return (
    <div className="lg-evals">
      <RunPicker
        runs={runs}
        run={run}
        baselineId={baselineId}
        onRun={id => { setShowAllMetrics(false); update({ run: id, base: null }); }}
        onBaseline={id => update({ base: id })}
      />

      <p className="lg-run-meta">
        {run.eval_set.name} · {run.doc_type.toUpperCase()} · {run.eval_set.question_count} questions
        {!showDeltas && <span className="lg-muted"> · only one run in this set</span>}
      </p>

      <div className="lg-metrics lg-metrics--evals">
        {visibleMetrics.map(m => {
          const baseMetric = showDeltas ? baseline!.metrics.find(b => b.key === m.key) ?? null : null;
          const series = setRuns.map(r => r.metrics.find(x => x.key === m.key)?.value ?? NaN);
          const clean = series.every(v => !Number.isNaN(v));
          return (
            <MetricCard
              key={m.key}
              metric={m}
              baseline={baseMetric}
              series={clean ? series : []}
              selectedIdx={selectedIdx}
            />
          );
        })}
      </div>
      {cards.rest.length > 0 && (
        <button
          type="button"
          className="lg-btn lg-btn--ghost lg-metrics-more"
          onClick={() => setShowAllMetrics(v => !v)}
          aria-expanded={showAllMetrics}
        >
          {showAllMetrics ? "Show fewer" : `+${cards.rest.length} more`}
        </button>
      )}

      {detail.loading && <Skeleton rows={5} />}
      {detail.error && <ErrorState message={detail.error} onRetry={detail.reload} />}
      {d && (
        <>
          {d.notes ? (
            <FixCard notes={d.notes} changes={d.config_changes} />
          ) : (
            d.config_changes.length > 0 && <ChangeTags changes={d.config_changes} />
          )}
          {baseDetail.error && (
            <ErrorState message={`Baseline questions: ${baseDetail.error}`} onRetry={baseDetail.reload} />
          )}
          <QuestionList
            questions={d.questions}
            baselineQuestions={showDeltas ? bd?.questions ?? null : null}
            metrics={run.metrics}
          />
        </>
      )}
    </div>
  );
}
