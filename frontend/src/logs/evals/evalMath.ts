import type { Metric, RunQuestion, RunSummary } from "../types";

export const MAX_METRIC_CARDS = 4;
const OVERALL = "overall";

export function setKey(run: RunSummary): string {
  return `${run.eval_set.id}::${run.doc_type}`;
}

export function comparable(a: RunSummary, b: RunSummary): boolean {
  return a.eval_set.id === b.eval_set.id && a.doc_type === b.doc_type;
}

export function byDateAsc(a: RunSummary, b: RunSummary): number {
  return a.created_at.localeCompare(b.created_at);
}

export function byDateDesc(a: RunSummary, b: RunSummary): number {
  return b.created_at.localeCompare(a.created_at);
}

export function runsOfSet(runs: RunSummary[], run: RunSummary): RunSummary[] {
  return runs.filter(r => comparable(r, run)).sort(byDateAsc);
}

export function baselineOptions(runs: RunSummary[], run: RunSummary): RunSummary[] {
  return runs.filter(r => comparable(r, run) && r.id !== run.id).sort(byDateDesc);
}

export function defaultBaseline(runs: RunSummary[], run: RunSummary): string | null {
  const first = runsOfSet(runs, run).find(r => r.id !== run.id);
  return first ? first.id : null;
}

export interface RunGroup {
  key: string;
  label: string;
  runs: RunSummary[];
}

export function groupRuns(runs: RunSummary[]): RunGroup[] {
  const groups = new Map<string, RunGroup>();
  for (const r of [...runs].sort(byDateDesc)) {
    const k = setKey(r);
    if (!groups.has(k)) groups.set(k, { key: k, label: r.eval_set.name, runs: [] });
    groups.get(k)!.runs.push(r);
  }
  return Array.from(groups.values());
}

export function hasCounts(m: Metric): m is Metric & { passed: number; total: number } {
  return typeof m.passed === "number" && typeof m.total === "number" && m.total > 0;
}

export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatMetric(m: Metric): { main: string; sub: string | null } {
  if (hasCounts(m)) return { main: `${m.passed}/${m.total}`, sub: formatPercent(m.passed / m.total) };
  return { main: m.value.toFixed(2), sub: null };
}

export function formatMetricText(m: Metric): string {
  const f = formatMetric(m);
  return f.sub ? `${f.main} · ${f.sub}` : f.main;
}

export function formatMetricDelta(delta: number, m: Metric): string {
  if (!hasCounts(m)) {
    const r = Math.round(delta * 100) / 100;
    return `${r >= 0 ? "+" : "−"}${Math.abs(r).toFixed(2)}`;
  }
  const pts = Math.round(delta * 100);
  return `${pts >= 0 ? "+" : "−"}${Math.abs(pts)} pts`;
}

export function pickMetricCards(metrics: Metric[], baseline: Metric[] | null): { shown: Metric[]; rest: Metric[] } {
  const overall = metrics.filter(m => m.key === OVERALL);
  const others = metrics.filter(m => m.key !== OVERALL);
  if (baseline) {
    const change = (m: Metric) => {
      const b = baseline.find(x => x.key === m.key);
      return b ? Math.abs(m.value - b.value) : -1;
    };
    others.sort((a, b) => change(b) - change(a));
  }
  const ordered = [...overall, ...others];
  return { shown: ordered.slice(0, MAX_METRIC_CARDS), rest: ordered.slice(MAX_METRIC_CARDS) };
}

export function meanScore(q: RunQuestion): number | null {
  const vals = Object.values(q.scores);
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export function questionColumns(questions: RunQuestion[], metricOrder: string[]): string[] {
  const present = new Set<string>();
  questions.forEach(q => Object.keys(q.scores).forEach(k => present.add(k)));
  const ordered = metricOrder.filter(k => present.has(k));
  const extra = Array.from(present).filter(k => !metricOrder.includes(k)).sort();
  return [...ordered, ...extra];
}

export function prettyKey(k: string): string {
  const s = k.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function formatRunDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}
