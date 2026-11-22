import type { RunQuestion, RunSummary } from "../types";

export function setKey(run: RunSummary): string {
  return `${run.eval_set.id}::${run.doc_type}`;
}

export function comparable(a: RunSummary, b: RunSummary): boolean {
  return a.eval_set.id === b.eval_set.id && a.doc_type === b.doc_type;
}

export function byDateAsc(a: RunSummary, b: RunSummary): number {
  return a.created_at.localeCompare(b.created_at);
}

export function runsOfSet(runs: RunSummary[], run: RunSummary): RunSummary[] {
  return runs.filter(r => comparable(r, run)).sort(byDateAsc);
}

export function defaultBaseline(runs: RunSummary[], run: RunSummary): string | null {
  const first = runsOfSet(runs, run)[0];
  return first && first.id !== run.id ? first.id : null;
}

export interface RunGroup {
  key: string;
  label: string;
  runs: RunSummary[];
}

export function groupRuns(runs: RunSummary[]): RunGroup[] {
  const groups = new Map<string, RunGroup>();
  for (const r of runs) {
    const k = setKey(r);
    if (!groups.has(k)) groups.set(k, { key: k, label: `${r.eval_set.name} · ${r.doc_type.toUpperCase()}`, runs: [] });
    groups.get(k)!.runs.push(r);
  }
  return Array.from(groups.values());
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
