import type { RunSummary } from "../types";
import { baselineOptions, formatRunDate, groupRuns } from "./evalMath";

function optionLabel(r: RunSummary): string {
  return `${r.name} — ${formatRunDate(r.created_at)} (${r.id})`;
}

export default function RunPicker({ runs, run, baselineId, onRun, onBaseline }: {
  runs: RunSummary[];
  run: RunSummary;
  baselineId: string | null;
  onRun: (id: string) => void;
  onBaseline: (id: string) => void;
}) {
  const groups = groupRuns(runs);
  const options = baselineOptions(runs, run);
  return (
    <div className="lg-picker">
      <label className="lg-picker-field">
        <span className="lg-picker-label">Run</span>
        <select className="lg-select" value={run.id} onChange={e => onRun(e.target.value)}>
          {groups.map(g => (
            <optgroup key={g.key} label={g.label}>
              {g.runs.map(r => <option key={r.id} value={r.id}>{optionLabel(r)}</option>)}
            </optgroup>
          ))}
        </select>
      </label>
      {options.length > 0 && baselineId && (
        <label className="lg-picker-field">
          <span className="lg-picker-label">vs baseline</span>
          <select
            className="lg-select"
            value={baselineId}
            onChange={e => onBaseline(e.target.value)}
          >
            <optgroup label={run.eval_set.name}>
              {options.map(r => <option key={r.id} value={r.id}>{optionLabel(r)}</option>)}
            </optgroup>
          </select>
        </label>
      )}
    </div>
  );
}
