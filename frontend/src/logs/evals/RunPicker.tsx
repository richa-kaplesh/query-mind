import type { RunSummary } from "../types";
import { comparable, formatRunDate, groupRuns } from "./evalMath";

function optionLabel(r: RunSummary): string {
  return `${r.name} — ${formatRunDate(r.created_at)} (${r.id})`;
}

export default function RunPicker({ runs, run, baselineId, onRun, onBaseline }: {
  runs: RunSummary[];
  run: RunSummary;
  baselineId: string | null;
  onRun: (id: string) => void;
  onBaseline: (id: string | null) => void;
}) {
  const groups = groupRuns(runs);
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
      <label className="lg-picker-field">
        <span className="lg-picker-label">vs baseline</span>
        <select
          className="lg-select"
          value={baselineId ?? ""}
          onChange={e => onBaseline(e.target.value || null)}
        >
          <option value="">No baseline</option>
          {groups.map(g => {
            const sameSet = comparable(g.runs[0], run);
            return (
              <optgroup key={g.key} label={sameSet ? g.label : `${g.label} — different eval set`}>
                {g.runs.map(r => (
                  <option key={r.id} value={r.id} disabled={!sameSet || r.id === run.id}>
                    {optionLabel(r)}
                  </option>
                ))}
              </optgroup>
            );
          })}
        </select>
      </label>
    </div>
  );
}
