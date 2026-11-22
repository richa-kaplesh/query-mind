import type { RunNotes } from "../types";

const FIELDS: { key: keyof RunNotes; label: string }[] = [
  { key: "problem", label: "Problem" },
  { key: "root_cause", label: "Root cause" },
  { key: "fix", label: "Fix" },
];

export function ChangeTags({ changes }: { changes: string[] }) {
  if (changes.length === 0) return null;
  return (
    <ul className="lg-tags" aria-label="Config changes">
      {changes.map(c => <li key={c} className="lg-tag">{c}</li>)}
    </ul>
  );
}

export default function FixCard({ notes, changes }: { notes: RunNotes; changes: string[] }) {
  return (
    <div className="lg-card lg-fix">
      <h2 className="lg-section-title">Latest fix</h2>
      <dl className="lg-fix-list">
        {FIELDS.filter(f => notes[f.key]).map(f => (
          <div key={f.key} className="lg-fix-item">
            <dt>{f.label}</dt>
            <dd>{notes[f.key]}</dd>
          </div>
        ))}
      </dl>
      {notes.result && (
        <div className="lg-fix-result">
          <span className="lg-fix-result-label">Result</span>
          <p>{notes.result}</p>
        </div>
      )}
      <ChangeTags changes={changes} />
    </div>
  );
}
