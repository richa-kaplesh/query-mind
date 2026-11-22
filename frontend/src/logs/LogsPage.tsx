import { useNavigate, useSearchParams } from "react-router-dom";
import EvaluationsTab from "./evals/EvaluationsTab";
import TracesTab from "./traces/TracesTab";
import CostTab from "./cost/CostTab";
import "./logs.css";

const TABS = [
  { id: "evaluations", label: "Evaluations" },
  { id: "traces", label: "Traces" },
  { id: "cost", label: "Cost" },
] as const;

type TabId = typeof TABS[number]["id"];

function toTab(value: string | null): TabId {
  return TABS.some(t => t.id === value) ? (value as TabId) : "evaluations";
}

export default function LogsPage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const active = toTab(params.get("tab"));

  function select(id: TabId) {
    const next = new URLSearchParams(params);
    if (id === "evaluations") next.delete("tab");
    else next.set("tab", id);
    setParams(next, { replace: true });
  }

  return (
    <div className="db-root lg-root">
      <header className="lg-header">
        <div className="lg-header-text">
          <h1 className="lg-title">Under the hood</h1>
          <p className="lg-subtitle">How QueryMind is measured, traced and costed.</p>
        </div>
        <button className="lg-btn lg-btn--ghost" onClick={() => navigate("/")}>
          <span aria-hidden>‹</span> Back to chat
        </button>
      </header>

      <nav className="lg-tabs" role="tablist" aria-label="Sections">
        {TABS.map(t => (
          <button
            key={t.id}
            role="tab"
            id={`lg-tab-${t.id}`}
            aria-selected={active === t.id}
            aria-controls={`lg-panel-${t.id}`}
            className={`lg-tab ${active === t.id ? "is-active" : ""}`}
            onClick={() => select(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <section
        className={`lg-panel lg-panel--${active}`}
        role="tabpanel"
        id={`lg-panel-${active}`}
        aria-labelledby={`lg-tab-${active}`}
      >
        {active === "evaluations" && <EvaluationsTab />}
        {active === "traces" && <TracesTab />}
        {active === "cost" && <CostTab />}
      </section>
    </div>
  );
}
