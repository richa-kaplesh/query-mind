import { useMemo } from "react";
import { useJson } from "../api";
import type { CostCall } from "../types";
import { EmptyState, ErrorState, Skeleton } from "../StateViews";
import { costUsd, priceFor } from "../pricing";
import { CALL_KINDS, KIND_ORDER, kindOf } from "./callTypes";

const MAX_ROWS = 200;

function fmtInt(n: number): string {
  return n.toLocaleString();
}

function fmtUsd(n: number): string {
  if (n === 0) return "$0.00";
  return n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`;
}

function fmtPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function fmtTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function CostTab() {
  const res = useJson<{ calls: CostCall[] }>("/api/costs");
  const calls = useMemo(() => res.data?.calls ?? [], [res.data]);

  const summary = useMemo(() => {
    let prompt = 0;
    let completion = 0;
    let cost = 0;
    const byKind = new Map<string, { calls: number; prompt: number; completion: number; cost: number }>();
    const unpriced = new Set<string>();
    for (const c of calls) {
      const usd = costUsd(c.model, c.prompt_tokens, c.completion_tokens);
      if (!priceFor(c.model).confirmed) unpriced.add(c.model);
      prompt += c.prompt_tokens;
      completion += c.completion_tokens;
      cost += usd;
      const k = kindOf(c.call_type).id;
      const row = byKind.get(k) ?? { calls: 0, prompt: 0, completion: 0, cost: 0 };
      row.calls += 1;
      row.prompt += c.prompt_tokens;
      row.completion += c.completion_tokens;
      row.cost += usd;
      byKind.set(k, row);
    }
    const total = prompt + completion;
    const kinds = KIND_ORDER.filter(k => byKind.has(k)).map(k => ({ kind: CALL_KINDS[k], ...byKind.get(k)! }));
    return { prompt, completion, total, cost, kinds, unpriced: Array.from(unpriced) };
  }, [calls]);

  if (res.loading) return <div className="lg-pad"><Skeleton cards={4} rows={5} /></div>;
  if (res.error) return <ErrorState message={res.error} onRetry={res.reload} />;
  if (calls.length === 0) {
    return (
      <EmptyState title="No LLM calls recorded yet">
        <p className="lg-state-text">Every LLM call made while answering a question is logged here with its token counts.</p>
      </EmptyState>
    );
  }

  const promptShare = summary.total ? summary.prompt / summary.total : 0;
  const top = [...summary.kinds].sort((a, b) => (b.prompt + b.completion) - (a.prompt + a.completion))[0];

  return (
    <div className="lg-cost">
      <p className="lg-headline">
        {fmtPct(promptShare)} of tokens are prompt/context; {top.kind.label.toLowerCase()} calls use the most
        ({fmtPct((top.prompt + top.completion) / (summary.total || 1))}).
      </p>

      <div className="lg-metrics">
        <div className="lg-card lg-kpi">
          <div className="lg-metric-label">LLM calls</div>
          <div className="lg-metric-value">{fmtInt(calls.length)}</div>
        </div>
        <div className="lg-card lg-kpi">
          <div className="lg-metric-label">Total tokens</div>
          <div className="lg-metric-value">{fmtInt(summary.total)}</div>
        </div>
        <div className="lg-card lg-kpi">
          <div className="lg-metric-label">Prompt / completion</div>
          <div className="lg-metric-value">{fmtInt(summary.prompt)} <span className="lg-muted">/ {fmtInt(summary.completion)}</span></div>
        </div>
        <div className="lg-card lg-kpi">
          <div className="lg-metric-label">Estimated cost</div>
          <div className="lg-metric-value">{fmtUsd(summary.cost)}</div>
          {summary.unpriced.length > 0 && (
            <div className="lg-muted lg-kpi-note">Price not set for {summary.unpriced.join(", ")} (logs/pricing.ts)</div>
          )}
        </div>
      </div>

      <div className="lg-card">
        <h2 className="lg-section-title">Tokens by call type</h2>
        <ul className="lg-bars">
          {summary.kinds.map(k => {
            const tokens = k.prompt + k.completion;
            const share = summary.total ? tokens / summary.total : 0;
            return (
              <li key={k.kind.id} className="lg-bar-row">
                <span className="lg-bar-label">
                  <span className="lg-swatch" style={{ background: k.kind.color }} aria-hidden />
                  {k.kind.label}
                </span>
                <span className="lg-bar-track" aria-hidden>
                  <span className="lg-bar-fill" style={{ width: `${share * 100}%`, background: k.kind.color }} />
                </span>
                <span className="lg-num">{fmtPct(share)}</span>
                <span className="lg-num lg-muted">{fmtInt(k.calls)} calls · {fmtInt(tokens)} tok · {fmtUsd(k.cost)}</span>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="lg-card">
        <h2 className="lg-section-title">
          Calls <span className="lg-muted">newest first{calls.length > MAX_ROWS ? `, latest ${MAX_ROWS} of ${fmtInt(calls.length)}` : ""}</span>
        </h2>
        <div className="lg-table-wrap">
          <table className="lg-table">
            <thead>
              <tr>
                <th scope="col">Time</th>
                <th scope="col">Call type</th>
                <th scope="col">Model</th>
                <th scope="col" className="lg-num">Prompt</th>
                <th scope="col" className="lg-num">Completion</th>
                <th scope="col" className="lg-num">Est. cost</th>
              </tr>
            </thead>
            <tbody>
              {calls.slice(0, MAX_ROWS).map((c, i) => {
                const kind = kindOf(c.call_type);
                return (
                  <tr key={`${c.timestamp}-${i}`}>
                    <td className="lg-nowrap">{fmtTime(c.timestamp)}</td>
                    <td className="lg-nowrap" title={c.call_type}>
                      <span className="lg-swatch" style={{ background: kind.color }} aria-hidden />
                      {kind.label}
                    </td>
                    <td className="lg-mono">{c.model}</td>
                    <td className="lg-num">{fmtInt(c.prompt_tokens)}</td>
                    <td className="lg-num">{fmtInt(c.completion_tokens)}</td>
                    <td className="lg-num">{fmtUsd(costUsd(c.model, c.prompt_tokens, c.completion_tokens))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
