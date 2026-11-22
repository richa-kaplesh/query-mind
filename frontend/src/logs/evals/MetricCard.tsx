import type { Metric } from "../types";
import { deltaTone, formatDelta } from "../thresholds";
import Sparkline from "./Sparkline";

export default function MetricCard({ metric, baselineValue, series, selectedIdx }: {
  metric: Metric;
  baselineValue: number | null;
  series: number[];
  selectedIdx: number;
}) {
  const delta = baselineValue === null ? null : metric.value - baselineValue;
  const tone = delta === null ? "neutral" : deltaTone(delta, metric.higher_is_better);
  return (
    <div className="lg-card lg-metric">
      <div className="lg-metric-label">{metric.label}</div>
      <div className="lg-metric-row">
        <span className="lg-metric-value">{metric.value.toFixed(2)}</span>
        {delta !== null && (
          <span
            className={`lg-delta lg-delta--${tone}`}
            title={`Baseline ${baselineValue!.toFixed(2)}`}
          >
            {formatDelta(delta)}
          </span>
        )}
      </div>
      <Sparkline values={series} selectedIdx={selectedIdx} label={metric.label} />
    </div>
  );
}
