import type { Metric } from "../types";
import { deltaTone } from "../thresholds";
import { formatMetric, formatMetricDelta, formatMetricText } from "./evalMath";
import Sparkline from "./Sparkline";

export default function MetricCard({ metric, baseline, series, selectedIdx }: {
  metric: Metric;
  baseline: Metric | null;
  series: number[];
  selectedIdx: number;
}) {
  const delta = baseline === null ? null : metric.value - baseline.value;
  const tone = delta === null ? "neutral" : deltaTone(delta, metric.higher_is_better);
  const shown = formatMetric(metric);
  return (
    <div className="lg-card lg-metric">
      <div className="lg-metric-label">{metric.label}</div>
      <div className="lg-metric-row">
        <span className="lg-metric-value">
          {shown.main}
          {shown.sub && <span className="lg-metric-sub"> · {shown.sub}</span>}
        </span>
        {delta !== null && baseline && (
          <span
            className={`lg-delta lg-delta--${tone}`}
            title={`Baseline ${formatMetricText(baseline)}`}
          >
            {formatMetricDelta(delta, metric)}
          </span>
        )}
      </div>
      <Sparkline values={series} selectedIdx={selectedIdx} label={metric.label} />
    </div>
  );
}
