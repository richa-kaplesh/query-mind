const W = 120;
const H = 32;
const PAD = 4;

export default function Sparkline({ values, selectedIdx, label }: {
  values: number[];
  selectedIdx: number;
  label: string;
}) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => ({
    x: PAD + (i * (W - PAD * 2)) / (values.length - 1),
    y: max === min ? H / 2 : PAD + (1 - (v - min) / span) * (H - PAD * 2),
  }));
  const sel = pts[selectedIdx];
  return (
    <svg
      className="lg-spark"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`${label} across ${values.length} runs, from ${values[0].toFixed(2)} to ${values[values.length - 1].toFixed(2)}`}
    >
      <polyline className="lg-spark-line" points={pts.map(p => `${p.x},${p.y}`).join(" ")} />
      {sel && <circle className="lg-spark-dot" cx={sel.x} cy={sel.y} r={3} />}
    </svg>
  );
}
