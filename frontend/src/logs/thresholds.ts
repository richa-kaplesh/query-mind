export const SCORE_THRESHOLD = 0.7;

export const DELTA_EPSILON = 0.005;

export const LARGE_REGRESSION = 0.05;

export type DeltaTone = "success" | "warning" | "danger" | "neutral";

export function deltaTone(delta: number, higherIsBetter: boolean): DeltaTone {
  const signed = higherIsBetter ? delta : -delta;
  if (signed >= DELTA_EPSILON) return "success";
  if (signed < -1e-9) return -signed >= LARGE_REGRESSION ? "danger" : "warning";
  return "neutral";
}

export function isBelowThreshold(value: number, higherIsBetter = true): boolean {
  return higherIsBetter ? value < SCORE_THRESHOLD : value > 1 - SCORE_THRESHOLD;
}

export function formatDelta(delta: number): string {
  const r = Math.round(delta * 100) / 100;
  if (r === 0) return delta >= 0 ? "+0.00" : "−0.00";
  return `${r > 0 ? "+" : "−"}${Math.abs(r).toFixed(2)}`;
}
