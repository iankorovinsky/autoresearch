export type ExperimentPoint = {
  id: string;
  iteration: number;
  runtime: number;
  baseline: number;
  configuration: string;
};
export type ChartMetric = "runtime" | "speedup";
export const sampleExperiments: ExperimentPoint[] = [
  2.99, 2.73, 2.51, 2.42, 2.54, 2.3, 2.21, 2.28, 2.12, 1.67, 1.48, 1.24,
].map((runtime, i) => ({
  id: `EXP-${String(i + 1).padStart(3, "0")}`,
  iteration: i + 1,
  runtime,
  baseline: 2.99,
  configuration:
    i === 11
      ? "Tile 64 × 64 · shared memory"
      : i === 10
        ? "Tile 32 × 32 · loop unrolling"
        : i === 9
          ? "Tile 32 × 32 · shared memory"
          : i === 8
            ? "Tile 16 × 16 · shared memory"
            : "Exploratory configuration",
}));
export function metricValue(point: ExperimentPoint, metric: ChartMetric) {
  return metric === "runtime" ? point.runtime : point.baseline / point.runtime;
}
export function bestHistory(points: ExperimentPoint[], metric: ChartMetric) {
  let best = Infinity;
  return points.map((p) => {
    best = Math.min(best, p.runtime);
    return metric === "runtime" ? best : p.baseline / best;
  });
}
export function clampRange(
  start: number,
  end: number,
  length: number,
): [number, number] {
  if (length < 2) return [0, Math.max(0, length - 1)];
  const low = Math.max(
    0,
    Math.min(length - 2, Math.round(Math.min(start, end))),
  );
  return [
    low,
    Math.max(low + 1, Math.min(length - 1, Math.round(Math.max(start, end)))),
  ];
}
export function nearestPoint(fraction: number, start: number, end: number) {
  return Math.round(start + Math.max(0, Math.min(1, fraction)) * (end - start));
}
export function chartDomain(values: number[]) {
  const low = Math.min(...values),
    high = Math.max(...values);
  const padding = (high - low) * 0.12 || Math.abs(high) * 0.1 || 0.1;
  return [Math.max(0, low - padding), high + padding] as const;
}
export function toggleComparison(selected: number[], index: number) {
  return selected.includes(index)
    ? selected.filter((i) => i !== index)
    : [...selected.slice(-1), index];
}
export function experimentsCsv(points: ExperimentPoint[]) {
  const quote = (v: string | number) => `"${String(v).replaceAll('"', '""')}"`;
  return (
    [
      "experiment,iteration,runtime_ms,baseline_runtime_ms,speedup,configuration,data_source",
      ...points.map((p) =>
        [
          p.id,
          p.iteration,
          p.runtime,
          p.baseline,
          (p.baseline / p.runtime).toFixed(4),
          p.configuration,
          "sample",
        ]
          .map(quote)
          .join(","),
      ),
    ].join("\r\n") + "\r\n"
  );
}
