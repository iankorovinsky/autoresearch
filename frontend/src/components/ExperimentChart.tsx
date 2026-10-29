"use client";
import { useId, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowUpRight,
  RotateCcw,
  Table2,
  ZoomIn,
} from "lucide-react";
import {
  bestHistory,
  chartDomain,
  clampRange,
  experimentsCsv,
  metricValue,
  nearestPoint,
  sampleExperiments,
  toggleComparison,
} from "../lib/charts/experiments";
import type { ChartMetric } from "../lib/charts/experiments";
import "./experiment-chart.css";
const points = sampleExperiments;
const last = points.length - 1;
const fmt = (value: number) => value.toFixed(2);
export default function ExperimentChart({
  large = false,
  onInspect,
}: {
  large?: boolean;
  onInspect?: (id: string) => void;
}) {
  const [metric, setMetric] = useState<ChartMetric>("runtime");
  const [baseline, setBaseline] = useState(true);
  const [best, setBest] = useState(false);
  const [range, setRange] = useState<[number, number]>([0, last]);
  const [hover, setHover] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number[]>([]);
  const [table, setTable] = useState(false);
  const [drag, setDrag] = useState<[number, number] | null>(null);
  const startRef = useRef<{ index: number; x: number } | null>(null);
  const id = useId();
  const visible = points.slice(range[0], range[1] + 1);
  const history = bestHistory(points, metric);
  const values = visible.map((p) => metricValue(p, metric));
  const baselineValue = metric === "runtime" ? points[0].baseline : 1;
  const [low, high] = chartDomain([
    ...values,
    ...(baseline ? [baselineValue] : []),
    ...(best ? history.slice(range[0], range[1] + 1) : []),
  ]);
  const x = (index: number) =>
    52 + ((index - range[0]) / (range[1] - range[0])) * 540;
  const y = (value: number) => 201 - ((value - low) / (high - low)) * 174;
  const proposed = hover ?? pinned.at(-1) ?? range[1];
  const active = Math.max(range[0], Math.min(range[1], proposed));
  const point = points[active];
  const line = visible
    .map((p, i) => `${x(i + range[0])},${y(metricValue(p, metric))}`)
    .join(" ");
  function locate(event: React.PointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return nearestPoint(
      (((event.clientX - rect.left) / rect.width) * 620 - 52) / 540,
      range[0],
      range[1],
    );
  }
  function select(index: number) {
    setPinned((current) => toggleComparison(current, index));
    setHover(index);
  }
  function changeRange(start: number, end: number) {
    setRange(clampRange(start, end, points.length));
    setHover(null);
    setDrag(null);
  }
  function reset() {
    changeRange(0, last);
    setPinned([]);
    setBaseline(true);
    setBest(false);
  }
  function exportCsv() {
    const url = URL.createObjectURL(
      new Blob([experimentsCsv(visible)], { type: "text/csv;charset=utf-8" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `autoresearch-sample-experiments-${range[0] + 1}-${range[1] + 1}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <section
      className={`experiment-chart ${large ? "expanded-chart" : ""}`}
      aria-label="Interactive sample experiment chart"
    >
      <div className="ec-toolbar">
        <div className="ec-segment" aria-label="Chart metric">
          <button
            aria-pressed={metric === "runtime"}
            onClick={() => setMetric("runtime")}
          >
            Runtime <span>ms</span>
          </button>
          <button
            aria-pressed={metric === "speedup"}
            onClick={() => setMetric("speedup")}
          >
            Speedup <span>×</span>
          </button>
        </div>
        <div className="ec-actions">
          <button
            onClick={exportCsv}
            aria-label="Export visible sample measurements as CSV"
            title="Export visible data"
          >
            <ArrowDownToLine size={14} />
          </button>
          <button
            aria-pressed={table}
            onClick={() => setTable(!table)}
            aria-label="Show chart data table"
            title="Data table"
          >
            <Table2 size={14} />
          </button>
          <button onClick={reset} aria-label="Reset chart" title="Reset chart">
            <RotateCcw size={14} />
          </button>
        </div>
      </div>
      <div className="ec-legend">
        <span>
          <i />
          Candidate
        </span>
        <button aria-pressed={baseline} onClick={() => setBaseline(!baseline)}>
          <i className="ec-baseline-key" />
          Baseline
        </button>
        <button aria-pressed={best} onClick={() => setBest(!best)}>
          <i className="ec-best-key" />
          Best so far
        </button>
        <small>
          {metric === "runtime" ? "Lower is better" : "Higher is better"}
        </small>
      </div>
      <div className="ec-readout" aria-live="polite">
        <div>
          <span>{point.id}</span>
          <strong>
            {fmt(point.runtime)} <small>ms</small>
          </strong>
        </div>
        <div>
          <span>vs. baseline</span>
          <strong>
            {fmt(point.baseline / point.runtime)}
            <small>×</small>
          </strong>
        </div>
        <p>
          {point.configuration}
          <span>
            {pinned.includes(active)
              ? "Pinned for comparison"
              : "Click a point to compare"}
          </span>
        </p>
      </div>
      <div
        className="ec-plot"
        tabIndex={0}
        role="group"
        aria-roledescription="interactive chart"
        aria-label="Experiment plot"
        aria-describedby={`${id}-help`}
        onBlur={() => setHover(null)}
        onKeyDown={(event) => {
          let target = active;
          if (event.key === "ArrowLeft")
            target = Math.max(range[0], active - 1);
          else if (event.key === "ArrowRight")
            target = Math.min(range[1], active + 1);
          else if (event.key === "Home") target = range[0];
          else if (event.key === "End") target = range[1];
          else if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            select(active);
            return;
          } else if (event.key === "Escape") {
            setPinned([]);
            setDrag(null);
            return;
          } else return;
          event.preventDefault();
          setHover(target);
        }}
      >
        <svg
          viewBox="0 0 620 235"
          role="img"
          aria-label={`${metric === "runtime" ? "Runtime in milliseconds" : "Speedup versus baseline"}, experiments ${range[0] + 1} to ${range[1] + 1}. Sample data.`}
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            const index = locate(event);
            startRef.current = { index, x: event.clientX };
            event.currentTarget.setPointerCapture(event.pointerId);
            setHover(index);
            setDrag([index, index]);
            event.currentTarget.parentElement?.focus();
          }}
          onPointerMove={(event) => {
            const index = locate(event);
            setHover(index);
            if (startRef.current) setDrag([startRef.current.index, index]);
          }}
          onPointerUp={(event) => {
            const start = startRef.current;
            startRef.current = null;
            setDrag(null);
            if (!start) return;
            const index = locate(event);
            if (Math.abs(event.clientX - start.x) > 8 && index !== start.index)
              changeRange(start.index, index);
            else select(index);
            if (event.currentTarget.hasPointerCapture(event.pointerId))
              event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          onPointerCancel={() => {
            startRef.current = null;
            setDrag(null);
          }}
          onPointerLeave={() => {
            if (!startRef.current) setHover(null);
          }}
        >
          <defs>
            <linearGradient id={`${id}-fill`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="var(--accent)" stopOpacity=".16" />
              <stop offset="1" stopColor="var(--accent)" stopOpacity="0" />
            </linearGradient>
          </defs>
          {[0, 1, 2, 3, 4].map((i) => {
            const value = low + ((high - low) * i) / 4;
            return (
              <g key={i}>
                <line
                  className="ec-grid"
                  x1="52"
                  x2="592"
                  y1={y(value)}
                  y2={y(value)}
                />
                <text x="42" y={y(value) + 4} textAnchor="end">
                  {fmt(value)}
                </text>
              </g>
            );
          })}
          <path
            d={`M ${x(range[0])} 201 L ${line.replaceAll(",", " ")} L ${x(range[1])} 201 Z`}
            fill={`url(#${id}-fill)`}
          />
          {baseline && (
            <line
              className="ec-baseline"
              x1="52"
              x2="592"
              y1={y(baselineValue)}
              y2={y(baselineValue)}
            />
          )}
          {best && (
            <polyline
              className="ec-best"
              points={visible
                .map((_, i) => `${x(i + range[0])},${y(history[i + range[0]])}`)
                .join(" ")}
            />
          )}
          <polyline className="ec-line" points={line} />
          {active >= range[0] && active <= range[1] && (
            <line
              className="ec-crosshair"
              x1={x(active)}
              x2={x(active)}
              y1="22"
              y2="201"
            />
          )}
          {visible.map((p, i) => {
            const index = i + range[0];
            return (
              <circle
                key={p.id}
                className={`ec-point ${pinned.includes(index) ? "pinned" : ""}`}
                cx={x(index)}
                cy={y(metricValue(p, metric))}
                r={index === active || pinned.includes(index) ? 5 : 3}
              >
                <title>{`${p.id}: ${fmt(p.runtime)} ms, ${fmt(p.baseline / p.runtime)}× speedup`}</title>
              </circle>
            );
          })}
          {visible
            .filter(
              (_, i) =>
                i === 0 ||
                i === visible.length - 1 ||
                i % Math.ceil(visible.length / 5) === 0,
            )
            .map((p) => (
              <text
                key={p.id}
                x={x(p.iteration - 1)}
                y="222"
                textAnchor="middle"
              >
                {String(p.iteration).padStart(2, "0")}
              </text>
            ))}
          {drag && drag[0] !== drag[1] && (
            <rect
              className="ec-brush"
              x={x(Math.min(...drag))}
              y="22"
              width={Math.abs(x(drag[0]) - x(drag[1]))}
              height="179"
            />
          )}
        </svg>
      </div>
      <div className="ec-range-label">
        <span>
          <ZoomIn size={12} /> Experiments {range[0] + 1}–{range[1] + 1}{" "}
          <small>/ {points.length}</small>
        </span>
        <button
          onClick={() => changeRange(0, last)}
          disabled={range[0] === 0 && range[1] === last}
        >
          Show all
        </button>
      </div>
      <div className="ec-range">
        <label>
          From <output>{range[0] + 1}</output>
          <input
            aria-label="First visible experiment"
            type="range"
            min="1"
            max={last}
            value={range[0] + 1}
            onChange={(e) =>
              changeRange(
                Math.min(Number(e.target.value) - 1, range[1] - 1),
                range[1],
              )
            }
          />
        </label>
        <label>
          To <output>{range[1] + 1}</output>
          <input
            aria-label="Last visible experiment"
            type="range"
            min="2"
            max={points.length}
            value={range[1] + 1}
            onChange={(e) =>
              changeRange(
                range[0],
                Math.max(Number(e.target.value) - 1, range[0] + 1),
              )
            }
          />
        </label>
      </div>
      <p className="ec-help" id={`${id}-help`}>
        Drag to zoom · click two points to compare · arrow keys explore, Enter
        pins · sample data
      </p>
      {!!pinned.length && (
        <div className="ec-comparison">
          <div className="ec-comparison-heading">
            <strong>Compare candidates</strong>
            <button onClick={() => setPinned([])}>Clear</button>
          </div>
          <div className="ec-comparison-cards">
            {pinned.map((index) => (
              <div key={index}>
                <button
                  aria-label={`Remove ${points[index].id} from comparison`}
                  className="ec-unpin"
                  onClick={() => setPinned((p) => p.filter((i) => i !== index))}
                >
                  ×
                </button>
                <span>{points[index].id}</span>
                <strong>
                  {fmt(points[index].runtime)} <small>ms</small>
                </strong>
                <p>
                  {fmt(points[index].baseline / points[index].runtime)}× speedup
                </p>
                {onInspect && (
                  <button
                    className="ec-inspect"
                    onClick={() => onInspect(points[index].id)}
                  >
                    Inspect code <ArrowUpRight size={12} />
                  </button>
                )}
              </div>
            ))}
          </div>
          {pinned.length === 2 ? (
            <p className="ec-delta">
              {points[pinned[1]].id} is{" "}
              <strong>
                {fmt(
                  Math.abs(
                    1 - points[pinned[1]].runtime / points[pinned[0]].runtime,
                  ) * 100,
                )}
                %{" "}
                {points[pinned[1]].runtime <= points[pinned[0]].runtime
                  ? "faster"
                  : "slower"}
              </strong>{" "}
              than {points[pinned[0]].id} by runtime.
            </p>
          ) : (
            <p className="ec-delta">
              Choose another point to compare runtimes.
            </p>
          )}
        </div>
      )}
      {large && (
        <div className="ec-bars">
          <h3>
            Candidate speedup <span>Visible experiments</span>
          </h3>
          {visible.map((p) => (
            <button
              key={p.id}
              aria-pressed={pinned.includes(p.iteration - 1)}
              onClick={() => select(p.iteration - 1)}
              onMouseEnter={() => setHover(p.iteration - 1)}
              onMouseLeave={() => setHover(null)}
            >
              <span>{p.id}</span>
              <span className="ec-bar-track">
                <i
                  style={{
                    width: `${(p.baseline / p.runtime / Math.max(...visible.map((v) => v.baseline / v.runtime))) * 100}%`,
                  }}
                />
              </span>
              <strong>{fmt(p.baseline / p.runtime)}×</strong>
            </button>
          ))}
        </div>
      )}
      {table && (
        <div className="ec-table-wrap">
          <table>
            <caption>
              Visible sample measurements · select up to two to compare
            </caption>
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Runtime</th>
                <th>Speedup</th>
                <th>Compare</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((p) => (
                <tr key={p.id}>
                  <th scope="row">{p.id}</th>
                  <td>{fmt(p.runtime)} ms</td>
                  <td>{fmt(p.baseline / p.runtime)}×</td>
                  <td>
                    <button
                      aria-label={`Compare ${p.id}`}
                      aria-pressed={pinned.includes(p.iteration - 1)}
                      onClick={() => select(p.iteration - 1)}
                    >
                      {pinned.includes(p.iteration - 1) ? "Selected" : "Select"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
