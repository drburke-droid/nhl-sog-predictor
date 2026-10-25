import { useEffect, useMemo, useState } from "react";
import { Route } from "../model/equilibrium";
import { LETTERS, ROUTE_DASH, ROUTE_VAR, fmt, fmtInt } from "./ui";

const W = 900;
const H = 330;
const O: [number, number] = [70, 165];
const D: [number, number] = [830, 165];
// control points of the three roads: A over the top, B straight across, C underneath
const CURVES: [number, number][][] = [
  [O, [250, 25], [650, 25], D],
  [O, [320, 165], [580, 165], D],
  [O, [250, 305], [650, 305], D],
];

function bez(c: [number, number][], t: number): [number, number] {
  const u = 1 - t;
  return [
    u * u * u * c[0][0] + 3 * u * u * t * c[1][0] + 3 * u * t * t * c[2][0] + t * t * t * c[3][0],
    u * u * u * c[0][1] + 3 * u * u * t * c[1][1] + 3 * u * t * t * c[2][1] + t * t * t * c[3][1],
  ];
}

/** points evenly spaced by arc length, so cars move at constant speed */
function sample(c: [number, number][], n = 240): [number, number][] {
  const fine = Array.from({ length: 2001 }, (_, i) => bez(c, i / 2000));
  const cum = [0];
  for (let i = 1; i < fine.length; i++) cum.push(cum[i - 1] + Math.hypot(fine[i][0] - fine[i - 1][0], fine[i][1] - fine[i - 1][1]));
  const L = cum[cum.length - 1];
  const out: [number, number][] = [];
  let j = 0;
  for (let k = 0; k <= n; k++) {
    const s = (k / n) * L;
    while (j < cum.length - 2 && cum[j + 1] < s) j++;
    const f = (s - cum[j]) / (cum[j + 1] - cum[j] || 1);
    out.push([fine[j][0] + f * (fine[j + 1][0] - fine[j][0]), fine[j][1] + f * (fine[j + 1][1] - fine[j][1])]);
  }
  return out;
}

const SAMPLES = CURVES.map((c) => sample(c));
const PATHS = CURVES.map((c) => `M${c[0]}C${c[1]} ${c[2]} ${c[3]}`);
const LABEL_AT: [number, number][] = [[450, 24], [450, 122], [450, 316]];

function useClock(): number {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      setT((now - t0) / 1000);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  return t;
}

const STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];

export function RouteSchematic(p: { routes: Route[]; flows: number[]; times: number[]; used: boolean[]; open: boolean[]; caption?: string }) {
  const t = useClock();
  const maxCap = Math.max(1, ...p.routes.map((r) => r.cap));
  // vehicles on each road at once = flow x travel time (Little's law); one dot stands for `per` of them
  const onRoad = p.flows.map((f, i) => (p.open[i] && Number.isFinite(p.times[i]) ? (f * p.times[i]) / 60 : 0));
  const per = useMemo(() => STEPS.find((s) => onRoad.reduce((a, b) => a + b, 0) / s <= 150) ?? 10000, [onRoad]);
  const maxFlow = Math.max(1, ...p.flows);
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H + 20}`} role="img" aria-label="Three routes from origin to destination; road width shows capacity, dots show cars">
        {p.routes.map((r, i) => {
          const casing = 8 + 22 * (Math.max(0, r.cap) / maxCap);
          return (
            <g key={`road${i}`}>
              <path d={PATHS[i]} fill="none" stroke="var(--road-edge)" strokeWidth={casing + 3} strokeLinecap="round" opacity={p.open[i] ? 1 : 0.4} />
              <path d={PATHS[i]} fill="none" stroke="var(--road)" strokeWidth={casing} strokeLinecap="round" opacity={p.open[i] ? 1 : 0.4} />
              {p.used[i] && (
                <path d={PATHS[i]} fill="none" stroke={ROUTE_VAR[i]} strokeWidth={2 + 6 * (p.flows[i] / maxFlow)} strokeDasharray={ROUTE_DASH[i] || undefined} strokeLinecap="round" opacity={0.85} />
              )}
              {!p.used[i] && p.open[i] && (
                <path d={PATHS[i]} fill="none" stroke={ROUTE_VAR[i]} strokeWidth={1.5} strokeDasharray="2 7" strokeLinecap="round" opacity={0.7} />
              )}
            </g>
          );
        })}
        {p.routes.map((_, i) => {
          if (!p.used[i]) return null;
          const n = Math.min(120, Math.round(onRoad[i] / per));
          if (n <= 0) return null;
          const lap = Math.max(0.5, p.times[i] * 0.22); // seconds on screen per trip: slower roads crawl
          const dots = [];
          for (let k = 0; k < n; k++) {
            const s = (((t / lap + k / n) % 1) + 1) % 1;
            const [x, y] = SAMPLES[i][Math.round(s * (SAMPLES[i].length - 1))];
            dots.push(<circle key={k} cx={x} cy={y} r={3.6} fill={ROUTE_VAR[i]} stroke="var(--surface)" strokeWidth={1.2} />);
          }
          return <g key={`cars${i}`}>{dots}</g>;
        })}
        {[O, D].map(([x, y], k) => (
          <g key={`end${k}`}>
            <circle cx={x} cy={y} r={15} fill="var(--ink)" />
            <text x={x} y={y + 4.5} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--surface)">{k ? "D" : "O"}</text>
            <text x={x} y={y + 34} textAnchor="middle" fontSize={12} fill="var(--ink-2)">{k ? "destination" : "origin"}</text>
          </g>
        ))}
        {p.routes.map((_, i) => {
          const [x, y] = LABEL_AT[i];
          const w = 238;
          const status = !p.open[i] ? "CLOSED" : p.used[i] ? `${fmtInt(p.flows[i])} veh/h` : "UNUSED · no faster";
          return (
            <g key={`lab${i}`} transform={`translate(${x - w / 2},${y - 16})`}>
              <rect width={w} height={30} rx={15} fill="var(--surface)" stroke="var(--ring)" />
              <circle cx={16} cy={15} r={10} fill={ROUTE_VAR[i]} />
              <text x={16} y={19} textAnchor="middle" fontSize={11} fontWeight={700} fill="#fff">{LETTERS[i]}</text>
              <text x={32} y={19.5} fontSize={13} fontWeight={650} fill="var(--ink)" style={{ fontVariantNumeric: "tabular-nums" }}>
                {Number.isFinite(p.times[i]) && p.open[i] ? `${fmt(p.times[i])} min` : "—"}
              </text>
              <text x={w - 12} y={19.5} fontSize={12} textAnchor="end" fill={p.used[i] ? "var(--ink-2)" : "var(--muted)"} style={{ fontVariantNumeric: "tabular-nums" }}>{status}</text>
            </g>
          );
        })}
        <text x={W - 10} y={H + 14} textAnchor="end" fontSize={11} fill="var(--muted)">
          Road width: capacity · coloured line: flow · each dot ≈ {per.toLocaleString("en-US")} vehicle{per > 1 ? "s" : ""} on the road (flow × trip time)
        </text>
        {p.caption && <text x={10} y={H + 14} fontSize={11} fill="var(--muted)">{p.caption}</text>}
      </svg>
    </div>
  );
}
