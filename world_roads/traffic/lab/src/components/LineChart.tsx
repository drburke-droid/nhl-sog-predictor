import { ReactNode, useRef, useState } from "react";

export interface Series {
  id: string;
  name: string;
  color: string;
  dash?: string;
  width?: number;
  opacity?: number;
  points: [number, number][]; // sorted by x
  tooltip?: boolean; // include in the hover tooltip (default true)
  unit?: string;
}

export interface ChartProps {
  title?: ReactNode;
  sub?: ReactNode;
  series: Series[];
  xDomain: [number, number];
  yDomain?: [number, number];
  xLabel: string;
  yLabel: string;
  xFormat?: (x: number) => string;
  yFormat?: (y: number) => string;
  cursorX?: number;
  cursorLabel?: string;
  refY?: { y: number; label: string; color?: string }[];
  bands?: { x0: number; x1: number; label?: string }[];
  markers?: { x: number; y: number; color: string; label?: string }[];
  onPickX?: (x: number) => void;
  height?: number;
  legend?: ReactNode;
}

export function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function interp(pts: [number, number][], x: number): number | null {
  if (!pts.length || x < pts[0][0] - 1e-9 || x > pts[pts.length - 1][0] + 1e-9) return null;
  let lo = 0;
  let hi = pts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (pts[mid][0] <= x) lo = mid;
    else hi = mid;
  }
  const [x0, y0] = pts[lo];
  const [x1, y1] = pts[hi];
  if (x1 === x0) return y0;
  return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
}

export function LineChart(p: ChartProps) {
  const W = 640;
  const H = p.height ?? 250;
  const m = { l: 52, r: 16, t: 12, b: 38 };
  const ref = useRef<SVGSVGElement>(null);
  const [hx, setHx] = useState<number | null>(null);
  const [x0, x1] = p.xDomain;
  let [y0, y1] = p.yDomain ?? [Infinity, -Infinity];
  if (!p.yDomain) {
    for (const s of p.series) for (const [, y] of s.points) if (Number.isFinite(y)) { y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    for (const r of p.refY ?? []) { y0 = Math.min(y0, r.y); y1 = Math.max(y1, r.y); }
    if (!Number.isFinite(y0)) { y0 = 0; y1 = 1; }
    const pad = (y1 - y0) * 0.08 || 1;
    y0 = Math.max(0, y0 - pad);
    y1 = y1 + pad;
  }
  const sx = (x: number) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (y: number) => H - m.b - ((Math.min(Math.max(y, y0 - (y1 - y0)), y1 + (y1 - y0)) - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const xf = p.xFormat ?? ((x: number) => x.toLocaleString("en-US"));
  const yf = p.yFormat ?? ((y: number) => y.toLocaleString("en-US"));
  const xt = niceTicks(x0, x1, 6);
  const yt = niceTicks(y0, y1, 5);
  const path = (pts: [number, number][]) =>
    pts.filter(([, y]) => Number.isFinite(y)).map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");

  const toX = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    const px = ((clientX - r.left) / r.width) * W;
    return Math.min(x1, Math.max(x0, x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0)));
  };
  const rows = hx === null ? [] : p.series.filter((s) => s.tooltip !== false).map((s) => ({ s, y: interp(s.points, hx) })).filter((r) => r.y !== null);
  const tipLeft = hx === null ? 0 : (sx(hx) / W) * 100;

  return (
    <div className="chart">
      {p.title && <div className="title">{p.title}</div>}
      {p.sub && <div className="sub">{p.sub}</div>}
      {p.legend}
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={typeof p.title === "string" ? p.title : p.yLabel}
        onMouseMove={(e) => setHx(toX(e.clientX))} onMouseLeave={() => setHx(null)}
        onClick={(e) => p.onPickX?.(toX(e.clientX))} style={{ cursor: p.onPickX ? "pointer" : "crosshair" }}>
        {(p.bands ?? []).map((b, i) => (
          <g key={i}>
            <rect x={sx(Math.max(b.x0, x0))} y={m.t} width={Math.max(0, sx(Math.min(b.x1, x1)) - sx(Math.max(b.x0, x0)))} height={H - m.t - m.b} fill="var(--band)" />
            {b.label && <text x={(sx(Math.max(b.x0, x0)) + sx(Math.min(b.x1, x1))) / 2} y={m.t + 14} textAnchor="middle" fontSize={11} fill="var(--ink-2)">{b.label}</text>}
          </g>
        ))}
        {yt.map((v) => (
          <g key={`y${v}`}>
            <line x1={m.l} x2={W - m.r} y1={sy(v)} y2={sy(v)} stroke="var(--grid)" strokeWidth={1} />
            <text x={m.l - 6} y={sy(v) + 4} textAnchor="end" fontSize={11} fill="var(--muted)" style={{ fontVariantNumeric: "tabular-nums" }}>{yf(v)}</text>
          </g>
        ))}
        {xt.map((v) => (
          <text key={`x${v}`} x={sx(v)} y={H - m.b + 16} textAnchor="middle" fontSize={11} fill="var(--muted)" style={{ fontVariantNumeric: "tabular-nums" }}>{xf(v)}</text>
        ))}
        <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke="var(--axis)" />
        <text x={(m.l + W - m.r) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--ink-2)">{p.xLabel}</text>
        <text x={12} y={(m.t + H - m.b) / 2} textAnchor="middle" fontSize={11} fill="var(--ink-2)" transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>{p.yLabel}</text>
        {(p.refY ?? []).map((r, i) => (
          <g key={`r${i}`}>
            <line x1={m.l} x2={W - m.r} y1={sy(r.y)} y2={sy(r.y)} stroke={r.color ?? "var(--ink-2)"} strokeDasharray="5 4" strokeWidth={1.2} />
            <text x={W - m.r - 4} y={sy(r.y) - 5} textAnchor="end" fontSize={11} fill="var(--ink-2)">{r.label}</text>
          </g>
        ))}
        {p.series.map((s) => (
          <path key={s.id} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash} opacity={s.opacity ?? 1} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {(p.markers ?? []).filter((mk) => mk.x >= x0 && mk.x <= x1).map((mk, i) => (
          <g key={`m${i}`}>
            <circle cx={sx(mk.x)} cy={sy(mk.y)} r={5} fill={mk.color} stroke="var(--surface)" strokeWidth={2} />
            {mk.label && <text x={sx(mk.x) + 8} y={sy(mk.y) + 16} fontSize={11} fill="var(--ink-2)">{mk.label}</text>}
          </g>
        ))}
        {p.cursorX !== undefined && p.cursorX >= x0 && p.cursorX <= x1 && (
          <g>
            <line x1={sx(p.cursorX)} x2={sx(p.cursorX)} y1={m.t} y2={H - m.b} stroke="var(--ink)" strokeWidth={1.5} />
            {p.cursorLabel && <text x={sx(p.cursorX) + 5} y={m.t + 12} fontSize={11} fill="var(--ink)">{p.cursorLabel}</text>}
          </g>
        )}
        {hx !== null && <line x1={sx(hx)} x2={sx(hx)} y1={m.t} y2={H - m.b} stroke="var(--axis)" strokeWidth={1} />}
        {rows.map(({ s, y }) => (
          <circle key={`h${s.id}`} cx={sx(hx!)} cy={sy(y!)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
        ))}
      </svg>
      {hx !== null && rows.length > 0 && (
        <div className="tip" style={{ left: `calc(${tipLeft}% + ${tipLeft > 60 ? -12 : 12}px)`, top: 40, transform: tipLeft > 60 ? "translateX(-100%)" : undefined }}>
          <div style={{ color: "var(--ink-2)", marginBottom: 3 }}>{p.xLabel.replace(/\s*\(.*\)/, "")}: <b>{xf(hx)}</b></div>
          {rows.map(({ s, y }) => (
            <div className="row" key={s.id}>
              <svg width="18" height="6" aria-hidden><line x1="0" x2="18" y1="3" y2="3" stroke={s.color} strokeWidth="3" strokeDasharray={s.dash} /></svg>
              <span>{s.name}</span> <b>{yf(y!)}{s.unit ?? ""}</b>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
