import { ReactNode, useState } from "react";
import { niceTicks } from "./LineChart";
import { LETTERS } from "./ui";

export interface BarGroup {
  label: string;
  color: string;
  values: { key: string; v: number; ghost?: boolean; note?: string }[];
}

/** Grouped bars on one axis: one group per route, one bar per scenario (e.g. user equilibrium vs system optimum). */
export function Bars(p: { title: ReactNode; sub?: ReactNode; unit: string; groups: BarGroup[]; format: (v: number) => string; max?: number; refY?: { y: number; label: string } }) {
  const W = 320;
  const H = 210;
  const m = { l: 46, r: 8, t: 14, b: 30 };
  const [hover, setHover] = useState<{ g: number; b: number } | null>(null);
  let vmax = p.max ?? Math.max(1e-9, ...p.groups.flatMap((g) => g.values.map((v) => (Number.isFinite(v.v) ? v.v : 0))));
  if (p.refY) vmax = Math.max(vmax, p.refY.y);
  const ticks = niceTicks(0, vmax * 1.08, 4);
  const top = ticks[ticks.length - 1] || 1;
  const sy = (v: number) => H - m.b - (Math.min(v, top) / top) * (H - m.t - m.b);
  const gw = (W - m.l - m.r) / p.groups.length;
  const nb = Math.max(1, ...p.groups.map((g) => g.values.length));
  const bw = Math.min(34, (gw * 0.62) / nb);
  return (
    <div className="chart">
      <div className="title">{p.title}</div>
      {p.sub && <div className="sub">{p.sub}</div>}
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={typeof p.title === "string" ? p.title : p.unit}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
            <text x={m.l - 6} y={sy(t) + 4} fontSize={10.5} textAnchor="end" fill="var(--muted)" style={{ fontVariantNumeric: "tabular-nums" }}>{p.format(t)}</text>
          </g>
        ))}
        <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke="var(--axis)" />
        {p.refY && (
          <g>
            <line x1={m.l} x2={W - m.r} y1={sy(p.refY.y)} y2={sy(p.refY.y)} stroke="var(--ink-2)" strokeDasharray="5 4" />
            <text x={W - m.r} y={sy(p.refY.y) - 4} fontSize={10.5} textAnchor="end" fill="var(--ink-2)">{p.refY.label}</text>
          </g>
        )}
        {p.groups.map((g, gi) => {
          const cx = m.l + gw * (gi + 0.5);
          const x0 = cx - (g.values.length * bw + (g.values.length - 1) * 2) / 2;
          return (
            <g key={gi}>
              {g.values.map((v, bi) => {
                const x = x0 + bi * (bw + 2);
                const val = Number.isFinite(v.v) ? v.v : 0;
                const y = sy(val);
                const h = Math.max(0, H - m.b - y);
                const r = Math.min(4, h / 2, bw / 2);
                const d = h > 0 ? `M${x},${H - m.b}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${H - m.b}Z` : "";
                return (
                  <g key={bi} onMouseEnter={() => setHover({ g: gi, b: bi })} onMouseLeave={() => setHover(null)}>
                    <rect x={x - 3} y={m.t} width={bw + 6} height={H - m.t - m.b} fill="transparent" />
                    {v.ghost ? (
                      <path d={d} fill="none" stroke={g.color} strokeWidth={2} strokeDasharray="3 2" />
                    ) : (
                      <path d={d} fill={g.color} />
                    )}
                    {h === 0 && <line x1={x} x2={x + bw} y1={H - m.b - 1} y2={H - m.b - 1} stroke={g.color} strokeWidth={2} />}
                    <text x={x + bw / 2} y={y - 4} fontSize={10.5} textAnchor="middle" fill="var(--ink)" style={{ fontVariantNumeric: "tabular-nums" }}>
                      {Number.isFinite(v.v) ? p.format(v.v) : "closed"}
                    </text>
                  </g>
                );
              })}
              <text x={cx} y={H - m.b + 17} fontSize={11.5} fontWeight={600} textAnchor="middle" fill="var(--ink)">{LETTERS[gi] ?? g.label}</text>
            </g>
          );
        })}
      </svg>
      {hover && (
        <div className="tip" style={{ left: `${((m.l + gw * (hover.g + 0.5)) / W) * 100}%`, top: 30, transform: "translateX(-50%)" }}>
          <div><b>{p.groups[hover.g].label}</b> · {p.groups[hover.g].values[hover.b].key}</div>
          <div><b>{p.format(p.groups[hover.g].values[hover.b].v)}</b> {p.unit}</div>
          {p.groups[hover.g].values[hover.b].note && <div style={{ color: "var(--muted)" }}>{p.groups[hover.g].values[hover.b].note}</div>}
        </div>
      )}
    </div>
  );
}
