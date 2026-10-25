import { useMemo, useState } from "react";
import { appShown, randomRoutes, selectionStats } from "../model/app";
import { Slider, Tile, fmt } from "../components/ui";
import { trips } from "../scenarios";

export interface DParams {
  source: "random" | "nyc" | "calgary";
  n: number;
  spread: number;
  seed: number;
  k: number;
  detour: number; // 0 = off, else percent
  round: boolean;
}

export const DEFAULT_D: DParams = { source: "random", n: 20, spread: 1, seed: 3, k: 3, detour: 0, round: true };

function Strip({ times, shown, round }: { times: number[]; shown: number[]; round: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 760;
  const H = 124;
  const lo = Math.floor(Math.min(...times) - 1);
  const hi = Math.ceil(Math.max(...times) + 1);
  const sx = (t: number) => 30 + ((t - lo) / (hi - lo)) * (W - 60);
  const sorted = times.map((t, i) => ({ t, i })).sort((a, b) => a.t - b.t);
  const stack = new Map<number, number>();
  const pos = new Map<number, [number, number]>();
  for (const { t, i } of sorted) {
    const bin = Math.round(sx(t) / 11);
    const k = stack.get(bin) ?? 0;
    stack.set(bin, k + 1);
    pos.set(i, [sx(t), 70 - k * 11]);
  }
  const step = hi - lo > 40 ? 10 : hi - lo > 15 ? 5 : 2;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  const shownSet = new Set(shown);
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Every candidate route's travel time; the ones the app shows are highlighted">
        <line x1={30} x2={W - 30} y1={80} y2={80} stroke="var(--axis)" />
        {ticks.map((v) => (
          <g key={v}>
            <line x1={sx(v)} x2={sx(v)} y1={80} y2={84} stroke="var(--axis)" />
            <text x={sx(v)} y={99} textAnchor="middle" fontSize={12} fill="var(--muted)">{v} min</text>
          </g>
        ))}
        {times.map((_, i) => {
          const [x, y] = pos.get(i)!;
          const on = shownSet.has(i);
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <circle cx={x} cy={y} r={9} fill="transparent" />
              <circle cx={x} cy={y} r={on ? 6.5 : 4.8} fill={on ? "var(--ink)" : "var(--surface)"} stroke={on ? "var(--surface)" : "var(--muted)"} strokeWidth={on ? 2 : 1.4} />
            </g>
          );
        })}
        {shown.length > 0 && (
          <g>
            <line x1={sx(times[shown[0]])} x2={sx(times[shown[shown.length - 1]])} y1={114} y2={114} stroke="var(--ink)" strokeWidth={2.5} />
            <text x={sx(times[shown[shown.length - 1]]) + 6} y={118} fontSize={12} fill="var(--ink)">listed by the app</text>
          </g>
        )}
      </svg>
      {hover !== null && (
        <div className="tip" style={{ left: `${(pos.get(hover)![0] / W) * 100}%`, top: 0, transform: "translateX(-50%)" }}>
          {fmt(times[hover])} min{round ? ` (shown as ${Math.round(times[hover])} min)` : ""} · {shownSet.has(hover) ? "listed by the app" : "not listed"}
        </div>
      )}
    </div>
  );
}

export function ModeD({ d, setD }: { d: DParams; setD: (p: Partial<DParams>) => void }) {
  const trip = d.source === "random" ? null : trips[d.source];
  const times = useMemo(() => (trip ? trip.via_sample : randomRoutes(d.n, 24, d.spread, d.seed)), [trip, d.n, d.spread, d.seed]);
  const shown = appShown(times, d.k, d.detour > 0 ? d.detour / 100 : null);
  const disp = shown.map((i) => (d.round ? Math.round(times[i]) : times[i]));
  const spreadShown = disp.length ? Math.max(...disp) - Math.min(...disp) : 0;
  const spreadAll = Math.max(...times) - Math.min(...times);
  const stats = useMemo(() => selectionStats(1000, d.n, 24, d.spread, d.k, d.round, 1), [d.n, d.spread, d.k, d.round]);
  return (
    <div className="layout">
      <aside className="side panel">
        <h2>What the app shows</h2>
        <p className="note">A second, separate reason ETAs look alike: an app lists only a few of the quickest routes. This tab has no traffic model at all.</p>
        <div className="btnrow" role="group" aria-label="Candidate routes">
          <button className="btn" aria-pressed={d.source === "random"} onClick={() => setD({ source: "random" })}>Random city</button>
          <button className="btn" aria-pressed={d.source === "nyc"} onClick={() => setD({ source: "nyc" })}>New York at 3 a.m.</button>
          <button className="btn" aria-pressed={d.source === "calgary"} onClick={() => setD({ source: "calgary" })}>Calgary at 3 a.m.</button>
        </div>
        {!trip ? (
          <>
            <Slider label="Possible routes" value={d.n} min={3} max={100} step={1} onChange={(v) => setD({ n: v })} format={String} />
            <Slider label="How much slower the worst is" value={d.spread} min={0.05} max={2} step={0.05} onChange={(v) => setD({ spread: v })} format={(v) => `up to +${fmt(100 * v, 0)} %`}
              hint="Route times are drawn uniformly between 24 min and this much slower. No traffic, no equilibrium." />
            <Slider label="Random seed" value={d.seed} min={1} max={50} step={1} onChange={(v) => setD({ seed: v })} format={String} />
          </>
        ) : (
          <p className="note">{trip.via_sample.length} routes from {trip.origin} to {trip.dest} ({trip.name}), each forced through a different random junction on a major road: the fastest way to the junction, then the fastest way on. Free-flow times at the speed limit (no traffic, no lights), from OpenStreetMap data.</p>
        )}
        <h3>The app</h3>
        <Slider label="Routes it lists" value={d.k} min={1} max={5} step={1} onChange={(v) => setD({ k: v })} format={String} />
        <Slider label="Hide routes slower than the fastest by more than" value={d.detour} min={0} max={100} step={5} onChange={(v) => setD({ detour: v })} format={(v) => (v ? `${v} %` : "off")} />
        <label style={{ fontSize: 13 }}><input type="checkbox" checked={d.round} onChange={(e) => setD({ round: e.target.checked })} /> Round ETAs to whole minutes</label>
        <p className="note">A stylized filter. Real apps also drop routes that mostly duplicate each other, and their ETAs are predictions with error, not measurements.</p>
      </aside>
      <section className="main">
        <div className="panel" style={{ display: "flex", flexWrap: "wrap", gap: 18, alignItems: "flex-start" }}>
          <div className="phone" aria-label="What the app lists" style={{ flex: "none" }}>
            <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>Suggested routes</div>
            {shown.map((i, k) => (
              <div className="sug" key={i}>
                <span>Route {k + 1}</span>
                <span className="eta">{d.round ? `${Math.round(times[i])} min` : `${fmt(times[i])} min`}</span>
              </div>
            ))}
          </div>
          <div className="tiles" style={{ flex: 1, minWidth: 260, gridTemplateColumns: "1fr" }}>
          <Tile k="Spread of the listed ETAs" v={`${fmt(spreadShown, d.round ? 0 : 1)} min`} s={`${shown.length} listed`} />
          <Tile k="Spread of all candidate routes" v={`${fmt(spreadAll, 1)} min`} s={`${times.length} candidates`} />
          {d.source === "random" && (
            <Tile k="Across 1,000 random cities like this" v={`${fmt(stats.shownSpread, 1)} vs ${fmt(stats.allSpread, 1)} min`} s={`listed vs all; every listed ETA within 1 min in ${fmt(100 * stats.shownWithin1, 0)} %`} />
          )}
          </div>
        </div>
        <div className="panel">
          <div className="title" style={{ fontWeight: 600 }}>Every possible route (open dots) and the ones the app lists (filled)</div>
          <Strip times={times} shown={shown} round={d.round} />
        </div>
        <div className="explain">
          <h2>What you're seeing</h2>
          <p>
            The {shown.length} listed routes span {fmt(spreadShown, d.round ? 0 : 1)} min, while all {times.length} candidates span {fmt(spreadAll, 1)} min.
            {" "}Picking the quickest few out of many always picks routes that are close to each other: they are the bunched-up front of the pack.
          </p>
          <p>This has nothing to do with traffic. Even on empty roads, an app that lists the quickest few will show similar ETAs.</p>
          <h2 style={{ marginTop: 10 }}>Why this matters</h2>
          <p>Two effects stack. Traffic equilibrium (tab A) pulls the times of routes people actually use together. The app's filter hides everything else. Seeing three similar ETAs does not, by itself, tell you which effect you are looking at.</p>
          <p className="note">With n routes spread evenly over a range R, the gap between the fastest and the third fastest averages about 2R/(n+1): with 20 routes over 24 minutes, about 2.3 min, before any traffic at all.</p>
        </div>
      </section>
    </div>
  );
}
