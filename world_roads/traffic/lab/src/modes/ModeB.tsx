import { useEffect, useMemo, useRef, useState } from "react";
import { parallelUE, summarize } from "../model/equilibrium";
import { defaultStart, learning } from "../model/learning";
import { LineChart, Series } from "../components/LineChart";
import { RouteSchematic } from "../components/RouteSchematic";
import { LETTERS, ROUTE_DASH, ROUTE_VAR, RouteKey, Slider, Tile, fmt, fmtInt } from "../components/ui";
import { RouteControls } from "./ModeA";
import { Shared, demandMax } from "../scenarios";

export interface BParams {
  replan: number;
  delay: number;
  theta: number;
  sigma: number;
  noise: number;
  seed: number;
  start: "fastest" | "even" | "slowest";
  days: number;
  speed: number;
}

export const DEFAULT_B: BParams = { replan: 0.2, delay: 1, theta: 0.5, sigma: 0, noise: 0, seed: 1, start: "fastest", days: 120, speed: 8 };

const B_PRESETS: { label: string; title: string; p: Partial<BParams> }[] = [
  { label: "Settles", title: "20 % re-plan, ignore savings under half a minute", p: { replan: 0.2, delay: 1, theta: 0.5, sigma: 0, noise: 0 } },
  { label: "Keeps wobbling", title: "20 % re-plan, every saving counts", p: { replan: 0.2, delay: 1, theta: 0, sigma: 0, noise: 0 } },
  { label: "Everyone chases yesterday's winner", title: "100 % re-plan on yesterday's times", p: { replan: 1, delay: 1, theta: 0, sigma: 0, noise: 0 } },
  { label: "Ignore under 2 min", title: "Nobody switches to save less than 2 minutes", p: { replan: 0.2, delay: 1, theta: 2, sigma: 0, noise: 0, start: "slowest" } },
  { label: "Fuzzy perceptions + noise", title: "Perceived times differ by drivers; day-to-day noise", p: { replan: 0.3, delay: 1, theta: 0, sigma: 1.5, noise: 0.1 } },
  { label: "Stale information", title: "Drivers act on times from 3 days ago", p: { replan: 0.3, delay: 3, theta: 0, sigma: 0, noise: 0 } },
];

export function ModeB({ s, set, b, setB }: { s: Shared; set: (p: Partial<Shared>) => void; b: BParams; setB: (p: Partial<BParams>) => void }) {
  const [day, setDay] = useState(0);
  const [playing, setPlaying] = useState(false);
  const start = useMemo(() => {
    const n = s.routes.length;
    if (b.start === "even") return new Array(n).fill(1);
    if (b.start === "slowest") {
      const k = s.routes.reduce((m, r, i) => (r.cap > 0 && r.t0 > s.routes[m].t0 ? i : m), 0);
      return s.routes.map((_, i) => (i === k ? 1 : 0));
    }
    return defaultStart(s.routes);
  }, [s.routes, b.start]);
  const run = useMemo(
    () => learning(s.routes, s.demand, { days: b.days, start, replan: b.replan, delay: b.delay, theta: b.theta, sigma: b.sigma, noise: b.noise, seed: b.seed, alpha: s.alpha, beta: s.beta }),
    [s, b.days, start, b.replan, b.delay, b.theta, b.sigma, b.noise, b.seed],
  );
  const ue = useMemo(() => parallelUE(s.routes, s.demand, s.alpha, s.beta), [s]);
  const d = Math.min(day, b.days);
  const last = useRef(0);
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    last.current = performance.now();
    let acc = 0;
    const tick = (now: number) => {
      acc += ((now - last.current) / 1000) * b.speed;
      last.current = now;
      if (acc >= 1) {
        const k = Math.floor(acc);
        acc -= k;
        setDay((x) => {
          const nx = Math.min(b.days, x + k);
          if (nx >= b.days) setPlaying(false);
          return nx;
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, b.speed, b.days]);

  const sm = summarize(s.routes, run.flows[d], s.alpha, s.beta);
  const seen = run.flows.slice(0, d + 1);
  const etaSeries: Series[] = s.routes.map((_, i) => ({
    id: `t${i}`, name: LETTERS[i], color: ROUTE_VAR[i], dash: ROUTE_DASH[i] || undefined, unit: " min",
    points: seen.map((f, k) => [k, f[i] > 0.5 ? run.times[k][i] : NaN] as [number, number]).filter(([, y]) => Number.isFinite(y)),
  }));
  const shareSeries: Series[] = s.routes.map((_, i) => ({
    id: `f${i}`, name: LETTERS[i], color: ROUTE_VAR[i], dash: ROUTE_DASH[i] || undefined, unit: " %",
    points: seen.map((f, k) => [k, s.demand > 0 ? (100 * f[i]) / s.demand : 0] as [number, number]),
  }));
  const tAll = run.times.flatMap((T, k) => T.filter((t, i) => Number.isFinite(t) && run.flows[k][i] > 0.5));
  const tmax = Math.min(Math.max(...tAll, ue.level || 0) * 1.05, Math.max(ue.level || 0, ...s.routes.map((r) => r.t0)) * 4);
  const tail = run.avgGain.slice(-25);
  const amp = Math.max(...tail);
  const swing = Math.max(...tail) - Math.min(...tail);
  const explain: string[] = [];
  if (s.demand <= 0) explain.push("No demand, so there is nothing to learn. Raise the demand on the left.");
  else {
    explain.push(`Day ${d}: ${s.routes.map((_, i) => `${LETTERS[i]} ${fmt((100 * run.flows[d][i]) / s.demand, 0)} %`).join(", ")} of drivers. Each day ${fmt(100 * b.replan, 0)} % of drivers reconsider, using the times from ${b.delay === 1 ? "yesterday" : `${b.delay} days ago`}.`);
    if (b.replan >= 0.6 && b.sigma < 0.5 && b.theta < 0.5)
      explain.push("So many drivers jump to yesterday's quickest route that it becomes today's jam. The traffic sloshes back and forth and never settles: overshooting.");
    else if (amp < 0.02) explain.push(b.theta > 0 ? `The traffic has settled. Drivers ignore savings under ${fmt(b.theta)} min, so the routes can freeze with times up to ${fmt(b.theta)} min apart instead of exactly equal.` : "The traffic has settled at (or very near) the exact equilibrium: the dashed line.");
    else if (swing < 0.05 && b.sigma > 0) explain.push(`It has settled, but not at the exact equilibrium. Drivers perceive times differently (spread ${fmt(b.sigma)} min), so some keep using routes that are slower on average: a "stochastic" equilibrium.`);
    else if (amp < 1) explain.push("Close to the equilibrium, but each day's switchers overshoot a little, so the times keep wobbling around the dashed line.");
    else explain.push(`The times keep swinging: in the last 25 days the average driver could have saved up to ${fmt(amp)} min by switching. Each day's switchers all pick the same route and overshoot. Try fewer drivers re-planning, a switching threshold, or perception spread.`);
    if (b.noise > 0) explain.push(`Observed times carry day-to-day noise (±${fmt(100 * b.noise, 0)} %, seeded), like weather or incidents, so even a calm system keeps jittering.`);
  }
  const dmax = demandMax(s.routes, s.alpha, s.beta);
  return (
    <div className="layout">
      <aside className="side panel">
        <h2>Watch drivers find equilibrium</h2>
        <p className="note">A behavioural sketch, not an equilibrium solver and not a model of any real navigation app. The dashed line is the exact equilibrium from tab A.</p>
        <div className="btnrow">
          <button className="btn primary" onClick={() => { if (d >= b.days) setDay(0); setPlaying(!playing); }}>{playing ? "Pause" : "Start"}</button>
          <button className="btn" onClick={() => { setPlaying(false); setDay((x) => Math.min(b.days, x + 1)); }}>Step one day</button>
          <button className="btn" onClick={() => { setPlaying(false); setDay(0); }}>Reset</button>
        </div>
        <div className="btnrow" role="group" aria-label="Behaviour presets">
          {B_PRESETS.map((x) => (
            <button key={x.label} className="btn" title={x.title} onClick={() => { setB(x.p); setDay(0); setPlaying(true); }}>{x.label}</button>
          ))}
        </div>
        <Slider label="Speed" value={b.speed} min={1} max={40} step={1} onChange={(v) => setB({ speed: v })} format={(v) => `${v} days/s`} />
        <Slider label="Day" value={d} min={0} max={b.days} step={1} onChange={(v) => { setPlaying(false); setDay(v); }} format={(v) => `${v} of ${b.days}`} />
        <h3>Drivers</h3>
        <Slider label="Share who reconsider each day" value={b.replan} min={0.01} max={1} step={0.01} onChange={(v) => setB({ replan: v })} format={(v) => `${fmt(100 * v, 0)} %`} />
        <Slider label="Reaction delay" value={b.delay} min={1} max={5} step={1} onChange={(v) => setB({ delay: v })} format={(v) => `${v} day${v > 1 ? "s" : ""}`} hint="They act on times observed this many days earlier." />
        <Slider label="Switching threshold" value={b.theta} min={0} max={5} step={0.1} onChange={(v) => setB({ theta: v })} format={(v) => `${fmt(v)} min`} hint="Nobody switches to save less than this." />
        <Slider label="Perception spread" value={b.sigma} min={0} max={5} step={0.1} onChange={(v) => setB({ sigma: v })} format={(v) => `${fmt(v)} min`} hint="0: everyone sees the same times. Higher: choices get fuzzier (logit)." />
        <Slider label="Day-to-day noise" value={b.noise} min={0} max={0.3} step={0.01} onChange={(v) => setB({ noise: v })} format={(v) => `±${fmt(100 * v, 0)} %`} />
        <Slider label="Random seed" value={b.seed} min={1} max={50} step={1} onChange={(v) => setB({ seed: v })} format={(v) => String(v)} hint="Same seed, same run." />
        <label className="note">Day 0 assignment{" "}
          <select value={b.start} onChange={(e) => setB({ start: e.target.value as BParams["start"] })}>
            <option value="fastest">90 % on the fastest empty road</option>
            <option value="even">split evenly</option>
            <option value="slowest">everyone on the slowest road</option>
          </select>
        </label>
        <Slider label="Days to simulate" value={b.days} min={20} max={400} step={10} onChange={(v) => setB({ days: v })} format={(v) => String(v)} />
        <details>
          <summary>Routes and demand (shared with tab A)</summary>
          <Slider label="Total trip demand" value={s.demand} min={0} max={dmax} step={50} onChange={(v) => set({ demand: v })} format={(v) => `${fmtInt(v)} veh/h`} />
          <RouteControls s={s} set={set} showDemand={false} />
        </details>
      </aside>
      <section className="main">
        <div className="panel">
          <RouteSchematic routes={s.routes} flows={run.flows[d]} times={run.times[d]} used={sm.used} open={sm.open} caption={`Day ${d} of the learning process`} />
        </div>
        <div className="tiles">
          <Tile k="Day" v={String(d)} s={playing ? "running" : "paused"} />
          <Tile k="Most anyone could save by switching" v={`${fmt(sm.maxGainMin, 2)} min`} s="0 at an exact equilibrium" />
          <Tile k="Average saving available" v={`${fmt(sm.avgGainMin, 2)} min`} s="average trip minus the quickest route's time" />
          <Tile k="Average trip today" v={s.demand > 0 ? `${fmt(sm.avgMin)} min` : "—"} s={ue.used.some(Boolean) ? `exact equilibrium: ${fmt(ue.level)} min` : undefined} />
        </div>
        <div className="panel">
          <LineChart title="Travel time on each route, day by day" sub="Only routes that carry traffic are drawn. Dashed: the exact user-equilibrium time."
            legend={<div className="legend">{s.routes.map((r, i) => <span key={i}><RouteKey i={i} name={r.name} /></span>)}</div>}
            series={etaSeries} xDomain={[0, b.days]} yDomain={[0, Math.max(1, tmax)]} xLabel="Day" yLabel="Travel time (min)" yFormat={(v) => fmt(v, 0)}
            refY={ue.used.some(Boolean) ? [{ y: ue.level, label: `equilibrium ${fmt(ue.level)} min` }] : []} cursorX={d} onPickX={(x) => { setPlaying(false); setDay(Math.round(x)); }} />
        </div>
        <div className="panel">
          <LineChart title="Share of drivers on each route" series={shareSeries} xDomain={[0, b.days]} yDomain={[0, 100]} xLabel="Day" yLabel="Share of drivers (%)" yFormat={(v) => fmt(v, 0)}
            refY={s.demand > 0 ? s.routes.map((_, i) => ({ i })).filter(({ i }) => ue.used[i]).map(({ i }) => ({ y: (100 * ue.flows[i]) / s.demand, label: `${LETTERS[i]} at equilibrium`, color: ROUTE_VAR[i] })) : []}
            cursorX={d} height={210} onPickX={(x) => { setPlaying(false); setDay(Math.round(x)); }} />
        </div>
        <div className="explain">
          <h2>What you're seeing</h2>
          {explain.map((t, i) => <p key={i}>{t}</p>)}
          <h2 style={{ marginTop: 10 }}>Why this happens</h2>
          <p>Nobody computes the equilibrium. Drivers just try routes and remember what happened. If a few switch at a time, the routes' times are pushed together. If everyone reacts to the same stale information at once, they all pile onto the same road and overshoot.</p>
          <p>Real traffic is messier: different trips overlap, conditions change through the day, and only some drivers react to live directions.</p>
        </div>
      </section>
    </div>
  );
}
