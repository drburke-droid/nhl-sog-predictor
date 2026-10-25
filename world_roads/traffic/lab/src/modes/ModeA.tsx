import { useMemo } from "react";
import { Assignment, Route, joinsAt, parallelSO, parallelUE, presets, priceOfAnarchy } from "../model/equilibrium";
import { LineChart, Series } from "../components/LineChart";
import { Bars } from "../components/Bars";
import { RouteSchematic } from "../components/RouteSchematic";
import { LETTERS, ROUTE_DASH, ROUTE_VAR, RouteKey, Slider, Term, Tile, fmt, fmtInt } from "../components/ui";
import { CALGARY, NYC, SYMMETRIC, Shared, TEXTBOOK, demandMax, trips } from "../scenarios";

export function RouteControls({ s, set, showDemand = true }: { s: Shared; set: (p: Partial<Shared>) => void; showDemand?: boolean }) {
  const pre = presets(s.routes, s.alpha, s.beta);
  const dmax = demandMax(s.routes, s.alpha, s.beta);
  const setRoute = (i: number, p: Partial<Route>) => set({ routes: s.routes.map((r, k) => (k === i ? { ...r, ...p } : r)) });
  return (
    <>
      {showDemand && (
        <>
          <Slider label="Total trip demand" value={s.demand} min={0} max={dmax} step={50} onChange={(v) => set({ demand: v })} format={(v) => `${fmtInt(v)} veh/h`}
            hint="Vehicles per hour wanting to make this trip, over one steady peak hour." />
          <div className="btnrow" role="group" aria-label="Demand presets">
            <button className="btn" onClick={() => set({ demand: pre.low })} title="Only the fastest route carries traffic">Low traffic</button>
            <button className="btn" onClick={() => set({ demand: pre.moderate })} title="Two routes carry traffic">Moderate</button>
            <button className="btn" onClick={() => set({ demand: pre.heavy })} title="Every route carries traffic, deep in congestion">Heavy</button>
            <button className="btn" onClick={() => set({ demand: pre.near })} title="All three ETAs within a minute, yet the slowest route still carries nobody">Near-converged</button>
          </div>
          <div className="note">Presets are computed from the current routes: low {fmtInt(pre.low)}, moderate {fmtInt(pre.moderate)}, near-converged {fmtInt(pre.near)}, heavy {fmtInt(pre.heavy)} veh/h.</div>
        </>
      )}
      <h3>Routes</h3>
      <div className="btnrow" role="group" aria-label="Route scenarios">
        <button className="btn" onClick={() => set({ routes: TEXTBOOK })}>Textbook</button>
        <button className="btn" onClick={() => set({ routes: NYC, demand: trips.nyc.d_rush })} title="JFK Airport to Times Square: free-flow times from real New York roads at the speed limit; capacities illustrative">New York: JFK → Times Square</button>
        <button className="btn" onClick={() => set({ routes: CALGARY, demand: trips.calgary.d_rush })} title="Rocky Ridge to Seton: free-flow times from real Calgary roads at the speed limit; capacities illustrative">Calgary cross-town</button>
        <button className="btn" onClick={() => set({ routes: SYMMETRIC, demand: 3000 })}>Identical routes</button>
      </div>
      {s.routes.map((r, i) => (
        <div className="route-editor" key={i}>
          <div className="head"><RouteKey i={i} name={r.name} /></div>
          <Slider label="Free-flow time" value={r.t0} min={1} max={60} step={0.1} onChange={(v) => setRoute(i, { t0: v })} format={(v) => `${fmt(v)} min`} />
          <Slider label="Capacity" value={r.cap} min={0} max={5000} step={50} onChange={(v) => setRoute(i, { cap: v })}
            format={(v) => (v > 0 ? `${fmtInt(v)} veh/h` : "closed")} />
        </div>
      ))}
      <details>
        <summary>Advanced: the congestion curve</summary>
        <p className="note">Time = free-flow time × (1 + α (flow ÷ capacity)<sup>β</sup>), the <Term k="BPR">BPR curve</Term>. A stylized curve: no queues, spillback or gridlock.</p>
        <Slider label="α (delay at capacity)" value={s.alpha} min={0} max={1} step={0.01} onChange={(v) => set({ alpha: v })} format={(v) => fmt(v, 2)}
          hint={`At capacity a route is ${fmt(100 * s.alpha, 0)} % slower than when empty.`} />
        <Slider label="β (how sharply it bends)" value={s.beta} min={1} max={10} step={0.5} onChange={(v) => set({ beta: v })} format={(v) => fmt(v, 1)} />
        <button className="btn" onClick={() => set({ alpha: 0.15, beta: 4 })}>Reset to α 0.15, β 4</button>
      </details>
    </>
  );
}

function explainA(s: Shared, ue: Assignment, so: Assignment, joins: number[]): { seeing: string[]; why: string[] } {
  const R = s.routes;
  const used = R.map((_, i) => i).filter((i) => ue.used[i]);
  const unused = R.map((_, i) => i).filter((i) => ue.open[i] && !ue.used[i]);
  const L = (i: number) => LETTERS[i];
  const seeing: string[] = [];
  const why: string[] = [];
  if (s.demand <= 0 || !used.length) {
    seeing.push("Nobody is travelling, so there is no traffic to balance. Each route shows its empty-road (free-flow) time.");
    why.push("Raise the demand and watch which route fills first.");
    return { seeing, why };
  }
  const lam = ue.level;
  if (used.length === 1) {
    const u = used[0];
    seeing.push(`At ${fmtInt(s.demand)} veh/h everyone takes ${L(u)}: even with its traffic it takes ${fmt(lam)} min, still quicker than any other route when empty.`);
  } else if (used.length === R.length) {
    seeing.push(`All ${used.length} routes carry traffic and all take ${fmt(lam)} min: completely different routes, the same travel time.`);
  } else {
    seeing.push(`${used.map(L).join(" and ")} carry traffic and both take ${fmt(lam)} min. If one were quicker, drivers would move to it until it wasn't.`);
  }
  for (const i of unused) {
    const gap = ue.times[i] - lam;
    seeing.push(
      gap < 1
        ? `${L(i)} carries nobody, yet its empty-road time (${fmt(ue.times[i])} min) is only ${fmt(gap)} min slower. An app could list it with almost the same ETA even though nobody uses it.`
        : `${L(i)} carries nobody: even empty it takes ${fmt(ue.times[i])} min. That is allowed at equilibrium; unused routes only need to be no faster. It starts to carry traffic at ${fmtInt(joins[i])} veh/h.`,
    );
  }
  const save = ue.totalVehMin > 0 ? 1 - so.totalVehMin / ue.totalVehMin : 0;
  if (save < 0.002) {
    seeing.push(`A central planner could do almost no better here: the system optimum saves ${fmt(100 * save, 2)} % of total travel time.`);
  } else {
    seeing.push(`A central planner minimizing total time (the system optimum) would average ${fmt(so.avgMin)} min instead of ${fmt(ue.avgMin)}: ${fmt(100 * save, 1)} % less time on the road in total.`);
  }
  why.push("Each driver picks the quickest route for themselves. The quickest route attracts drivers, its traffic slows it down, and its lead shrinks until no one gains by switching.");
  if (used.length > 1) why.push("That is why used routes share one time. It is Wardrop's first principle, a Nash equilibrium of the route-choice game.");
  why.push("The system optimum differs because a driver does not count the delay they add to everyone behind them (their marginal social cost).");
  return { seeing, why };
}

export function ModeA({ s, set }: { s: Shared; set: (p: Partial<Shared>) => void }) {
  const ue = useMemo(() => parallelUE(s.routes, s.demand, s.alpha, s.beta), [s]);
  const so = useMemo(() => parallelSO(s.routes, s.demand, s.alpha, s.beta), [s]);
  const joins = useMemo(() => joinsAt(s.routes, s.alpha, s.beta), [s]);
  const poa = priceOfAnarchy(ue, so);
  const dmax = demandMax(s.routes, s.alpha, s.beta);
  const curves = useMemo(() => {
    const N = 160;
    const ds = Array.from({ length: N + 1 }, (_, k) => (k / N) * dmax);
    // make sure each joining point is sampled exactly, so the kink is sharp
    joins.forEach((j) => { if (j > 0 && j < dmax) ds.push(j); });
    ds.sort((a, b) => a - b);
    const ueR = ds.map((d) => parallelUE(s.routes, d, s.alpha, s.beta));
    const soR = ds.map((d) => parallelSO(s.routes, d, s.alpha, s.beta));
    const series: Series[] = [];
    s.routes.forEach((r, i) => {
      if (!(r.cap > 0)) return;
      const usedPts: [number, number][] = [];
      const idle: [number, number][] = [];
      ds.forEach((d, k) => {
        if (d >= joins[i] - 1e-9) usedPts.push([d, ueR[k].times[i]]);
        if (d <= joins[i] + 1e-9) idle.push([d, ueR[k].times[i]]);
      });
      series.push({ id: `idle${i}`, name: `${LETTERS[i]} (unused)`, color: ROUTE_VAR[i], width: 1.5, opacity: 0.45, dash: "2 5", points: idle, unit: " min" });
      series.push({ id: `used${i}`, name: LETTERS[i], color: ROUTE_VAR[i], dash: ROUTE_DASH[i] || undefined, width: 2.4, points: usedPts, unit: " min" });
    });
    const avg: Series[] = [
      { id: "ue", name: "Selfish (UE) average", color: "var(--ink)", points: ds.map((d, k) => [d, d > 0 ? ueR[k].avgMin : NaN] as [number, number]).filter(([, y]) => Number.isFinite(y)), unit: " min" },
      { id: "so", name: "System optimum average", color: "var(--ink-2)", dash: "6 4", points: ds.map((d, k) => [d, d > 0 ? soR[k].avgMin : NaN] as [number, number]).filter(([, y]) => Number.isFinite(y)), unit: " min" },
    ];
    return { series, avg };
  }, [s, dmax, joins]);
  const ex = explainA(s, ue, so, joins);
  const tmaxBar = Math.max(...ue.times.filter(Number.isFinite), ...(s.showSO ? so.times.filter(Number.isFinite) : []));
  const legend = (
    <div className="legend">
      {s.routes.map((r, i) => (<span key={i}><RouteKey i={i} name={r.name} /></span>))}
      <span><svg width="22" height="6" aria-hidden><line x1="0" x2="22" y1="3" y2="3" stroke="var(--muted)" strokeWidth="1.5" strokeDasharray="2 5" /></svg> faint: route unused (its empty-road time)</span>
    </div>
  );
  return (
    <div className="layout">
      <aside className="side panel">
        <h2>Three competing routes</h2>
        <RouteControls s={s} set={set} />
        <h3>Compare</h3>
        <label style={{ fontSize: 13 }}>
          <input type="checkbox" checked={s.showSO} onChange={(e) => set({ showSO: e.target.checked })} /> Show the <Term k="system optimum" /> in the charts
        </label>
      </aside>
      <section className="main">
        <div className="panel">
          <RouteSchematic routes={s.routes} flows={ue.flows} times={ue.times} used={ue.used} open={ue.open} caption="Exact user equilibrium" />
        </div>
        <div className="tiles">
          <Tile k={<><Term k="equilibrium">Equilibrium</Term> time on used routes</>} v={ue.used.some(Boolean) ? `${fmt(ue.level)} min` : "—"} s={`${ue.used.filter(Boolean).length} of ${s.routes.length} routes carry traffic`} />
          <Tile k="Most anyone could save by switching alone" v={`${fmt(ue.maxGainMin, 2)} min`} s="0 at an exact equilibrium" />
          <Tile k="Average trip: selfish vs. system optimum" v={s.demand > 0 ? `${fmt(ue.avgMin)} vs ${fmt(so.avgMin)}` : "—"} s={s.demand > 0 ? `${fmtInt(ue.totalVehMin)} vs ${fmtInt(so.totalVehMin)} vehicle-min per hour` : "no demand"} />
          <Tile k={<Term k="price of anarchy" />} v={Number.isFinite(poa) ? fmt(poa, 3) : "—"} s={Number.isFinite(poa) ? `selfish routing costs ${fmt(100 * (poa - 1), 2)} % more time` : "undefined without demand"} />
        </div>
        <div className="panel charts2">
          <Bars title="Flow on each route" sub={s.showSO ? "solid: user equilibrium · dashed outline: system optimum" : "user equilibrium"} unit="veh/h" format={fmtInt}
            groups={s.routes.map((r, i) => ({ label: `${LETTERS[i]}: ${r.name}`, color: ROUTE_VAR[i], values: [
              { key: "user equilibrium", v: ue.flows[i], note: ue.used[i] ? undefined : "unused" },
              ...(s.showSO ? [{ key: "system optimum", v: so.flows[i], ghost: true }] : []),
            ] }))} />
          <Bars title="Travel time on each route" sub={`${s.showSO ? "solid: user equilibrium · dashed outline: system optimum" : "user equilibrium"}${ue.used.some(Boolean) ? " · dashed line: the shared equilibrium time" : ""}`} unit="min" format={(v) => fmt(v)} max={tmaxBar}
            refY={ue.used.some(Boolean) ? { y: ue.level, label: "" } : undefined}
            groups={s.routes.map((r, i) => ({ label: `${LETTERS[i]}: ${r.name}`, color: ROUTE_VAR[i], values: [
              { key: "user equilibrium", v: r.cap > 0 ? ue.times[i] : NaN, note: ue.used[i] ? undefined : "unused: empty-road time" },
              ...(s.showSO ? [{ key: "system optimum", v: r.cap > 0 ? so.times[i] : NaN, ghost: true, note: "actual trip time, not marginal cost" }] : []),
            ] }))} />
        </div>
        <div className="explain">
          <h2>What you're seeing</h2>
          {ex.seeing.map((t, i) => <p key={i}>{t}</p>)}
          <h2 style={{ marginTop: 10 }}>Why this happens</h2>
          {ex.why.map((t, i) => <p key={i}>{t}</p>)}
        </div>
        <div className="panel">
          <LineChart title="Travel time on each route as demand grows" sub="Exact user equilibrium at every demand. Click the chart to set the demand." legend={legend}
            series={curves.series} xDomain={[0, dmax]} xLabel="Total demand (veh/h)" yLabel="Travel time (min)" xFormat={fmtInt} yFormat={(v) => fmt(v, 0)}
            cursorX={s.demand} cursorLabel="now" onPickX={(x) => set({ demand: Math.round(x / 50) * 50 })}
            markers={joins.map((j, i) => ({ j, i })).filter(({ j }) => j > 0 && Number.isFinite(j) && j < dmax).map(({ j, i }) => ({ x: j, y: s.routes[i].t0, color: ROUTE_VAR[i], label: `${LETTERS[i]} joins at ${fmtInt(j)}` }))} />
        </div>
        <div className="panel">
          <LineChart title="Average trip: everyone for themselves vs. the best split for the group" sub="The gap is the price of anarchy. With these curves it is often small; Braess's paradox (tab C) is where it gets large."
            legend={<div className="legend"><span><svg width="22" height="6"><line x1="0" x2="22" y1="3" y2="3" stroke="var(--ink)" strokeWidth="2" /></svg>user equilibrium</span><span><svg width="22" height="6"><line x1="0" x2="22" y1="3" y2="3" stroke="var(--ink-2)" strokeWidth="2" strokeDasharray="6 4" /></svg>system optimum</span></div>}
            series={curves.avg} xDomain={[0, dmax]} xLabel="Total demand (veh/h)" yLabel="Average trip (min)" xFormat={fmtInt} yFormat={(v) => fmt(v, 0)} cursorX={s.demand} height={210} />
        </div>
        <p className="note">Toy model: independent parallel routes (real routes often share roads, which couples their times), one steady peak hour, the BPR congestion curve. Values are illustrative, not measurements of New York, Calgary or any city.</p>
      </section>
    </div>
  );
}
