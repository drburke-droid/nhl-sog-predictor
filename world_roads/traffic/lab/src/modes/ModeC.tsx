import { useMemo } from "react";
import { NetworkResult, braessNetwork, networkUE } from "../model/network";
import { LineChart } from "../components/LineChart";
import { Slider, Tile, fmt, fmtInt } from "../components/ui";

export interface CParams {
  connector: boolean;
  demand: number;
  slope: number;
  constant: number;
  connCost: number;
}

export const DEFAULT_C: CParams = { connector: true, demand: 4000, slope: 100, constant: 45, connCost: 0 };

const NODES: Record<string, [number, number]> = { S: [80, 190], A: [380, 50], B: [380, 330], T: [680, 190] };

function linkLabel(l: { a: number; b: number }, slope: number): string {
  if (l.b > 0 && l.a === 0) return `flow ÷ ${fmtInt(slope)} min`;
  if (l.b > 0) return `${fmt(l.a, 0)} + flow ÷ ${fmtInt(slope)} min`;
  return `${fmt(l.a, l.a % 1 ? 1 : 0)} min`;
}

function Diagram({ r, c, net }: { r: NetworkResult; c: CParams; net: ReturnType<typeof braessNetwork> }) {
  const all = braessNetwork(true, c.demand, c.slope, c.constant, c.connCost).links;
  const maxF = Math.max(1, c.demand);
  return (
    <svg viewBox="0 0 760 390" role="img" aria-label="Four-node network S, A, B, T with an optional connector from A to B">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0L10,5L0,10z" fill="var(--ink-2)" />
        </marker>
      </defs>
      {all.map((l, k) => {
        const [x0, y0] = NODES[l.from];
        const [x1, y1] = NODES[l.to];
        const isConn = l.from === "A" && l.to === "B";
        const idx = net.links.findIndex((q) => q.from === l.from && q.to === l.to);
        const on = idx >= 0;
        const f = on ? r.linkFlows[idx] : 0;
        const t = on ? r.linkTimes[idx] : NaN;
        const dx = x1 - x0;
        const dy = y1 - y0;
        const L = Math.hypot(dx, dy);
        const ux = dx / L;
        const uy = dy / L;
        const sx = x0 + ux * 26;
        const sy = y0 + uy * 26;
        const ex = x1 - ux * 30;
        const ey = y1 - uy * 30;
        const mx = (x0 + x1) / 2;
        const my = (y0 + y1) / 2;
        // labels run along each link, on the side away from the diamond's centre
        let px = -uy;
        let py = ux;
        if (px * (mx - 380) + py * (my - 190) < 0) { px = -px; py = -py; }
        let ang = (Math.atan2(dy, dx) * 180) / Math.PI;
        if (ang > 90) ang -= 180;
        if (ang < -90) ang += 180;
        const below = py > 0;
        return (
          <g key={k}>
            <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="var(--road)" strokeWidth={isConn ? 16 : 22} strokeLinecap="round" opacity={on ? 1 : 0.5} />
            {on && f > 0.5 && <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="var(--ink)" strokeWidth={2 + 10 * (f / maxF)} strokeLinecap="round" opacity={0.85} />}
            <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={on ? "var(--ink-2)" : "var(--muted)"} strokeWidth={1.4} strokeDasharray={on ? undefined : "5 5"} markerEnd="url(#arr)" />
            {isConn ? (
              <g>
                <rect x={mx + 14} y={my - 34} width={176} height={68} rx={10} fill="var(--surface)" stroke="var(--ring)" />
                <text x={mx + 26} y={my - 13} fontSize={13} fontWeight={650} fill="var(--ink)">{on ? "Connector OPEN" : "Connector CLOSED"}</text>
                <text x={mx + 26} y={my + 5} fontSize={12} fill="var(--ink-2)">A → B, one way, {linkLabel(l, c.slope)}</text>
                <text x={mx + 26} y={my + 22} fontSize={12} fill="var(--ink-2)" style={{ fontVariantNumeric: "tabular-nums" }}>{on ? `${fmtInt(f)} veh/h` : "no traffic"}</text>
                {!on && <text x={mx} y={my + 7} textAnchor="middle" fontSize={22} fontWeight={700} fill="var(--muted)">✕</text>}
              </g>
            ) : (
              <g transform={`translate(${mx + px * (below ? 22 : 36)},${my + py * (below ? 22 : 36)}) rotate(${ang})`}>
                <text textAnchor="middle" fontSize={12.5} fill="var(--ink-2)">{linkLabel(l, c.slope)}</text>
                <text y={16} textAnchor="middle" fontSize={12.5} fontWeight={650} fill="var(--ink)" style={{ fontVariantNumeric: "tabular-nums" }}>
                  {fmtInt(f)} veh/h · {fmt(t)} min
                </text>
              </g>
            )}
          </g>
        );
      })}
      {Object.entries(NODES).map(([n, [x, y]]) => (
        <g key={n}>
          <circle cx={x} cy={y} r={22} fill="var(--ink)" />
          <text x={x} y={y + 6} textAnchor="middle" fontSize={17} fontWeight={700} fill="var(--surface)">{n}</text>
        </g>
      ))}
      <text x={NODES.S[0]} y={NODES.S[1] + 42} textAnchor="middle" fontSize={12} fill="var(--muted)">start</text>
      <text x={NODES.T[0]} y={NODES.T[1] + 42} textAnchor="middle" fontSize={12} fill="var(--muted)">destination</text>
    </svg>
  );
}

export function ModeC({ c, setC }: { c: CParams; setC: (p: Partial<CParams>) => void }) {
  const netOpen = braessNetwork(true, c.demand, c.slope, c.constant, c.connCost);
  const netClosed = braessNetwork(false, c.demand, c.slope, c.constant, c.connCost);
  const ro = useMemo(() => networkUE(netOpen), [c.demand, c.slope, c.constant, c.connCost]); // eslint-disable-line
  const rc = useMemo(() => networkUE(netClosed), [c.demand, c.slope, c.constant, c.connCost]); // eslint-disable-line
  const r = c.connector ? ro : rc;
  const net = c.connector ? netOpen : netClosed;
  const dmax = 12000;
  const curve = useMemo(() => {
    const ds = Array.from({ length: 241 }, (_, k) => (k / 240) * dmax);
    const o = ds.map((d) => networkUE(braessNetwork(true, d, c.slope, c.constant, c.connCost)));
    const cl = ds.map((d) => networkUE(braessNetwork(false, d, c.slope, c.constant, c.connCost)));
    const bands: { x0: number; x1: number; label?: string }[] = [];
    let b0: number | null = null;
    ds.forEach((d, k) => {
      const worse = d > 0 && o[k].avgMin > cl[k].avgMin + 1e-6;
      if (worse && b0 === null) b0 = d;
      if ((!worse || k === ds.length - 1) && b0 !== null) {
        bands.push({ x0: b0, x1: worse ? d : ds[k - 1], label: "connector makes everyone slower" });
        b0 = null;
      }
    });
    return {
      series: [
        { id: "closed", name: "Connector closed", color: "var(--ink-2)", dash: "6 4", unit: " min", points: ds.slice(1).map((d, k) => [d, cl[k + 1].avgMin] as [number, number]) },
        { id: "open", name: "Connector open", color: "var(--ink)", unit: " min", points: ds.slice(1).map((d, k) => [d, o[k + 1].avgMin] as [number, number]) },
      ],
      bands,
    };
  }, [c.slope, c.constant, c.connCost]);
  const dAvg = ro.avgMin - rc.avgMin;
  const order = ["S-A-T", "S-B-T", "S-A-B-T"];
  const rows = order.map((name) => {
    const i = r.paths.indexOf(name);
    if (i < 0) return { name, flow: 0, time: NaN, used: false, exists: false };
    return { name, flow: r.pathFlows[i], time: r.pathTimes[i], used: r.used[i], exists: true };
  });
  const isClassic = c.demand === 4000 && c.slope === 100 && c.constant === 45 && c.connCost === 0;
  const seeing: string[] = [];
  if (c.demand <= 0) seeing.push("No drivers, no paradox. Raise the number of drivers.");
  else if (c.connector) {
    const sabt = rows[2];
    if (sabt.used && rows[0].used) seeing.push(`With the connector open, drivers split three ways and every used path takes ${fmt(sabt.time)} min.`);
    else if (sabt.used) seeing.push(`With the connector open, all ${fmtInt(c.demand)} drivers take S-A-B-T: ${fmt(sabt.time)} min. A lone driver switching to S-A-T or S-B-T would take ${fmt(rows[0].time)} min, so nobody switches.`);
    else seeing.push(`Even with the connector open nobody uses it: at this demand the two outer roads are quicker.`);
    if (dAvg > 1e-6) seeing.push(`Closing the connector would cut everyone's trip by ${fmt(dAvg)} min (to ${fmt(rc.avgMin)} min). The extra road makes everyone slower.`);
    else if (dAvg < -1e-6) seeing.push(`Here the connector genuinely helps: trips average ${fmt(ro.avgMin)} min with it, ${fmt(rc.avgMin)} min without.`);
    else seeing.push("With or without the connector, trips take the same time at this demand.");
  } else {
    seeing.push(`With the connector closed, drivers split evenly between S-A-T and S-B-T (${fmtInt(rows[0].flow)} each) and both take ${fmt(rows[0].time)} min.`);
    seeing.push(dAvg > 1e-6 ? `Open it and the selfish equilibrium becomes ${fmt(ro.avgMin)} min for everyone: ${fmt(dAvg)} min worse.` : dAvg < -1e-6 ? `Opening it would help here: ${fmt(ro.avgMin)} min.` : "Opening it would change nothing at this demand.");
  }
  if (isClassic) seeing.push("These are the classic textbook numbers: 65 minutes without the connector, 80 with it.");
  return (
    <div className="layout">
      <aside className="side panel">
        <h2>Braess's paradox</h2>
        <button className="toggle-big" aria-pressed={c.connector} onClick={() => setC({ connector: !c.connector })}>
          <span className="knob" aria-hidden />
          {c.connector ? "Connector A → B is OPEN" : "Connector A → B is CLOSED"}
        </button>
        <Slider label="Drivers" value={c.demand} min={0} max={dmax} step={100} onChange={(v) => setC({ demand: v })} format={(v) => `${fmtInt(v)} veh/h`} />
        <details>
          <summary>Advanced: link times</summary>
          <Slider label="S→A and B→T: minutes = flow ÷" value={c.slope} min={20} max={400} step={10} onChange={(v) => setC({ slope: v })} format={(v) => fmtInt(v)} />
          <Slider label="A→T and S→B: constant" value={c.constant} min={0} max={120} step={1} onChange={(v) => setC({ constant: v })} format={(v) => `${v} min`} />
          <Slider label="Connector A→B: constant" value={c.connCost} min={0} max={60} step={0.5} onChange={(v) => setC({ connCost: v })} format={(v) => `${fmt(v)} min`} />
        </details>
        <div className="btnrow"><button className="btn" onClick={() => setC({ ...DEFAULT_C, connector: c.connector })}>Classic numbers (4,000 drivers)</button></div>
        <p className="note">Toy link costs; flow is vehicles in the model's common period. Equilibrium found by path equilibration (tested: 65 and 80 min).</p>
      </aside>
      <section className="main">
        <div className="panel"><Diagram r={r} c={c} net={net} /></div>
        <div className="tiles">
          <Tile k="Average trip, connector closed" v={c.demand > 0 ? `${fmt(rc.avgMin)} min` : "—"} s={`${fmtInt(rc.totalVehMin)} vehicle-min in total`} />
          <Tile k="Average trip, connector open" v={c.demand > 0 ? `${fmt(ro.avgMin)} min` : "—"} s={`${fmtInt(ro.totalVehMin)} vehicle-min in total`} />
          <Tile k="Effect of opening the connector" v={c.demand > 0 ? `${dAvg > 0 ? "+" : ""}${fmt(dAvg)} min` : "—"} s={dAvg > 1e-6 ? "everyone is slower" : dAvg < -1e-6 ? "everyone is quicker" : "no change"} />
        </div>
        <div className="panel">
          <table className="paths">
            <thead><tr><th>Path</th><th>Drivers</th><th>Travel time</th><th>Status</th></tr></thead>
            <tbody>
              {rows.map((x) => (
                <tr key={x.name}>
                  <td>{x.name}</td>
                  <td>{x.exists ? fmtInt(x.flow) : "—"}</td>
                  <td>{x.exists ? `${fmt(x.time)} min` : "—"}</td>
                  <td>{!x.exists ? <span className="pill closed">needs the connector</span> : x.used ? <span className="pill used">used</span> : <span className="pill unused">unused · no faster</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="explain">
          <h2>What you're seeing</h2>
          {seeing.map((t, i) => <p key={i}>{t}</p>)}
          <h2 style={{ marginTop: 10 }}>Why this happens</h2>
          <p>The connector is free, so each driver gains by using it, given what the others do. But every driver who switches adds traffic to S→A and B→T, the two roads whose time grows with flow. At the new equilibrium nobody can do better alone, yet everyone does worse together. Selfish choices, a bad result for the group.</p>
          <p>This shows a paradox is possible, not that new roads usually make traffic worse. Below, it happens only in a band of demand. Induced demand (new roads attracting new trips) is a different effect.</p>
        </div>
        <div className="panel">
          <LineChart title="Average trip with and without the connector" sub="Shaded: the demands at which opening the connector slows everyone down. Click to set the number of drivers."
            legend={<div className="legend"><span><svg width="22" height="6"><line x1="0" x2="22" y1="3" y2="3" stroke="var(--ink)" strokeWidth="2" /></svg>open</span><span><svg width="22" height="6"><line x1="0" x2="22" y1="3" y2="3" stroke="var(--ink-2)" strokeWidth="2" strokeDasharray="6 4" /></svg>closed</span></div>}
            series={curve.series} bands={curve.bands} xDomain={[0, dmax]} xLabel="Drivers (veh/h)" yLabel="Average trip (min)" xFormat={fmtInt} yFormat={(v) => fmt(v, 0)}
            cursorX={c.demand} onPickX={(x) => setC({ demand: Math.round(x / 100) * 100 })} />
        </div>
      </section>
    </div>
  );
}
