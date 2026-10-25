// User equilibrium on a small link network by pairwise path equilibration. Mirrors model.py: network_ue.

export interface Link {
  from: string;
  to: string;
  a: number; // constant time, min
  b: number; // min per (veh/h) of flow: time = a + b * flow
}

export interface Network {
  links: Link[];
  origin: string;
  dest: string;
  demand: number;
}

export interface NetworkResult {
  paths: string[];
  pathLinks: number[][];
  pathFlows: number[];
  pathTimes: number[];
  linkFlows: number[];
  linkTimes: number[];
  used: boolean[];
  totalVehMin: number;
  avgMin: number;
  maxGainMin: number;
}

/** The classic four-node example (Braess 1968, with the usual textbook numbers). */
export function braessNetwork(connector = true, D = 4000, slope = 100, constant = 45, connCost = 0): Network {
  const links: Link[] = [
    { from: "S", to: "A", a: 0, b: 1 / slope },
    { from: "A", to: "T", a: constant, b: 0 },
    { from: "S", to: "B", a: constant, b: 0 },
    { from: "B", to: "T", a: 0, b: 1 / slope },
  ];
  if (connector) links.push({ from: "A", to: "B", a: connCost, b: 0 });
  return { links, origin: "S", dest: "T", demand: D };
}

export function simplePaths(links: Link[], o: string, d: string): number[][] {
  const adj = new Map<string, [string, number][]>();
  links.forEach((l, k) => {
    if (!adj.has(l.from)) adj.set(l.from, []);
    adj.get(l.from)!.push([l.to, k]);
  });
  const out: number[][] = [];
  const walk = (u: string, seen: Set<string>, path: number[]) => {
    if (u === d) {
      out.push([...path]);
      return;
    }
    for (const [v, k] of adj.get(u) ?? []) {
      if (!seen.has(v)) walk(v, new Set([...seen, v]), [...path, k]);
    }
  };
  walk(o, new Set([o]), []);
  return out;
}

export function networkUE(net: Network, iters = 10000, tol = 1e-9): NetworkResult {
  const { links } = net;
  const D = net.demand;
  const paths = simplePaths(links, net.origin, net.dest);
  const m = links.length;
  const P = paths.length;
  const inc = paths.map((p) => links.map((_, k) => p.includes(k)));
  const load = (pf: number[]) => {
    const x = new Array(m).fill(0);
    paths.forEach((p, i) => p.forEach((k) => (x[k] += pf[i])));
    return x;
  };
  const ltimes = (x: number[]) => links.map((l, k) => l.a + l.b * x[k]);
  const ptimes = (x: number[]) => {
    const t = ltimes(x);
    return paths.map((p) => p.reduce((s, k) => s + t[k], 0));
  };
  const argmin = (c: number[]) => c.reduce((b, v, i) => (v < c[b] ? i : b), 0);
  const pf = new Array(P).fill(0);
  if (D > 0 && P) pf[argmin(ptimes(new Array(m).fill(0)))] = D;
  for (let it = 0; it < iters && D > 0 && P >= 2; it++) {
    const c = ptimes(load(pf));
    let w = -1;
    for (let i = 0; i < P; i++) if (pf[i] > 0 && (w < 0 || c[i] > c[w])) w = i;
    const best = argmin(c);
    if (c[w] - c[best] <= tol * Math.max(1, c[best])) break;
    let slope = 0;
    for (let k = 0; k < m; k++) {
      const bk = links[k].b;
      slope += (inc[w][k] ? bk : 0) + (inc[best][k] ? bk : 0) - (inc[w][k] && inc[best][k] ? 2 * bk : 0);
    }
    const d = slope <= 0 ? pf[w] : Math.min(pf[w], (c[w] - c[best]) / slope);
    pf[w] -= d;
    pf[best] += d;
  }
  const x = load(pf);
  const t = ltimes(x);
  const c = ptimes(x);
  const total = x.reduce((s, xk, k) => s + xk * t[k], 0);
  const used = pf.map((q) => q > 1e-9 * Math.max(D, 1));
  const best = c.length ? Math.min(...c) : NaN;
  const usedT = c.filter((_, i) => used[i]);
  return {
    paths: paths.map((p) => [net.origin, ...p.map((k) => links[k].to)].join("-")),
    pathLinks: paths,
    pathFlows: pf,
    pathTimes: c,
    linkFlows: x,
    linkTimes: t,
    used,
    totalVehMin: total,
    avgMin: D > 0 ? total / D : NaN,
    maxGainMin: usedT.length ? Math.max(0, Math.max(...usedT) - best) : 0,
  };
}
