// Independent parallel routes with BPR congestion: exact user equilibrium (UE) and system optimum (SO).
// Mirrors ../../model.py. Times in minutes; flows, capacities and demand in vehicles per hour (veh/h).

export const ALPHA = 0.15;
export const BETA = 4;
export const EPS_FLOW = 1e-9; // a route carrying less than this (veh/h) counts as unused ...
export const USED_SHARE = 1e-4; // ... as does one carrying under 0.01 % of all trips

export interface Route {
  name: string;
  t0: number; // free-flow time, min
  cap: number; // capacity, veh/h
}

export interface Assignment {
  flows: number[];
  times: number[];
  used: boolean[];
  open: boolean[];
  demand: number;
  totalVehMin: number;
  avgMin: number;
  maxGainMin: number; // the most one driver could save by switching alone
  avgGainMin: number; // what the average driver could save
  level: number; // common time (UE) or common marginal cost (SO); NaN without demand
  msc?: number[];
}

export function bpr(t0: number, cap: number, f: number, alpha = ALPHA, beta = BETA): number {
  if (f <= 0) return t0;
  if (cap <= 0) return Infinity;
  return t0 * (1 + alpha * Math.pow(f / cap, beta));
}

export function msc(t0: number, cap: number, f: number, alpha = ALPHA, beta = BETA): number {
  if (f <= 0) return t0;
  if (cap <= 0) return Infinity;
  return t0 * (1 + alpha * (1 + beta) * Math.pow(f / cap, beta));
}

function flowAt(level: number, t0: number, cap: number, a: number, beta: number): number {
  if (cap <= 0 || level <= t0) return 0;
  return cap * Math.pow((level / t0 - 1) / a, 1 / beta);
}

export function openRoutes(routes: Route[]): number[] {
  const out: number[] = [];
  routes.forEach((r, i) => {
    if (r.cap > 0 && r.t0 > 0 && Number.isFinite(r.t0)) out.push(i);
  });
  return out;
}

function solveLevel(routes: Route[], D: number, a: number, beta: number): { lam: number; f: number[] } {
  const n = routes.length;
  const ok = openRoutes(routes);
  const f = new Array(n).fill(0);
  if (D <= 0 || ok.length === 0) {
    return { lam: ok.length ? Math.min(...ok.map((i) => routes[i].t0)) : NaN, f };
  }
  if (a <= 0) {
    const best = Math.min(...ok.map((i) => routes[i].t0));
    const tied = ok.filter((i) => routes[i].t0 <= best * (1 + 1e-12));
    const cs = tied.reduce((s, i) => s + routes[i].cap, 0);
    tied.forEach((i) => (f[i] = (D * routes[i].cap) / cs));
    return { lam: best, f };
  }
  let lo = Math.min(...ok.map((i) => routes[i].t0));
  let hi = lo * 2;
  const total = (lam: number) => ok.reduce((s, i) => s + flowAt(lam, routes[i].t0, routes[i].cap, a, beta), 0);
  for (let k = 0; k < 400 && total(hi) < D; k++) hi *= 2;
  for (let k = 0; k < 200; k++) {
    const mid = 0.5 * (lo + hi);
    if (total(mid) < D) lo = mid;
    else hi = mid;
    if (hi - lo <= 1e-13 * hi) break;
  }
  const lam = 0.5 * (lo + hi);
  ok.forEach((i) => (f[i] = flowAt(lam, routes[i].t0, routes[i].cap, a, beta)));
  const s = f.reduce((x, y) => x + y, 0);
  if (s > 0) for (let i = 0; i < n; i++) f[i] *= D / s;
  return { lam, f };
}

export function summarize(routes: Route[], f: number[], alpha = ALPHA, beta = BETA): Assignment {
  const n = routes.length;
  const ok = new Set(openRoutes(routes));
  const T = routes.map((r, i) => (ok.has(i) ? bpr(r.t0, r.cap, f[i], alpha, beta) : Infinity));
  const D = f.reduce((x, y) => x + y, 0);
  const usedIdx = f.map((x, i) => (x > Math.max(EPS_FLOW, USED_SHARE * D) ? i : -1)).filter((i) => i >= 0);
  let total = 0;
  for (let i = 0; i < n; i++) if (f[i] > 0) total += f[i] * T[i];
  const okT = [...ok].map((i) => T[i]);
  const best = okT.length ? Math.min(...okT) : NaN;
  const gain = usedIdx.length ? Math.max(...usedIdx.map((i) => T[i])) - best : 0;
  const avg = D > 0 ? total / D : NaN;
  return {
    flows: [...f],
    times: T,
    used: routes.map((_, i) => usedIdx.includes(i)),
    open: routes.map((_, i) => ok.has(i)),
    demand: D,
    totalVehMin: total,
    avgMin: avg,
    maxGainMin: Math.max(0, gain),
    avgGainMin: D > 0 ? Math.max(0, avg - best) : 0,
    level: NaN,
  };
}

/** Wardrop user equilibrium: used routes share one travel time; unused routes are no faster. */
export function parallelUE(routes: Route[], D: number, alpha = ALPHA, beta = BETA): Assignment {
  const { lam, f } = solveLevel(routes, D, alpha, beta);
  const out = summarize(routes, f, alpha, beta);
  out.level = D > 0 ? lam : NaN;
  return out;
}

/** System optimum: equal marginal social cost on used routes; reported times are BPR times, not MSC. */
export function parallelSO(routes: Route[], D: number, alpha = ALPHA, beta = BETA): Assignment {
  const { lam, f } = solveLevel(routes, D, alpha * (1 + beta), beta);
  const out = summarize(routes, f, alpha, beta);
  out.level = D > 0 ? lam : NaN;
  out.msc = routes.map((r, i) => msc(r.t0, r.cap, f[i], alpha, beta));
  return out;
}

export function priceOfAnarchy(ue: Assignment, so: Assignment): number {
  return so.totalVehMin > 0 ? ue.totalVehMin / so.totalVehMin : NaN;
}

/** Demand (veh/h) at which each route starts to carry traffic (0 for the fastest; Infinity if closed). */
export function joinsAt(routes: Route[], alpha = ALPHA, beta = BETA, system = false): number[] {
  const a = system ? alpha * (1 + beta) : alpha;
  const ok = openRoutes(routes);
  return routes.map((rj, j) => {
    if (!ok.includes(j)) return Infinity;
    return ok.filter((i) => i !== j).reduce((s, i) => s + flowAt(rj.t0, routes[i].t0, routes[i].cap, a, beta), 0);
  });
}

export interface Presets {
  low: number;
  moderate: number;
  near: number;
  heavy: number;
}

/** Demand presets computed from the model (see model.py: presets). */
export function presets(routes: Route[], alpha = ALPHA, beta = BETA): Presets {
  const ok = openRoutes(routes);
  if (!ok.length) return { low: 0, moderate: 0, near: 0, heavy: 0 };
  const ja = joinsAt(routes, alpha, beta);
  const j = ok.map((i) => ja[i]).sort((x, y) => x - y);
  const totalCap = ok.reduce((s, i) => s + routes[i].cap, 0);
  const second = j.length > 1 ? j[1] : totalCap;
  const last = j.length > 1 ? j[j.length - 1] : totalCap;
  const third = j.length > 2 ? j[2] : last;
  const slow = ok.reduce((b, i) => (routes[i].t0 > routes[b].t0 ? i : b), ok[0]);
  const lamNear = Math.max(routes[slow].t0 - 0.5, Math.min(...ok.map((i) => routes[i].t0)) + 0.01);
  const near = ok.reduce((s, i) => s + flowAt(lamNear, routes[i].t0, routes[i].cap, alpha, beta), 0);
  return {
    low: Math.round(0.6 * second),
    moderate: Math.round(0.5 * (second + third)),
    near: Math.round(near),
    heavy: Math.round(Math.max(1.25 * last, 1.1 * totalCap)),
  };
}
