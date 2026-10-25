// Day-to-day route choice: a behavioural sketch, NOT an equilibrium solver. Mirrors model.py: learning.
import { ALPHA, BETA, Route, bpr, openRoutes, summarize } from "./equilibrium";
import { gauss, mulberry32 } from "./rng";

export interface LearningParams {
  days: number;
  start?: number[]; // initial shares (any scale); default: 90 % on the fastest empty road
  replan: number; // share of each route's drivers who reconsider each day (0..1)
  delay: number; // drivers use the times observed this many days earlier (>= 1)
  theta: number; // switching threshold: only switch for a saving above this (min)
  sigma: number; // spread of perceived times (logit scale, min); 0 = everyone sees the same
  noise: number; // day-to-day disturbance of observed times (relative SD, e.g. 0.1)
  seed: number;
  alpha?: number;
  beta?: number;
}

export interface LearningRun {
  flows: number[][];
  times: number[][];
  observed: number[][];
  maxGain: number[];
  avgGain: number[];
}

/** P[i][j]: share of re-planning drivers now on route i who pick route j. */
export function choiceMatrix(T: number[], ok: number[], theta: number, sigma: number): number[][] {
  const n = T.length;
  const P = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const i of ok) {
    const cost = new Map<number, number>();
    for (const j of ok) cost.set(j, T[j] - (j === i ? theta : 0));
    const m = Math.min(...cost.values());
    if (sigma > 0) {
      let s = 0;
      const w = new Map<number, number>();
      for (const [j, c] of cost) {
        const v = Math.exp(-(c - m) / sigma);
        w.set(j, v);
        s += v;
      }
      for (const j of ok) P[i][j] = w.get(j)! / s;
    } else if (cost.get(i)! <= m + 1e-12) {
      P[i][i] = 1;
    } else {
      const best = ok.filter((j) => cost.get(j)! <= m + 1e-12);
      for (const j of best) P[i][j] = 1 / best.length;
    }
  }
  return P;
}

export function defaultStart(routes: Route[]): number[] {
  const ok = openRoutes(routes);
  const start = new Array(routes.length).fill(0);
  if (!ok.length) return start;
  const fastest = ok.reduce((b, i) => (routes[i].t0 < routes[b].t0 ? i : b), ok[0]);
  for (const i of ok) start[i] = i === fastest ? 0.9 : 0.1 / Math.max(1, ok.length - 1);
  if (ok.length === 1) start[fastest] = 1;
  return start;
}

export function learning(routes: Route[], D: number, p: LearningParams): LearningRun {
  const alpha = p.alpha ?? ALPHA;
  const beta = p.beta ?? BETA;
  const ok = openRoutes(routes);
  const start = p.start ?? defaultStart(routes);
  const s0 = ok.reduce((s, i) => s + start[i], 0);
  let f = routes.map((_, i) => (ok.includes(i) && s0 > 0 ? (start[i] / s0) * D : 0));
  const rnd = mulberry32(p.seed);
  const out: LearningRun = { flows: [], times: [], observed: [], maxGain: [], avgGain: [] };
  const delay = Math.max(1, Math.round(p.delay));
  for (let day = 0; day <= p.days; day++) {
    const T = routes.map((r, i) => (ok.includes(i) ? bpr(r.t0, r.cap, f[i], alpha, beta) : Infinity));
    const obs = T.map((t, i) => (ok.includes(i) ? t * Math.max(0.05, 1 + p.noise * gauss(rnd)) : Infinity));
    out.flows.push([...f]);
    out.times.push(T);
    out.observed.push(obs);
    const sm = summarize(routes, f, alpha, beta);
    out.maxGain.push(sm.maxGainMin);
    out.avgGain.push(sm.avgGainMin);
    if (day === p.days) break;
    const seen = out.observed[Math.max(0, out.observed.length - delay)];
    const P = choiceMatrix(seen, ok, p.theta, p.sigma);
    const g = f.map((x) => (1 - p.replan) * x);
    for (const i of ok) for (const j of ok) g[j] += p.replan * f[i] * P[i][j];
    const s = g.reduce((x, y) => x + y, 0);
    f = g.map((x) => (s > 0 ? Math.max(0, x) * (D / s) : 0));
  }
  return out;
}
