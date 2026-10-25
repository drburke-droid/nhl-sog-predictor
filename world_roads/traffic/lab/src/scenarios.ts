import { ALPHA, BETA, Route, joinsAt } from "./model/equilibrium";
import calgary from "./data/calgary.json";

export interface Shared {
  routes: Route[];
  demand: number;
  alpha: number;
  beta: number;
  showSO: boolean;
}

export const TEXTBOOK: Route[] = [
  { name: "Expressway", t0: 12, cap: 2000 },
  { name: "Arterial", t0: 17, cap: 1500 },
  { name: "Local streets", t0: 22, cap: 1100 },
];

// free-flow times measured on the real Calgary roads (Rocky Ridge -> Seton at the speed limits, no traffic);
// the capacities are illustrative, not Calgary data
export const CALGARY: Route[] = [
  { name: "Stoney Trail (west)", t0: Math.round(calgary.main[0].minutes * 10) / 10, cap: 2000 },
  { name: "Crowchild & Deerfoot", t0: Math.round(calgary.main[1].minutes * 10) / 10, cap: 1500 },
  { name: "Barlow & 52 St (east)", t0: Math.round(calgary.main[2].minutes * 10) / 10, cap: 1100 },
];

export const SYMMETRIC: Route[] = [
  { name: "Route A", t0: 15, cap: 1200 },
  { name: "Route B", t0: 15, cap: 1200 },
  { name: "Route C", t0: 15, cap: 1200 },
];

export const DEFAULT_SHARED: Shared = { routes: TEXTBOOK, demand: 4200, alpha: ALPHA, beta: BETA, showSO: true };

/** slider and chart range: past the point where the last route joins, without drowning it in gridlock */
export function demandMax(routes: Route[], alpha = ALPHA, beta = BETA): number {
  const cap = routes.reduce((s, r) => s + Math.max(0, r.cap), 0);
  const last = Math.max(0, ...joinsAt(routes, alpha, beta).filter(Number.isFinite));
  return Math.max(1000, Math.ceil(Math.max(1.5 * cap, 1.3 * last) / 500) * 500);
}

export { calgary };
