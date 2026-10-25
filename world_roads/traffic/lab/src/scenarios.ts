import { ALPHA, BETA, Route, joinsAt } from "./model/equilibrium";
import trips from "./trips/trips.json";

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

export type TripId = keyof typeof trips;

// free-flow times measured on real roads at the speed limits, with no traffic (city_routes.py);
// the capacities are illustrative, not measurements of those roads
export function tripRoutes(id: TripId): Route[] {
  const caps = [2000, 1500, 1100];
  return trips[id].main.map((r, i) => ({ name: r.name, t0: Math.round(r.minutes * 10) / 10, cap: caps[i] }));
}
export const NYC = tripRoutes("nyc");
export const CALGARY = tripRoutes("calgary");

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

export { trips };
