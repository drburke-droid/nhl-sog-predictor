// What a route-suggesting app shows: a stylized filter, not any real app's logic. Mirrors model.py.
import { mulberry32 } from "./rng";

/** Indices of the k fastest routes, optionally only those within maxDetour (0.25 = 25 % slower). */
export function appShown(times: number[], k = 3, maxDetour: number | null = null): number[] {
  const order = times.map((_, i) => i).sort((a, b) => times[a] - times[b] || a - b);
  const best = order.length ? times[order[0]] : NaN;
  return order.slice(0, k).filter((i) => maxDetour === null || times[i] <= best * (1 + maxDetour));
}

/** n candidate route times with no traffic equilibrium at all: the fastest is `best`, the rest uniform up to (1+spread)x. */
export function randomRoutes(n: number, best = 24, spread = 1, seed = 1): number[] {
  const rnd = mulberry32(seed);
  const t = [best];
  for (let i = 1; i < n; i++) t.push(best * (1 + spread * rnd()));
  return t;
}

export interface SpreadStats {
  shownSpread: number; // mean (slowest shown - fastest), min
  allSpread: number; // mean (slowest candidate - fastest), min
  shownWithin1: number; // share of trials in which every shown route is within 1 min (after rounding)
}

/** Average over many seeded random cities: how close do the shown ETAs look, compared with all candidates? */
export function selectionStats(trials: number, n: number, best: number, spread: number, k: number, round: boolean, seed = 1): SpreadStats {
  let s1 = 0;
  let s2 = 0;
  let w1 = 0;
  for (let t = 0; t < trials; t++) {
    const times = randomRoutes(n, best, spread, seed * 7919 + t);
    const shown = appShown(times, k).map((i) => (round ? Math.round(times[i]) : times[i]));
    const sp = Math.max(...shown) - Math.min(...shown);
    s1 += sp;
    s2 += Math.max(...times) - Math.min(...times);
    if (sp <= 1 + 1e-9) w1 += 1;
  }
  return { shownSpread: s1 / trials, allSpread: s2 / trials, shownWithin1: w1 / trials };
}
