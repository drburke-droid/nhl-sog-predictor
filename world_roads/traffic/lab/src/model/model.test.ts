import { describe, expect, it } from "vitest";
import { Route, joinsAt, parallelSO, parallelUE, presets, priceOfAnarchy } from "./equilibrium";
import { learning } from "./learning";
import { braessNetwork, networkUE } from "./network";
import { appShown, randomRoutes, selectionStats } from "./app";
import { mulberry32 } from "./rng";
import fx from "./fixtures.json";

const R: Route[] = [
  { name: "A", t0: 12, cap: 2000 },
  { name: "B", t0: 17, cap: 1500 },
  { name: "C", t0: 22, cap: 1100 },
];
const close = (a: number, b: number, tol = 1e-6) => expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));
const sum = (x: number[]) => x.reduce((s, v) => s + v, 0);

describe("parallel-route equilibrium", () => {
  it("zero demand: no flow, no NaN times, no equilibrium claim", () => {
    for (const r of [parallelUE(R, 0), parallelSO(R, 0)]) {
      expect(r.flows).toEqual([0, 0, 0]);
      expect(r.used).toEqual([false, false, false]);
      expect(r.totalVehMin).toBe(0);
      expect(r.times.every(Number.isFinite)).toBe(true);
      expect(Number.isNaN(r.level)).toBe(true);
    }
    expect(Number.isNaN(priceOfAnarchy(parallelUE(R, 0), parallelSO(R, 0)))).toBe(true);
  });

  it.each([1, 500, 2600, 4000, 12000, 40000])("UE at D=%d: flows sum, used times equal, unused no faster", (D) => {
    const r = parallelUE(R, D);
    expect(r.flows.every((f) => f >= 0)).toBe(true);
    close(sum(r.flows), D, 1e-9);
    const used = r.times.filter((_, i) => r.used[i]);
    expect(Math.max(...used) - Math.min(...used)).toBeLessThan(1e-6);
    r.times.forEach((t, i) => {
      if (!r.used[i]) expect(t).toBeGreaterThanOrEqual(Math.min(...used) - 1e-9);
    });
    expect(r.maxGainMin).toBeLessThan(1e-6);
  });

  it("symmetric routes split equally", () => {
    const s = [0, 1, 2].map((i) => ({ name: String(i), t0: 15, cap: 1200 }));
    parallelUE(s, 3000).flows.forEach((f) => close(f, 1000, 1e-9));
  });

  it.each([300, 2500, 5000, 9000])("SO is never worse than UE (D=%d), PoA >= 1", (D) => {
    const ue = parallelUE(R, D);
    const so = parallelSO(R, D);
    expect(so.totalVehMin).toBeLessThanOrEqual(ue.totalVehMin * (1 + 1e-12));
    expect(priceOfAnarchy(ue, so)).toBeGreaterThanOrEqual(1 - 1e-12);
    const m = so.msc!.filter((_, i) => so.used[i]);
    expect(Math.max(...m) - Math.min(...m)).toBeLessThan(1e-6);
  });

  it("presets are honest about which routes are used", () => {
    const p = presets(R);
    expect(parallelUE(R, p.low).used).toEqual([true, false, false]);
    expect(parallelUE(R, p.moderate).used).toEqual([true, true, false]);
    const near = parallelUE(R, p.near);
    expect(near.used).toEqual([true, true, false]);
    close(22 - near.level, 0.5, 0.02);
    expect(parallelUE(R, p.heavy).used).toEqual([true, true, true]);
  });

  it("closed routes and extreme settings stay finite", () => {
    const r = parallelUE([R[0], { ...R[1], cap: 0 }, { ...R[2], cap: -1 }], 3000);
    expect(r.flows[1]).toBe(0);
    close(r.flows[0], 3000);
    expect(parallelUE(R.map((x) => ({ ...x, cap: 0 })), 3000).totalVehMin).toBe(0);
    for (const [a, b, D] of [
      [0.01, 1, 1e5],
      [1, 8, 1e5],
      [0.15, 4, 1e6],
      [2, 10, 1],
    ]) {
      const ue = parallelUE(R, D, a, b);
      const so = parallelSO(R, D, a, b);
      for (const v of [...ue.flows, ...ue.times, ...so.flows, ...so.times, ue.avgMin, so.avgMin]) expect(Number.isFinite(v)).toBe(true);
      expect(Number.isFinite(priceOfAnarchy(ue, so))).toBe(true);
    }
  });

  it("matches model.py", () => {
    fx.ue.forEach((c) => {
      const r = parallelUE(R, c.D);
      c.flows.forEach((f: number, i: number) => close(r.flows[i], f, 1e-9));
    });
    fx.so.forEach((c) => parallelSO(R, c.D).flows.forEach((f, i) => close(f, c.flows[i], 1e-9)));
    joinsAt(R).forEach((j, i) => close(j, fx.joins[i], 1e-9));
    expect(presets(R)).toEqual(fx.presets);
  });
});

describe("day-to-day learning", () => {
  const base = { days: 80, replan: 0.2, delay: 1, theta: 0, sigma: 0, noise: 0, seed: 1 };
  it.each([{ replan: 1 }, { replan: 0.1, delay: 3 }, { replan: 0.5, sigma: 2, noise: 0.2, seed: 9 }, { replan: 0.3, theta: 2 }])(
    "never creates or destroys vehicles, never negative (%o)",
    (kw) => {
      const r = learning(R, 4200, { ...base, ...kw });
      for (const f of r.flows) {
        expect(f.every((x) => x >= 0)).toBe(true);
        close(sum(f), 4200, 1e-9);
      }
      expect(r.maxGain.every(Number.isFinite)).toBe(true);
    },
  );
  it("is repeatable from its seed and matches model.py", () => {
    fx.learning.forEach((c) => {
      const kw = c.kw as unknown as Record<string, number>;
      const r = learning(R, 4200, { ...base, days: 40, ...kw });
      c.flows.forEach((f: number, i: number) => close(r.flows[40][i], f, 1e-9));
      c.day10.forEach((f: number, i: number) => close(r.flows[10][i], f, 1e-9));
    });
  });
  it("oscillates when everyone chases yesterday's winner", () => {
    const r = learning(R, 4200, { ...base, days: 200, replan: 1 });
    expect(Math.min(...r.avgGain.slice(-20))).toBeGreaterThan(1);
  });
});

describe("Braess network", () => {
  it("65 min without the connector, 80 min with it", () => {
    const closed = networkUE(braessNetwork(false));
    closed.pathTimes.forEach((t) => close(t, 65));
    closed.pathFlows.forEach((f) => close(f, 2000, 1e-6));
    const open = networkUE(braessNetwork(true));
    const i = open.paths.indexOf("S-A-B-T");
    close(open.pathFlows[i], 4000, 1e-6);
    close(open.pathTimes[i], 80);
    open.pathTimes.filter((_, k) => k !== i).forEach((t) => close(t, 85));
  });
  it("matches model.py", () => {
    fx.braess.forEach((c) => {
      const r = networkUE(braessNetwork(c.conn, c.D));
      expect(r.paths).toEqual(c.paths);
      c.flows.forEach((f: number, i: number) => close(r.pathFlows[i], f, 1e-6));
      close(r.avgMin, c.avg, 1e-9);
    });
  });
});

describe("what the app shows", () => {
  it("shows the k fastest, optionally within a detour limit", () => {
    const t = randomRoutes(20, 24, 1, 5);
    const s = appShown(t, 3);
    expect(t[s[0]]).toBe(Math.min(...t));
    expect(appShown([10, 30, 11], 3, 0.25)).toEqual([0, 2]);
  });
  it("selection alone makes the shown routes look close", () => {
    const st = selectionStats(500, 20, 24, 1, 3, false);
    expect(st.shownSpread).toBeLessThan(0.2 * st.allSpread);
  });
  it("PRNG matches the reference", () => {
    const r = mulberry32(42);
    expect([r(), r(), r()].map((x) => x.toFixed(10))).toEqual(["0.6011037519", "0.4482905590", "0.8524657935"]);
  });
});
