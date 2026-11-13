"""Chaos experiments behind the chaos-theory cut of the film.

    python3 chaos2.py growth   # how fast a tiny difference grows: per second, per bounce
    python3 chaos2.py replay   # find winning drops, replay them at a given precision: does it win again?
    python3 chaos2.py extra    # board temperature changes, and how robust each winner is
    python3 chaos2.py film     # the film's hook pair, release maps, precision fit and horizon
    python3 chaos2.py all

Everything runs on the perfect board (no pin imperfections, no random restitution) unless a "real-world"
perturbation is named: the chip's own thermal motion, or a board a thousandth of a degree warmer.
Outputs: results/chaos2_growth.json, results/chaos2_replay.json
"""
import json
import math
import sys
import time

import numpy as np
from numba import njit, prange

import board as B
import physics as P

E = json.load(open("results/strategy.json"))["calibrated_e"]
PRM = P.params(e=E, jitter_deg=0.0, e_jitter=0.0)
W = B.SLOT_PITCH
KT = 1.380649e-23 * 293.0           # J, room temperature
CHIP_MASS = 0.25                    # kg: a 13 cm x 1.9 cm acrylic disc with a metal rim is 0.25-0.3 kg
I_CHIP = 0.6 * CHIP_MASS * (B.CHIP_DIAMETER / 2) ** 2
SIG_V = math.sqrt(KT / CHIP_MASS)   # thermal (equipartition) speed per axis, m/s
SIG_W = math.sqrt(KT / I_CHIP)      # thermal spin, rad/s
ALPHA = 1.0e-5                      # thermal expansion of a wood/steel board, per kelvin


# ------------------------------------------------------------------------------------------- growth
def growth(n_pairs=160, d0=1e-12, fps=2000):
    """Twin drops d0 apart on the perfect board. Fit the exponential growth of their separation."""
    rng = np.random.default_rng(11)
    rates_t, rates_b, curves = [], [], []
    for i in range(n_pairs):
        x0 = B.drop_x(5) + rng.normal(0, 0.004)
        sa, ra = P.trajectory(x0, 0.0, 0, 0, 0.0, seed=1, prm=PRM, fps=fps)
        sb, rb = P.trajectory(x0 + d0, 0.0, 0, 0, 0.0, seed=1, prm=PRM, fps=fps)
        if sa == 0 or sb == 0:
            continue
        m = min(len(ra), len(rb))
        sep = np.hypot(ra[:m, 1] - rb[:m, 1], ra[:m, 2] - rb[:m, 2])
        bounces = np.cumsum(ra[:m, 4])
        t = ra[:m, 0]
        ok = (sep > 1e-11) & (sep < 1e-3)
        if ok.sum() < 20:
            continue
        ls = np.log(np.maximum(sep, 1e-30))
        i0, i1 = np.flatnonzero(ok)[[0, -1]]
        if t[i1] - t[i0] < 0.05 or bounces[i1] - bounces[i0] < 2:
            continue
        rates_t.append((ls[i1] - ls[i0]) / (t[i1] - t[i0]))
        rates_b.append((ls[i1] - ls[i0]) / (bounces[i1] - bounces[i0]))
        if len(curves) < 12:
            curves.append(dict(t=t[::20].tolist(), sep=sep[::20].tolist(), bounces=bounces[::20].tolist(),
                               slots=[int(sa), int(sb)]))
    lt, lb = float(np.median(rates_t)), float(np.median(rates_b))
    out = dict(d0=d0, n=len(rates_t), lyapunov_per_s=lt, doubling_time_s=math.log(2) / lt,
               per_bounce_factor=math.exp(lb), per_bounce_log=lb,
               bounces_per_drop=float(np.median([c["bounces"][-1] for c in curves])),
               quantiles_per_s=[float(q) for q in np.quantile(rates_t, [0.25, 0.75])],
               quantiles_factor=[float(math.exp(q)) for q in np.quantile(rates_b, [0.25, 0.75])],
               curves=curves)
    json.dump(out, open("results/chaos2_growth.json", "w"), indent=1)
    print(f"growth: x{math.exp(lb):.2f} per bounce (IQR {out['quantiles_factor'][0]:.2f}-{out['quantiles_factor'][1]:.2f}),"
          f" lambda {lt:.1f}/s, doubling every {1000 * math.log(2) / lt:.1f} ms, {len(rates_t)} pairs")
    return out


# ------------------------------------------------------------------------------------------- replay
@njit(parallel=True, cache=True)
def _batch_scaled(init, seeds, prm, geom, segs, eps):
    """Like physics.simulate_batch, but each drop gets a board scaled by (1 + eps[i]) about its centre."""
    n = init.shape[0]
    out = np.zeros(n, dtype=np.int64)
    dummy = np.zeros((1, 5))
    for i in prange(n):
        g = geom.copy()
        s = segs.copy()
        f = 1.0 + eps[i]
        g[0] *= f          # pitch
        g[1] *= f          # row spacing
        g[4] *= f          # divider tops
        g[5] *= f          # slot floor
        for k in range(s.shape[0]):
            for j in range(4):
                s[k, j] *= f
        x = init[i, 0] + 5.0 * geom[0] * eps[i]      # the board grows about its centre line
        r = P.simulate_one(x, init[i, 1], init[i, 2], init[i, 3], init[i, 4], seeds[i], prm, g, s, 0, dummy)
        out[i] = r[0]
    return out


def run(init, eps=None):
    n = len(init)
    seeds = np.ones(n, dtype=np.uint64)
    if eps is None:
        s, _, _ = P.simulate_batch(init, seeds, PRM, P.GEOM, P.SEGS)
        return s
    return _batch_scaled(init, seeds, PRM, P.GEOM, P.SEGS, eps)


def replay(n_candidates=1200, n_rep=300):
    rng = np.random.default_rng(2024)
    # candidate drops: a robot releasing at rest near the middle gap
    cand = np.zeros((n_candidates, 5))
    cand[:, 0] = B.drop_x(5) + rng.uniform(-0.03, 0.03, n_candidates)
    s0 = run(cand)
    base = float((s0 == 5).mean())
    winners = cand[s0 == 5][:60]
    print(f"base rate (perfect board, release anywhere in the middle 6 cm): {base:.3f}; {len(winners)} winners")
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 3e-11, 1e-11, 3e-12, 1e-12, 3e-13, 1e-13,
              1e-14, 1e-15]
    res = {"base_rate": base, "n_winners": len(winners), "n_replays_each": n_rep, "chip_mass_kg": CHIP_MASS,
           "thermal_speed_m_s": SIG_V, "thermal_spin_rad_s": SIG_W, "position": {}, "real_world": {}}
    t0 = time.time()
    for d in deltas:
        hits = 0
        tot = 0
        for w in winners:
            rep = np.repeat(w[None, :], n_rep, 0)
            rep[:, 0] += rng.normal(0, d, n_rep)
            s = run(rep)
            ok = s > 0
            hits += (s[ok] == 5).sum()
            tot += ok.sum()
        res["position"][f"{d:g}"] = hits / tot
        print(f"replay precision {d:8.0e} m: wins again {hits / tot:.3f}   ({time.time() - t0:.0f}s)", flush=True)
    # real-world limits, with the release itself perfect
    scen = {
        "chip's own thermal motion": dict(v=SIG_V, w=SIG_W, eps=0.0),
        "board 0.001 C warmer or cooler": dict(v=0.0, w=0.0, eps=ALPHA * 1e-3),
        "board 0.0001 C": dict(v=0.0, w=0.0, eps=ALPHA * 1e-4),
    }
    for name, sc in scen.items():
        hits = tot = 0
        for w in winners:
            rep = np.repeat(w[None, :], n_rep, 0)
            rep[:, 2] += rng.normal(0, sc["v"], n_rep) if sc["v"] else 0
            rep[:, 3] += rng.normal(0, sc["v"], n_rep) if sc["v"] else 0
            rep[:, 4] += rng.normal(0, sc["w"], n_rep) if sc["w"] else 0
            eps = rng.normal(0, sc["eps"], n_rep) if sc["eps"] else None
            s = run(rep, eps)
            ok = s > 0
            hits += (s[ok] == 5).sum()
            tot += ok.sum()
        res["real_world"][name] = hits / tot
        print(f"{name:34s}: wins again {hits / tot:.3f}", flush=True)
    json.dump(res, open("results/chaos2_replay.json", "w"), indent=1)
    return res




# ------------------------------------------------------------------------------------------- film data
def film_data():
    """Everything the chaos cut needs beyond growth() and replay():
    the hook pair, release maps around a robust and a knife-edge winner, the precision fit, the horizon."""
    rng = np.random.default_rng(2024)
    cand = np.zeros((1200, 5))
    cand[:, 0] = B.drop_x(5) + rng.uniform(-0.03, 0.03, 1200)
    s0 = run(cand)
    winners = cand[s0 == 5][:60]
    # how robust is each winner at atomic precision (deterministic +-0.1 nm probes, 200 of them)?
    probe = np.linspace(-1e-10, 1e-10, 201)
    rob = []
    for w in winners:
        rep = np.repeat(w[None, :], len(probe), 0)
        rep[:, 0] += probe
        s = run(rep)
        rob.append(float((s == 5).mean()))
    rob = np.array(rob)
    robust = winners[int(np.argmax(rob))]
    # the hook: a knife-edge winner whose replay 0.1 nm away lands far off, after a long shared stretch
    hook = None
    for w, r in sorted(zip(winners, rob), key=lambda z: z[1]):
        if r > 0.6:
            break
        sa, ra = P.trajectory(w[0], 0.0, 0, 0, 0.0, seed=1, prm=PRM, fps=60)
        for sgn in (1, -1):
            sb, rb = P.trajectory(w[0] + sgn * 1e-10, 0.0, 0, 0, 0.0, seed=1, prm=PRM, fps=60)
            if sa != 5 or sb == 0 or abs(sb - 5) < 3:
                continue
            m = min(len(ra), len(rb))
            sep = np.hypot(ra[:m, 1] - rb[:m, 1], ra[:m, 2] - rb[:m, 2])
            k = int(np.argmax(sep > 0.01))
            frac = ra[k, 2] / B.last_row_y()
            if 0.25 < frac < 0.8 and (hook is None or frac > hook[3]):
                hook = (w, ra, rb, frac, sgn, int(sb), k / 60)
    knife = hook[0]
    np.savez_compressed("results/chaos2_hook.npz", win=hook[1][:, :4], replay=hook[2][:, :4])
    # release maps around each: windows of 2 mm, 20 um, 200 nm, 2 nm, 20 pm (4,001 points each)
    maps = {}
    for name, w in (("robust", robust), ("knife", knife)):
        for k, half in enumerate([1e-3, 1e-5, 1e-7, 1e-9, 1e-11]):
            rep = np.repeat(w[None, :], 4001, 0)
            rep[:, 0] += np.linspace(-half, half, 4001)
            maps[f"{name}{k}"] = run(rep).astype(np.int8)
    np.savez_compressed("results/chaos2_maps.npz", **maps)
    # precision fit: miss chance vs precision on log-log, over 1e-15..1e-9 m
    rp = json.load(open("results/chaos2_replay.json"))
    d = np.array([float(k) for k in rp["position"]])
    p = np.array(list(rp["position"].values()))
    sel = (d <= 1e-9) & (d >= 1e-15)
    slope = float(np.polyfit(np.log10(d[sel]), np.log10(1 - p[sel]), 1)[0])
    g = json.load(open("results/chaos2_growth.json"))
    fac = g["per_bounce_factor"]
    horizon = {lab: float(math.log(1e-2 / dd) / math.log(fac)) for lab, dd in
               (("1 mm", 1e-3), ("1 um", 1e-6), ("1 nm", 1e-9), ("1 atom", 1e-10), ("an atomic nucleus", 1e-14))}
    out = dict(robustness_at_atom=sorted(rob.round(3).tolist()), share_robust=float((rob > 0.9).mean()),
               share_knife=float((rob < 0.3).mean()), robust_x=float(robust[0] - B.drop_x(5)),
               knife_x=float(knife[0] - B.drop_x(5)), hook_replay_slot=hook[5], hook_split_t=hook[6],
               hook_offset_m=1e-10 * hook[4], uncertainty_exponent=slope,
               miss_cut_per_10x=float(10 ** -slope), horizon_bounces=horizon, per_bounce_factor=fac,
               whole_drop_precision_m=float(1e-2 / fac ** g["bounces_per_drop"]))
    json.dump(out, open("results/chaos2_film.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "robustness_at_atom"}, indent=1))


# ------------------------------------------------------------------------------------------- extra
def extra(n=300):
    """Board temperature changes (0.001 to 1 C) and how robust each winner is (results/chaos2_extra.json)."""
    rng = np.random.default_rng(2024)
    cand = np.zeros((1200, 5))
    cand[:, 0] = B.drop_x(5) + rng.uniform(-0.03, 0.03, 1200)
    s0 = run(cand)
    winners = cand[s0 == 5][:60]

    def again(perturb):
        per = []
        for w in winners:
            rep = np.repeat(w[None, :], n, 0)
            eps = perturb(rep)
            s = run(rep, eps)
            ok = s > 0
            per.append((s[ok] == 5).mean())
        return np.array(per)
    out = {}
    for dT in [1e-3, 1e-2, 1e-1, 1.0]:
        p = again(lambda rep: rng.normal(0, ALPHA * dT, n))
        out[f"board {dT:g} C"] = float(p.mean())
        print(f"board +-{dT:g} C: wins again {p.mean():.3f}", flush=True)
    for d in [1e-10, 1e-6]:
        def f(rep, d=d):
            rep[:, 0] += rng.normal(0, d, n)
            return None
        p = again(f)
        out[f"per-winner {d:g}"] = np.round(np.sort(p), 2).tolist()
    json.dump(out, open("results/chaos2_extra.json", "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    if "growth" in what or "all" in what:
        growth()
    if "replay" in what or "all" in what:
        replay()
    if "extra" in what or "all" in what:
        extra()
    if "film" in what or "all" in what:
        film_data()
