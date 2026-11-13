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
import os
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
# A chip only fits between the two top pins if its centre is within +-(W/2 - pin radius - chip radius) = 1.93 cm
# of the middle; releases further out would start overlapping a pin. Keep 0.1 mm clear.
GAP = 0.5 * W - B.PEG_DIAMETER / 2 - B.CHIP_DIAMETER / 2 - 1e-4
N_CAND, N_WIN = 2000, 80


def candidates():
    """Robot releases at rest, uniformly across the whole valid width of the middle gap; and the winners."""
    rng = np.random.default_rng(2025)
    cand = np.zeros((N_CAND, 5))
    cand[:, 0] = B.drop_x(5) + rng.uniform(-GAP, GAP, N_CAND)
    s0 = run(cand)
    return cand, s0, cand[s0 == 5][:N_WIN]


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


def replay(n_rep=300):
    rng = np.random.default_rng(2024)
    cand, s0, winners = candidates()
    base = float((s0 == 5).mean())
    print(f"base rate (perfect board, release anywhere in the valid gap): {base:.3f}; {len(winners)} winners")
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 3e-11, 1e-11, 3e-12, 1e-12, 3e-13, 1e-13,
              1e-14, 1e-15]
    res = {"base_rate": base, "n_candidates": N_CAND, "release_half_range_m": GAP, "n_winners": len(winners),
           "n_replays_each": n_rep, "chip_mass_kg": CHIP_MASS,
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
def hook_release():
    """The opening shot's winning release point, exactly as it was found (seeded search over +-3 cm, 2024).

    It sits at -1.898 cm, inside the valid gap; recovered bit-for-bit so its path matches the rendered film."""
    rng = np.random.default_rng(2024)
    old = B.drop_x(5) + rng.uniform(-0.03, 0.03, 1200)
    x = float(old[np.argmin(np.abs(old - (B.drop_x(5) - 0.018982454584592134)))])
    h = np.load("results/chaos2_hook.npz")
    sa, ra = P.trajectory(x, 0.0, 0, 0, 0.0, seed=1, prm=PRM, fps=60)
    sb, rb = P.trajectory(x - 1e-10, 0.0, 0, 0, 0.0, seed=1, prm=PRM, fps=60)
    assert sa == 5 and np.allclose(ra[:, :4], h["win"]) and np.allclose(rb[:, :4], h["replay"]), "hook moved"
    assert abs(x - B.drop_x(5)) < GAP
    return x, int(sb)


def film_data():
    """Everything the chaos cut needs beyond growth() and replay(): how robust each winner is, release maps
    around an island and around the opening's (knife-edge) winner, a strip across the whole valid gap,
    the precision fit and the horizon."""
    cand, s0, winners = candidates()
    probe = np.linspace(-1e-10, 1e-10, 201)          # +-0.1 nm, deterministic probes
    rob = []
    for w in winners:
        rep = np.repeat(w[None, :], len(probe), 0)
        rep[:, 0] += probe
        rob.append(float((run(rep) == 5).mean()))
    rob = np.array(rob)
    robust = winners[int(np.argmax(rob))]
    xk, hook_slot = hook_release()
    knife = np.zeros(5)
    knife[0] = xk
    rep = np.repeat(knife[None, :], len(probe), 0)
    rep[:, 0] += probe
    knife_rob = float((run(rep) == 5).mean())
    maps = {}
    for name, w in (("robust", robust), ("knife", knife)):
        for k, half in enumerate([1e-3, 1e-5, 1e-7, 1e-9, 1e-11]):
            rep = np.repeat(w[None, :], 4001, 0)
            rep[:, 0] += np.linspace(-half, half, 4001)
            maps[f"{name}{k}"] = run(rep).astype(np.int8)
    np.savez_compressed("results/chaos2_maps.npz", **maps)
    ov = np.zeros((20001, 5))
    ov[:, 0] = B.drop_x(5) + np.linspace(-GAP, GAP, 20001)
    np.savez_compressed("results/chaos2_overview.npz", slots=run(ov).astype(np.int8),
                        dx=np.linspace(-GAP, GAP, 20001))
    rp = json.load(open("results/chaos2_replay.json"))
    d = np.array([float(k) for k in rp["position"]])
    p = np.array(list(rp["position"].values()))
    sel = (d <= 1e-9) & (d >= 1e-15)
    slope = float(np.polyfit(np.log10(d[sel]), np.log10(1 - p[sel]), 1)[0])
    g = json.load(open("results/chaos2_growth.json"))
    fac = g["per_bounce_factor"]
    horizon = {lab: float(math.log(1e-2 / dd) / math.log(fac)) for lab, dd in
               (("1 mm", 1e-3), ("1 um", 1e-6), ("1 nm", 1e-9), ("1 atom", 1e-10), ("an atomic nucleus", 1e-14))}
    old = json.load(open("results/chaos2_film.json")) if os.path.exists("results/chaos2_film.json") else {}
    out = dict(release_half_range_m=GAP, robustness_at_atom=sorted(rob.round(3).tolist()),
               share_robust=float((rob > 0.9).mean()), share_knife=float((rob < 0.3).mean()),
               robust_x=float(robust[0] - B.drop_x(5)), knife_x=float(xk - B.drop_x(5)),
               knife_robustness_at_atom=knife_rob, hook_replay_slot=hook_slot,
               hook_split_t=old.get("hook_split_t"), hook_offset_m=-1e-10, uncertainty_exponent=slope,
               miss_cut_per_10x=float(10 ** -slope), horizon_bounces=horizon, per_bounce_factor=fac,
               whole_drop_precision_m=float(1e-2 / fac ** g["bounces_per_drop"]))
    json.dump(out, open("results/chaos2_film.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "robustness_at_atom"}, indent=1))
    for name in ("robust", "knife"):
        print(name, [round(float((maps[f"{name}{k}"] == 5).mean()), 3) for k in range(5)])


# ------------------------------------------------------------------------------------------- extra
def extra(n=300):
    """Board temperature changes (0.001 to 1 C) and how robust each winner is (results/chaos2_extra.json)."""
    rng = np.random.default_rng(2024)
    cand, s0, winners = candidates()

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
