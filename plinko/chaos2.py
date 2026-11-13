"""Chaos experiments behind the chaos-theory cut of the film.

    python3 chaos2.py growth   # how fast a tiny difference grows: per second, per bounce
    python3 chaos2.py replay   # find winning drops, replay them at a given precision: does it win again?
    python3 chaos2.py extra    # board temperature changes, and how robust each winner is
    python3 chaos2.py film     # the film's hook pair, release maps, precision fit and horizon
    python3 chaos2.py onevar   # perfect release, only the board's temperature changes (one drop's own heat)
    python3 chaos2.py regionmap  # every release point across the top, up to 30 cm above the pins: the slot
    python3 chaos2.py region     # replay, thermal, robustness and warming for winners from anywhere on top
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


# ------------------------------------------------------------------------------------------- one variable
BOARD_KG, BOARD_C = 50.0, 1700.0    # backboard ~1.6 m x 2.5 m x 19 mm of wood (~45 kg) plus pins; wood J/(kg K)
N_CAND1 = 6000


def drop_heat():
    """Heat one drop leaves behind: the chip falls from the top row to rest on the slot floor and stops, so all
    of that potential energy becomes heat (an upper bound for the board: some warms the chip, some is sound).
    Spread evenly through the board, it warms it by dT."""
    along = B.last_row_y() + B.FLOOR_BELOW_LAST_ROW - B.CHIP_DIAMETER / 2
    h = along * math.cos(math.radians(B.BOARD_TILT_DEG))
    energy = CHIP_MASS * B.G * h
    return energy, h, energy / (BOARD_KG * BOARD_C)


def trajectory_scaled(x0, eps, fps=60):
    """P.trajectory on a board scaled by (1 + eps) about its centre line (as _batch_scaled)."""
    g, s = P.GEOM.copy(), P.SEGS.copy()
    f = 1.0 + eps
    g[[0, 1, 4, 5]] *= f
    s[:, :4] *= f
    every = max(1, int(round(1.0 / (fps * PRM[4]))))
    rec = np.zeros((int(PRM[5] * fps) + 10, 5))
    slot, t, c, nrec = P.simulate_one(x0 + 5.0 * P.GEOM[0] * eps, 0.0, 0.0, 0.0, 0.0, np.uint64(1), PRM, g, s,
                                      every, rec)
    return int(slot), rec[:nrec].copy()


def onevar():
    """A perfect release, everything held fixed except one thing: the board's temperature. Find winning drops,
    warm the board by dT (uniform expansion), replay each exactly: what share still win? Plus one pair for the
    film: a winner, and its exact replay on a board warmed by one drop's own heat
    (results/chaos2_onevar.json, results/chaos2_warm.npz)."""
    rng = np.random.default_rng(2026)
    cand = np.zeros((N_CAND1, 5))
    cand[:, 0] = B.drop_x(5) + rng.uniform(-GAP, GAP, N_CAND1)
    s0 = run(cand)
    win = cand[s0 == 5]
    n = len(win)
    assert (run(win, np.zeros(n)) == 5).all(), "an exact replay on the same board must win"
    energy, h, dT_drop = drop_heat()
    dTs = sorted(set([float(f"{v:.3g}") for v in np.logspace(-10, 0, 21)] + [float(f"{dT_drop:.3g}")]))
    curve = []
    for dT in dTs:
        s = run(win, np.full(n, ALPHA * dT))
        ok = s > 0
        curve.append(float((s[ok] == 5).mean()))
        print(f"board warmer by {dT:9.3g} C: {curve[-1]:.3f} of {n} winners win again", flush=True)
    s_drop = run(win, np.full(n, ALPHA * dT_drop))
    # the film's pair: the warmed replay misses, lands as far from $10,000 in value as possible, and parts from
    # the original in the middle of the board, where it's easy to see
    vals = B.SLOT_VALUES
    best = None
    for i in np.flatnonzero((s_drop > 0) & (s_drop != 5)):
        sa, ra = trajectory_scaled(win[i, 0], 0.0)
        sb, rb = trajectory_scaled(win[i, 0], ALPHA * dT_drop)
        if sa != 5 or sb != s_drop[i]:
            continue
        m = min(len(ra), len(rb))
        sep = np.hypot(ra[:m, 1] - rb[:m, 1], ra[:m, 2] - rb[:m, 2])
        k = int(np.argmax(sep > 0.01))
        frac = ra[k, 2] / B.last_row_y()
        score = (vals[sb - 1] == 0) * 2 + (vals[sb - 1] <= 100) + (1 - abs(frac - 0.55))
        if best is None or score > best[0]:
            best = (score, i, sa, ra, sb, rb, float(ra[k, 0]), float(frac))
    _, i, sa, ra, sb, rb, t_split, frac = best
    np.savez_compressed("results/chaos2_warm.npz", win=ra[:, :4], replay=rb[:, :4])
    out = dict(n_winners=n, n_candidates=N_CAND1, alpha_per_K=ALPHA, board_kg=BOARD_KG, board_c=BOARD_C,
               drop_energy_J=energy, drop_height_m=h, drop_dT_C=dT_drop,
               drop_expansion_m=ALPHA * dT_drop * 2.5, dT_C=dTs, wins_again=curve,
               wins_again_drop=float((s_drop[s_drop > 0] == 5).mean()),
               base_rate=float((s0 == 5).mean()), pair_release_m=float(win[i, 0] - B.drop_x(5)),
               pair_slots=[int(sa), int(sb)], pair_split_t=t_split, pair_split_frac=frac)
    json.dump(out, open("results/chaos2_onevar.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("dT_C", "wins_again")}, indent=1))


# ------------------------------------------------------------------------------------------- the whole top
# On the show a player may let go anywhere across the top: right at the top pins or anywhere in the ~12 in
# (30 cm) above them, all the way between the side walls. A release is valid where the chip touches neither a
# wall nor a top pin.
TOP = 0.3048
X_LO = 0.5 * W + B.CHIP_DIAMETER / 2 + 1e-4
X_HI = 9.5 * W - B.CHIP_DIAMETER / 2 - 1e-4
CLEAR = B.CHIP_DIAMETER / 2 + B.PEG_DIAMETER / 2 + 1e-4
DELTAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 3e-11, 1e-11, 3e-12, 1e-12, 3e-13, 1e-13, 1e-14,
          1e-15]


def valid_release(x, y):
    ok = (x >= X_LO) & (x <= X_HI) & (y <= 0.0) & (y >= -TOP)
    for px, py in P.PEGS[P.PEG_ROWS == 0]:
        ok &= np.hypot(x - px, y - py) >= CLEAR
    return ok


def region_releases(n, rng):
    """n releases at rest, uniform over the valid area across the top."""
    xs, ys = [], []
    while sum(len(v) for v in xs) < n:
        x = rng.uniform(0.5 * W, 9.5 * W, 2 * n)
        y = rng.uniform(-TOP, 0.0, 2 * n)
        ok = valid_release(x, y)
        xs.append(x[ok])
        ys.append(y[ok])
    c = np.zeros((n, 5))
    c[:, 0] = np.concatenate(xs)[:n]
    c[:, 1] = np.concatenate(ys)[:n]
    return c


def jitter_valid(w, d, n, rng):
    """n copies of release w, each moved by a random 2D error of d per axis; any that would start inside a
    pin or wall are redrawn (a real release can't be there)."""
    rep = np.repeat(w[None, :], n, 0)
    todo = np.arange(n)
    while len(todo):
        rep[todo, 0] = w[0] + rng.normal(0, d, len(todo))
        rep[todo, 1] = w[1] + rng.normal(0, d, len(todo))
        todo = todo[~valid_release(rep[todo, 0], rep[todo, 1])]
    return rep


def region_map(nx=1680):
    """Every release point across the top on a grid (1 px = 0.95 mm): the slot each one lands in
    (-1 where a chip can't be let go). results/chaos2_region_map.npz"""
    ny = int(round(nx * TOP / (9 * W)))
    xs = 0.5 * W + (np.arange(nx) + 0.5) * 9 * W / nx
    ys = -TOP + (np.arange(ny) + 0.5) * TOP / ny
    X, Y = np.meshgrid(xs, ys)
    ok = valid_release(X, Y)
    slots = np.full(X.shape, -1, dtype=np.int8)
    init = np.zeros((int(ok.sum()), 5))
    init[:, 0], init[:, 1] = X[ok], Y[ok]
    t0 = time.time()
    out = np.zeros(len(init), dtype=np.int64)
    for a in range(0, len(init), 50000):
        out[a:a + 50000] = run(init[a:a + 50000])
        print(f"map {a + 50000}/{len(init)} {time.time() - t0:.0f}s", flush=True)
    slots[ok] = out
    np.savez_compressed("results/chaos2_region_map.npz", slots=slots, x=xs, y=ys)
    print("map: share gold", float((slots == 5).sum() / ok.sum()), "stuck", float((slots == 0).sum() / ok.sum()))

    def band(ks):
        c = np.zeros_like(xs, bool)
        for k in ks:
            c |= (xs >= (k - 0.5) * W) & (xs < (k + 0.5) * W)
        sub = slots[:, c]
        return float((sub[sub >= 0] == 5).mean())
    summ = dict(gold=float((slots == 5).sum() / ok.sum()), middle=band([5]), walls=band([1, 9]),
                stuck=float((slots == 0).sum() / ok.sum()), n_points=int(ok.sum()))
    json.dump(summ, open("results/chaos2_region_map_summary.json", "w"), indent=1)
    print(summ)


def region(n_cand=6000, n_win=120, n_rep=300):
    """The replay experiment, the chip's thermal motion, how robust winners are, and the board-warming sweep,
    for winning releases from anywhere across the top. results/chaos2_region.json"""
    rng = np.random.default_rng(2027)
    cand = region_releases(n_cand, rng)
    s0 = run(cand)
    ok0 = s0 > 0
    base = float((s0[ok0] == 5).mean())
    allwin = cand[s0 == 5]
    win = allwin[:n_win]
    print(f"whole top: a fresh drop wins {base:.3f}; {len(allwin)} winners of {n_cand}", flush=True)
    out = dict(top_m=TOP, n_candidates=n_cand, base_rate=base, n_winners=len(win), n_replays_each=n_rep,
               position={}, per_winner={})
    t0 = time.time()
    for d in DELTAS:
        hits = tot = 0
        per = []
        for w in win:
            s = run(jitter_valid(w, d, n_rep, rng))
            ok = s > 0
            hits += (s[ok] == 5).sum()
            tot += ok.sum()
            per.append(float((s[ok] == 5).mean()))
        out["position"][f"{d:g}"] = hits / tot
        if d in (1e-6, 1e-10):
            out["per_winner"][f"{d:g}"] = np.round(np.sort(per), 2).tolist()
        print(f"whole top, replay {d:8.0e} m: wins again {hits / tot:.3f}   ({time.time() - t0:.0f}s)", flush=True)
    hits = tot = 0
    for w in win:
        rep = np.repeat(w[None, :], n_rep, 0)
        rep[:, 2] += rng.normal(0, SIG_V, n_rep)
        rep[:, 3] += rng.normal(0, SIG_V, n_rep)
        rep[:, 4] += rng.normal(0, SIG_W, n_rep)
        s = run(rep)
        hits += (s[s > 0] == 5).sum()
        tot += (s > 0).sum()
    out["thermal"] = hits / tot
    print(f"whole top, chip's thermal motion: {out['thermal']:.3f}", flush=True)
    probe = np.linspace(-1e-10, 1e-10, 201)
    rob = []
    for w in win:
        rep = np.repeat(w[None, :], len(probe), 0)
        rep[:, 0] += probe
        rob.append(float((run(rep) == 5).mean()))
    rob = np.array(rob)
    out["share_robust"] = float((rob > 0.9).mean())
    out["share_knife"] = float((rob < 0.3).mean())
    d = np.array(DELTAS)
    p = np.array([out["position"][f"{v:g}"] for v in DELTAS])
    sel = (d <= 1e-9) & (d >= 1e-15)
    slope = float(np.polyfit(np.log10(d[sel]), np.log10(1 - p[sel]), 1)[0])
    out["uncertainty_exponent"], out["miss_cut_per_10x"] = slope, float(10 ** -slope)
    energy, h, dT_drop = drop_heat()
    dTs = sorted(set([float(f"{v:.3g}") for v in np.logspace(-10, 0, 21)] + [float(f"{dT_drop:.3g}")]))
    n = len(allwin)
    assert (run(allwin, np.zeros(n)) == 5).all()
    out["warm_n_winners"], out["warm_dT_C"], out["warm_wins_again"] = n, dTs, []
    for dT in dTs:
        s = run(allwin, np.full(n, ALPHA * dT))
        out["warm_wins_again"].append(float((s[s > 0] == 5).mean()))
        print(f"whole top, board warmer by {dT:9.3g} C: {out['warm_wins_again'][-1]:.3f}", flush=True)
    s = run(allwin, np.full(n, ALPHA * dT_drop))
    out["warm_wins_again_drop"] = float((s[s > 0] == 5).mean())
    json.dump(out, open("results/chaos2_region.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_winner", "warm_dT_C", "warm_wins_again")},
                     indent=1))


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
    if "onevar" in what or "all" in what:
        onevar()
    if "regionmap" in what or "all" in what:
        region_map()
    if "region" in what or "all" in what:
        region()
