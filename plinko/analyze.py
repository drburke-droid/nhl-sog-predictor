"""Monte Carlo analysis behind every number in the video.

    python3 analyze.py kernels      # physics transfer matrices over a grid of bounciness (e)   ~8 min
    python3 analyze.py fit          # fit e and the contestants' drop habits to the real show data
    python3 analyze.py sensitivity  # how the answer moves with every uncertain input           ~10 min
    python3 analyze.py technique    # release styles (steady, sloppy, thrown, spun, off-centre)  ~4 min
    python3 analyze.py chaos        # butterfly effect: how fast two nearly identical drops diverge
    python3 analyze.py final        # 24,000 drops per position at the calibrated bounciness      ~3 min
    python3 analyze.py strategy     # EV / P($10k) per drop position, 5-chip totals, small prizes
    python3 analyze.py all

Outputs go to results/ (JSON + NPZ). Seeds are fixed; reruns reproduce the same numbers.
"""
import json
import sys
import time

import numpy as np

import board as B
import galton
import physics as P

RES = "results"
E_GRID = np.round(np.arange(0.30, 0.5751, 0.025), 3)
N_PER = 6000
CATS = {"$10,000": [4], "$0": [3, 5], "$1,000": [2, 6], "$500": [1, 7], "$100": [0, 8]}


def real():
    d = json.load(open("data/real_outcomes.json"))
    slots = np.array(d["totals"]["slots_left_to_right"], float)
    return d, slots


def cat_counts(slots9):
    return np.array([slots9[idx].sum() for idx in CATS.values()])


def drop_mix(lam):
    """Contestants' drop habits: P(position 5 +- k) proportional to exp(-k / lam), symmetric.
    lam -> 0: everyone drops dead centre; lam -> infinity: drops spread evenly over all nine."""
    k = np.abs(np.arange(1, 10) - 5)
    w = np.exp(-k / max(lam, 1e-9))
    return w / w.sum()


def loglik(K, q, counts5):
    p9 = q @ K
    p5 = np.array([p9[idx].sum() for idx in CATS.values()])
    p5 = np.clip(p5, 1e-12, 1)
    return float((counts5 * np.log(p5)).sum()), p5


# ------------------------------------------------------------------------------------------- kernels
def kernels():
    out = {}
    for e in E_GRID:
        t0 = time.time()
        prm = P.params(e=float(e))
        K, tm, st, hits = P.kernel(prm, n=N_PER, seed=11)
        out[f"{e:.3f}"] = dict(K=K, t=tm, stuck=st, hits=hits)
        print(f"e={e:.3f}  P10k|5={K[4,4]:.3f}  t={tm[4]:.2f}s  stuck={st[4]:.3%}  hits={hits[4]:.1f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{RES}/kernels_e.npz", e=E_GRID, K=np.array([out[f"{e:.3f}"]["K"] for e in E_GRID]),
             t=np.array([out[f"{e:.3f}"]["t"] for e in E_GRID]),
             stuck=np.array([out[f"{e:.3f}"]["stuck"] for e in E_GRID]),
             hits=np.array([out[f"{e:.3f}"]["hits"] for e in E_GRID]))


# ------------------------------------------------------------------------------------------- fit
LAMS = np.array([0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0])


def fit():
    d, slots = real()
    counts5 = cat_counts(slots)
    n = counts5.sum()
    z = np.load(f"{RES}/kernels_e.npz")
    models = {f"physics e={e:.3f}": z["K"][i] for i, e in enumerate(z["e"])}
    models["coin-flip"] = galton.kernel()
    sat = float((counts5 * np.log(counts5 / n)).sum())     # saturated model
    table = {}
    for name, K in models.items():
        rows = []
        for lam in LAMS:
            q = drop_mix(lam)
            ll, p5 = loglik(K, q, counts5)
            rows.append(dict(lam=float(lam), q_centre=float(q[4]), deviance=2 * (sat - ll), p5=p5.tolist(),
                             ev_per_chip=float(q @ K @ B.SLOT_VALUES)))
        best = min(rows, key=lambda r: r["deviance"])
        # also: best free symmetric drop mix (non-negative, 4 dof) by a dense random search + polish
        rng = np.random.default_rng(0)
        best_free = None
        for _ in range(20000):
            a = rng.dirichlet(np.ones(5) * 0.6)
            q = np.array([a[4], a[3], a[2], a[1], a[0], a[1], a[2], a[3], a[4]])
            q[4] = a[0] * 2
            q /= q.sum()
            ll, p5 = loglik(K, q, counts5)
            dev = 2 * (sat - ll)
            if best_free is None or dev < best_free["deviance"]:
                best_free = dict(deviance=dev, q=q.tolist(), p5=p5.tolist())
        table[name] = dict(by_lambda=rows, best=best, best_free=best_free,
                           centre_only=rows[0], uniform=rows[-1])
        print(f"{name:20s} centre-only dev {rows[0]['deviance']:7.1f}   best lam={best['lam']:6.2f} "
              f"dev {best['deviance']:6.1f}  (q_centre {best['q_centre']:.2f})   free-mix dev "
              f"{best_free['deviance']:5.1f}  free q_centre {best_free['q'][4]:.2f} "
              f"q_edges(1,2,8,9) {sum(best_free['q'][i] for i in (0,1,7,8)):.2f}")
    json.dump(dict(observed5=counts5.tolist(), categories=list(CATS), n=int(n), models=table),
              open(f"{RES}/fit.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------- final
def final():
    """High-precision transfer matrix at the calibrated bounciness (24,000 drops per position)."""
    e = calibrated_e()
    prm = P.params(e=e)
    K, tm, st, hits = P.kernel(prm, n=24000, seed=31)
    np.savez(f"{RES}/kernel_final.npz", e=e, K=K, t=tm, stuck=st, hits=hits)
    ev, sd = galton.ev_table(K)
    print("e", e, "EV", np.round(ev), "P10k", np.round(K[:, 4], 4), "t", np.round(tm, 2), "stuck",
          np.round(st, 4))


# ------------------------------------------------------------------------------------------- strategy
def calibrated_e():
    f = json.load(open(f"{RES}/fit.json"))
    # pick e whose best one-parameter drop habit fits best (lowest deviance)
    best = min(((k, v["best"]["deviance"]) for k, v in f["models"].items() if k.startswith("physics")),
               key=lambda kv: kv[1])
    return float(best[0].split("=")[1])


def strategy():
    e = calibrated_e()
    z = np.load(f"{RES}/kernels_e.npz")
    out = {"calibrated_e": e, "models": {}}
    models = {"coin-flip": galton.kernel()}
    for i, ee in enumerate(z["e"]):
        models[f"physics e={ee:.3f}"] = z["K"][i]
    zf = np.load(f"{RES}/kernel_final.npz")
    models["physics tuned"] = zf["K"]
    out["tuned"] = dict(e=float(zf["e"]), n_per_position=24000, t=zf["t"].tolist(), stuck=zf["stuck"].tolist(),
                        hits=zf["hits"].tolist(), params=P.DEFAULT | {"e": float(zf["e"])}, release=P.RELEASE)
    for name, K in models.items():
        ev, sd = galton.ev_table(K)
        out["models"][name] = dict(K=K.tolist(), ev=ev.tolist(), sd=sd.tolist(), p10k=K[:, 4].tolist(),
                                   p0=(K[:, 3] + K[:, 5]).tolist(), best_position=int(np.argmax(ev) + 1))
    # 5-chip totals for the calibrated model, dropping every chip in the centre vs spreading them
    Kc = models["physics tuned"]
    rng = np.random.default_rng(7)
    sims = {}
    for label, q in {"all centre": np.eye(9)[4], "random positions": np.ones(9) / 9}.items():
        tot = np.zeros(200000)
        for _ in range(5):
            pos = rng.choice(9, size=tot.size, p=q)
            u = rng.random(tot.size)
            cdf = np.cumsum(Kc[pos], axis=1)
            slot = (u[:, None] > cdf).sum(1)
            tot += B.SLOT_VALUES[np.minimum(slot, 8)]
        sims[label] = dict(mean=float(tot.mean()), p_zero=float((tot == 0).mean()),
                           p_at_least_one_10k=float((tot >= 10000).mean()),
                           p_20k_plus=float((tot >= 20000).mean()),
                           quantiles={str(qq): float(np.quantile(tot, qq)) for qq in (0.1, 0.25, 0.5, 0.75, 0.9)})
    out["five_chips_calibrated"] = sims
    # chips are worth having: value of the small-prize phase
    d, _ = real()
    sp = d["small_prizes"]
    n_items = sp["first_digit_was_correct"] + sp["last_digit_was_correct"]
    out["small_prizes"] = dict(
        items=n_items, last_digit_correct_share=sp["last_digit_was_correct"] / n_items,
        contestant_accuracy=(sp["contestant_right_when_first"] + sp["contestant_right_when_last"]) / n_items,
        accuracy_when_first_correct=sp["contestant_right_when_first"] / sp["first_digit_was_correct"],
        accuracy_when_last_correct=sp["contestant_right_when_last"] / sp["last_digit_was_correct"])
    _, slots = real()
    out["real"] = dict(per_chip=float(slots @ B.SLOT_VALUES / slots.sum()), p10k=float(slots[4] / slots.sum()),
                       slots=slots.tolist())
    json.dump(out, open(f"{RES}/strategy.json", "w"), indent=1)
    for name in ["coin-flip", "physics tuned"]:
        m = out["models"][name]
        print(name, "EV", np.round(m["ev"], 0), "P10k", np.round(m["p10k"], 3))
    print(json.dumps(sims, indent=1))
    print(out["small_prizes"], out["real"])


# ------------------------------------------------------------------------------------------- sensitivity
def sensitivity():
    """Re-run the centre and neighbouring drops with each uncertain input pushed to its plausible limits."""
    e = calibrated_e()
    base = dict(e=e)
    variants = {
        "baseline": {},
        "e -0.05": dict(e=e - 0.05), "e +0.05": dict(e=e + 0.05),
        "pin jitter 1 deg": dict(jitter_deg=1.0), "pin jitter 6 deg": dict(jitter_deg=6.0),
        "peg friction 0.1": dict(mu=0.1), "peg friction 0.4": dict(mu=0.4),
        "tilt 5 deg": dict(tilt_deg=5.0), "tilt 20 deg": dict(tilt_deg=20.0),
        "backboard friction 0.05": dict(mu_back=0.05), "backboard friction 0.25": dict(mu_back=0.25),
        "uniform-disc chip (I=0.5mR^2)": dict(inertia=0.5), "rim-heavy chip (I=0.9mR^2)": dict(inertia=0.9),
    }
    release_variants = {
        "sloppy hand (3 cm)": dict(sx=0.03, sy=0.03), "steady hand (2 mm)": dict(sx=0.002, sy=0.002),
        "tossed (0.5 m/s)": dict(sv=0.5), "spun (10 rad/s)": dict(sw=10.0),
    }
    out = {}

    def one(label, prm, **rel):
        K, tm, st, hits = P.kernel(prm, n=4000, seed=21, positions=range(1, 6), **rel)
        ev, _ = galton.ev_table(K)
        out[label] = dict(ev=ev.tolist(), p10k=K[:, 4].tolist(), t=tm.tolist(), stuck=st.tolist(),
                          best=int(np.argmax(ev) + 1))
        print(f"{label:32s} best {np.argmax(ev)+1}  EV5 ${ev[4]:6.0f} EV4 ${ev[3]:6.0f} EV3 ${ev[2]:6.0f} "
              f"EV1 ${ev[0]:6.0f}  P10k|5 {K[4,4]:.3f}  t {tm[4]:.2f}s  stuck {st[4]:.2%}", flush=True)

    for label, kw in variants.items():
        one(label, P.params(**{**base, **kw}))
    for label, kw in release_variants.items():
        one(label, P.params(**base), **kw)
    json.dump(dict(calibrated_e=e, variants=out), open(f"{RES}/sensitivity.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------- technique
def technique():
    """Does the way you let go matter? 24,000 drops over the middle for each release style."""
    e = calibrated_e()
    prm = P.params(e=e)
    styles = {
        "normal release (1 cm scatter)": {},
        "steady hand (2 mm scatter)": dict(sx=0.002, sy=0.002),
        "sloppy hand (3 cm scatter)": dict(sx=0.03, sy=0.03),
        "thrown down at 1 m/s": dict(vy_down=1.0),
        "spun hard (10 rad/s)": dict(spin=10.0),
        "aimed 2 cm off centre": dict(x_offset=0.02),
        "dropped from 15 cm higher": dict(y0=-0.15),
    }
    out = {}
    for i, (label, kw) in enumerate(styles.items()):
        slots, times, _, n_stuck = P.run(5, 24000, prm, seed=500 + i, **kw)
        ok = slots > 0
        v = B.SLOT_VALUES[slots[ok] - 1]
        out[label] = dict(ev=float(v.mean()), se=float(v.std() / np.sqrt(ok.sum())),
                          p10k=float((slots[ok] == 5).mean()), stuck=n_stuck / len(slots),
                          t=float(times[ok].mean()))
        print(f"{label:32s} EV ${v.mean():6.0f} +- {v.std()/np.sqrt(ok.sum()):3.0f}  P10k {(slots[ok]==5).mean():.3f}"
              f"  stuck {n_stuck/len(slots):.2%}", flush=True)
    json.dump(dict(calibrated_e=e, n=24000, styles=out), open(f"{RES}/technique.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------- chaos
def chaos():
    """Twin drops: same release except x differs by delta. Track separation vs impacts; and the
    probability that both land in the same slot."""
    e = calibrated_e()
    prm = P.params(e=e)
    rng = np.random.default_rng(3)
    deltas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    out = {"calibrated_e": e, "same_slot": {}, "examples": []}
    base = P.releases(5, 3000, rng)
    seeds = rng.integers(1, 2**62, 3000).astype(np.uint64)
    s0, _, _ = P.simulate_batch(base, seeds, prm, P.GEOM, P.SEGS)
    for dlt in deltas:
        b2 = base.copy()
        b2[:, 0] += dlt
        s1, _, _ = P.simulate_batch(b2, seeds, prm, P.GEOM, P.SEGS)
        ok = (s0 > 0) & (s1 > 0)
        out["same_slot"][f"{dlt:g}"] = float((s0[ok] == s1[ok]).mean())
        print(f"delta {dlt*1000:g} mm: same slot {out['same_slot'][f'{dlt:g}']:.3f}")
    # independent chips land in the same slot with probability sum p_k^2
    K = np.load(f"{RES}/kernels_e.npz")
    i = int(np.argmin(np.abs(K["e"] - e)))
    pk = K["K"][i][4]
    out["independent_same_slot"] = float((pk ** 2).sum())
    print("independent", out["independent_same_slot"])
    # separation growth for one pair, by time (pin jitter off so that only the 1 micron matters)
    prm0 = P.params(e=e, jitter_deg=0.0, e_jitter=0.0)
    for k in range(6):
        x0 = B.drop_x(5) + rng.normal(0, 0.004)
        sa, ra = P.trajectory(x0, 0.0, 0, 0, 0.3, seed=99, prm=prm0, fps=240)
        sb, rb = P.trajectory(x0 + 1e-6, 0.0, 0, 0, 0.3, seed=99, prm=prm0, fps=240)
        m = min(len(ra), len(rb))
        sep = np.hypot(ra[:m, 1] - rb[:m, 1], ra[:m, 2] - rb[:m, 2])
        imp = np.cumsum(ra[:m, 4])
        out["examples"].append(dict(slots=[int(sa), int(sb)], t=ra[:m, 0].tolist()[::6],
                                    sep=sep.tolist()[::6], impacts=imp.tolist()[::6]))
        print("pair", sa, sb, "sep at end", sep[-1])
    json.dump(out, open(f"{RES}/chaos.json", "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    steps = ["kernels", "fit", "final", "strategy", "sensitivity", "technique", "chaos"] if what == ["all"] else what
    for s in steps:
        print(f"== {s}", flush=True)
        globals()[s]()
