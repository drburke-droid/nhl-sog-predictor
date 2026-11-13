"""Release map: which slot a chip lands in as a function of exactly where it is let go.

The board is made a perfect, repeatable machine (no pin imperfections, no release scatter); only the
horizontal release point changes. Zooming in shows how fine a robot's aim would have to be.

    python3 releasemap.py      -> results/releasemap.npz, results/releasemap.json
"""
import json

import numpy as np

import board as B
import physics as P

E = json.load(open("results/strategy.json"))["calibrated_e"]
PRM = P.params(e=E, jitter_deg=0.0, e_jitter=0.0)
X0 = B.drop_x(5)
# (half-width, points): 9 cm (the whole drop gap), 1 mm, 10 um, 100 nm, 1 nm
LEVELS = [(0.045, 20001), (5e-4, 20001), (5e-6, 20001), (5e-8, 20001), (5e-10, 20001)]


def scan(half, n, centre=0.0):
    dx = np.linspace(-half, half, n) + centre
    init = np.zeros((n, 5))
    init[:, 0] = X0 + dx
    slots, times, hits = P.simulate_batch(init, np.ones(n, dtype=np.uint64), PRM, P.GEOM, P.SEGS)
    return dx, slots


def runs(slots, step):
    """Lengths (in metres) of unbroken stretches that land in the same slot."""
    edges = np.flatnonzero(np.diff(slots) != 0)
    bounds = np.concatenate([[-1], edges, [len(slots) - 1]])
    lens = np.diff(bounds) * step
    vals = slots[bounds[1:]]
    return lens, vals


if __name__ == "__main__":
    out, arrays = {"e": E, "levels": []}, {}
    centre = 0.0
    for k, (half, n) in enumerate(LEVELS):
        dx, s = scan(half, n, centre)
        step = 2 * half / (n - 1)
        lens, vals = runs(s, step)
        top = lens[vals == 5]
        lev = dict(half_width_m=half, step_m=step, n=n, p10k=float((s == 5).mean()),
                   p_stuck=float((s == 0).mean()), n_runs=int(len(lens)),
                   median_run_m=float(np.median(lens)), median_10k_run_m=float(np.median(top)) if len(top) else None,
                   widest_10k_run_m=float(top.max()) if len(top) else None, centre_m=centre)
        out["levels"].append(lev)
        arrays[f"dx{k}"], arrays[f"s{k}"] = dx, s.astype(np.int8)
        print(f"+-{half:.0e} m step {step:.1e}: P10k {lev['p10k']:.3f}  runs {len(lens)}  median run "
              f"{lev['median_run_m']:.2e} m  median $10k run {lev['median_10k_run_m']}  widest $10k "
              f"{lev['widest_10k_run_m']}", flush=True)
        # zoom next level onto the middle of the widest $10k stretch near the centre, if any
        if len(top):
            bounds = np.concatenate([[-1], np.flatnonzero(np.diff(s) != 0), [len(s) - 1]])
            mids = [(dx[bounds[i] + 1] + dx[bounds[i + 1]]) / 2 for i in range(len(bounds) - 1)
                    if s[bounds[i + 1]] == 5]
            nxt = LEVELS[k + 1][0] if k + 1 < len(LEVELS) else 0
            # an edge of a $10k stretch: zooming there shows the boundary structure
            j = int(np.argmin(np.abs(np.array(mids) - centre)))
            i_run = [i for i in range(len(bounds) - 1) if s[bounds[i + 1]] == 5][j]
            centre = float(dx[bounds[i_run + 1]])   # right edge of that stretch
    np.savez_compressed("results/releasemap.npz", **arrays)
    json.dump(out, open("results/releasemap.json", "w"), indent=1)
