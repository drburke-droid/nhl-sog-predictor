"""Baer's law: the share of river length curving left, by latitude, from baer.py's per-line tallies.

Uncertainty: lines in the same place are not independent (one river is many OSM lines), so whole 1-degree
cells are resampled (cluster bootstrap).

    python baer_stats.py      -> data/baer_stats.json
"""
import os, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = [(-60, -40), (-40, -20), (-20, -5), (-5, 5), (5, 20), (20, 40), (40, 60), (60, 80)]


def load():
    rows, cols = [], None
    for p in sorted(glob.glob(os.path.join(HERE, "data", "baer", "part-*.npz"))):
        z = np.load(p)
        if len(z["rows"]):
            rows.append(z["rows"])
            cols = [str(c) for c in z["cols"]]
    R = np.vstack(rows)
    return {c: R[:, i] for i, c in enumerate(cols)}


def share(d, m, sc):
    l, r = d[f"left_m_{sc}"][m], d[f"right_m_{sc}"][m]
    return l.sum() / (l.sum() + r.sum())


def boot(d, m, sc, n=400, seed=0):
    """cluster bootstrap over 1-degree cells: (estimate, 2.5%, 97.5%)"""
    rng = np.random.default_rng(seed)
    cell = (np.floor(d["lat"][m]) * 1000 + np.floor(d["lon"][m])).astype(np.int64)
    u, inv = np.unique(cell, return_inverse=True)
    l = np.bincount(inv, d[f"left_m_{sc}"][m])
    r = np.bincount(inv, d[f"right_m_{sc}"][m])
    est = l.sum() / (l.sum() + r.sum())
    bs = []
    for _ in range(n):
        k = rng.integers(0, len(u), len(u))
        bs.append(l[k].sum() / (l[k].sum() + r[k].sum()))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(u)


def main():
    d = load()
    out = {"lines": int(len(d["lat"])), "km": float(d["len_m"].sum() / 1000)}
    for sc in (250, 1000):
        res = []
        for lo, hi in BANDS:
            m = (d["lat"] >= lo) & (d["lat"] < hi)
            if m.sum() < 50:
                continue
            est, a, b, cells = boot(d, m, sc)
            res.append({"band": [lo, hi], "lines": int(m.sum()), "cells": cells, "km": float(d["len_m"][m].sum() / 1000),
                        "share_left": est, "ci": [a, b]})
        out[f"bands_{sc}"] = res
        for name, m in (("north", d["lat"] >= 5), ("south", d["lat"] <= -5)):
            est, a, b, cells = boot(d, m, sc)
            out[f"{name}_{sc}"] = {"share_left": est, "ci": [a, b], "lines": int(m.sum()), "cells": cells}
        # turning per bend side: are left bends tighter? (turning per metre of curving length)
        for name, m in (("north", d["lat"] >= 5), ("south", d["lat"] <= -5)):
            tl = d[f"turnL_{sc}"][m].sum() / d[f"left_m_{sc}"][m].sum() * 1000
            tr = d[f"turnR_{sc}"][m].sum() / d[f"right_m_{sc}"][m].sum() * 1000
            out[f"{name}_{sc}"]["deg_per_km_left"] = tl
            out[f"{name}_{sc}"]["deg_per_km_right"] = tr
    json.dump(out, open(os.path.join(HERE, "data", "baer_stats.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
