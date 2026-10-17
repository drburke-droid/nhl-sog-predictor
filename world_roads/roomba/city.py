"""The Roomba photon in Calgary: every street inside the city limits (the Calgary film's network), until
the last one has been driven.

    python city.py [n_walks]     -> data/calgary_cover.json (+ data/calgary_first_<seed>.npy for the map)
    python city.py many 100      -> 100 walks of each kind, summaries only -> data/calgary_cover_many.json
    python city.py feature 31    -> first/log/path files for the walk shown in the video
"""
import os, sys, json, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "calgary")); sys.path.insert(0, ROOT)
import cover as CV  # noqa: E402

DATA = os.path.join(HERE, "data")


def graph():
    import city as CAL                                   # calgary/city.py
    G = CAL.build()
    ea, eb, elen = G["ea"], G["eb"], G["elen"]
    n = len(G["nx"])
    comp = G["comp"]
    big = np.bincount(comp[ea], weights=elen).argmax()           # the component with the most road
    keep = comp[ea] == big
    total_km = float(elen.sum())
    ea, eb, elen = ea[keep], eb[keep], elen[keep]
    used, inv = np.unique(np.r_[ea, eb], return_inverse=True)
    m = len(ea)
    ea, eb = inv[:m], inv[m:]
    nx, ny = G["nx"][used], G["ny"][used]
    start = int(np.argmin(np.hypot(nx, ny)))                      # downtown
    return dict(n=len(used), m=m, ea=ea, eb=eb, elen=elen, nx=nx, ny=ny, start=start,
                km=float(elen.sum()), total_km=total_km)


def main(n_walks=8):
    os.makedirs(DATA, exist_ok=True)
    g = graph()
    print(f"Calgary: {g['km']:,.0f} km of streets in the main network ({g['km'] / g['total_km']:.1%} of {g['total_km']:,.0f} km), "
          f"{g['m']:,} segments, {g['n']:,} points", flush=True)
    indptr, adj, eid, alen = CV.csr(g["n"], g["ea"], g["eb"], g["elen"])
    np.savez(os.path.join(DATA, "calgary_graph.npz"), ea=g["ea"], eb=g["eb"], elen=g["elen"], nx=g["nx"], ny=g["ny"], start=g["start"])
    rows = []
    for uturn in (False, True):
        for seed in range(n_walks):
            t0 = time.time()
            steps, km, first, log, last = CV.cover_walk(indptr, adj, eid, alen, g["m"], g["start"], seed, uturn,
                                                       100_000, 20_000_000_000)
            done = bool(np.isfinite(first).all())
            sec = km / CV.C_KMS
            rows.append(dict(seed=seed, uturn=uturn, done=done, steps=int(steps), km=float(km), seconds=sec,
                             x_network=float(km / g["km"]), last=int(last)))
            if not uturn:
                np.save(os.path.join(DATA, f"calgary_first_{seed}.npy"), first.astype(np.float32))
                np.save(os.path.join(DATA, f"calgary_log_{seed}.npy"), log)
            print(f"{'plain' if uturn else 'no U-turn'} seed {seed}: {km:,.0f} km = {CV.fmt_time(sec)} at c "
                  f"({km / g['km']:,.0f}x the network), {steps:,} steps, {time.time() - t0:.1f}s", flush=True)
    out = dict(network_km=g["km"], total_km=g["total_km"], segments=g["m"], walks=rows)
    for uturn in (False, True):
        s = sorted(r["seconds"] for r in rows if r["uturn"] == uturn)
        out["median_s_" + ("plain" if uturn else "nonback")] = float(np.median(s))
    json.dump(out, open(os.path.join(DATA, "calgary_cover.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k.startswith("median")})


def feature(seed):
    """Everything the video needs for one walk: per-road first-driven distance, progress log and the full path."""
    z = np.load(os.path.join(DATA, "calgary_graph.npz"))
    m, n = len(z["ea"]), int(max(z["ea"].max(), z["eb"].max())) + 1
    indptr, adj, eid, alen = CV.csr(n, z["ea"], z["eb"], z["elen"])
    steps, km, first, log, last = CV.cover_walk(indptr, adj, eid, alen, m, int(z["start"]), seed, False, 100_000, 20_000_000_000)
    path = CV.walk_path(indptr, adj, eid, alen, m, int(z["start"]), seed, False, int(steps) + 10)
    assert len(path) == steps and abs(z["elen"][path].sum() - km) < 1e-3
    np.save(os.path.join(DATA, f"calgary_first_{seed}.npy"), first.astype(np.float32))
    np.save(os.path.join(DATA, f"calgary_log_{seed}.npy"), log)
    np.save(os.path.join(DATA, f"calgary_path_{seed}.npy"), path)
    print(f"seed {seed}: {km:,.0f} km = {CV.fmt_time(km / CV.C_KMS)}, {steps:,} steps; last road {last}")


def many(n_walks):
    """More walks on the saved graph, summaries only, for a steadier median and the spread."""
    z = np.load(os.path.join(DATA, "calgary_graph.npz"))
    m, n = len(z["ea"]), int(max(z["ea"].max(), z["eb"].max())) + 1
    indptr, adj, eid, alen = CV.csr(n, z["ea"], z["eb"], z["elen"])
    km_net = float(z["elen"].sum())
    out = dict(network_km=km_net)
    for uturn in (False, True):
        s = []
        t0 = time.time()
        for seed in range(n_walks):
            steps, km, first, log, last = CV.cover_walk(indptr, adj, eid, alen, m, int(z["start"]), seed, uturn,
                                                       10_000_000, 20_000_000_000)
            assert np.isfinite(first).all()
            s.append(km / CV.C_KMS)
        key = "plain" if uturn else "nonback"
        out["seconds_" + key] = s
        q = np.percentile(s, [0, 10, 50, 90, 100])
        out["pct_" + key] = dict(zip(("min", "p10", "median", "p90", "max"), map(float, q)))
        print(key, out["pct_" + key], f"{time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(os.path.join(DATA, "calgary_cover_many.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "feature":
        feature(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "many":
        many(int(sys.argv[2]))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
