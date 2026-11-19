"""The river name shown under the clock during the real-time run: in each half light-second, the longest named
river the fish is swimming (by the length of its whole mapped course), if it is a big one (over 1,500 km of
OpenStreetMap line with that name). English names where OpenStreetMap has them; names in other scripts with
no English form are skipped.

    python labels.py   -> data/now_labels.json   [(t0, t1, name), ...] in light-seconds
"""
import os, json, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
C = 299_792.458
BIN = 0.5
MAJOR_KM = 1200.0
LATIN = re.compile(r"^[\x20-\x7eÀ-ɏ'’-]+$")


def main():
    z = np.load(os.path.join(DATA, "network.npz"))
    t = np.load(os.path.join(DATA, "tour.npz"))
    goff, line, km, kind = z["goff"], z["line"], z["km"], z["kind"]
    nm = np.load(os.path.join(DATA, "river_names.npz"), allow_pickle=True)
    name_en, name, wd = nm["name_en"], nm["name"], nm["wikidata"]
    tf, tb = t["tf"] / C, t["tb"] / C
    riv = np.nonzero((line >= 0) & (kind == 2))[0]
    # one river = one Wikidata item (many different rivers share a name: Black River, Río Negro)
    nms = np.array([(name_en[l] or name[l] or "").strip() for l in line[riv]], dtype=object)
    wds = np.array([wd[l] or "" for l in line[riv]], dtype=object)
    ok = np.array([bool(n) and bool(LATIN.match(n)) for n in nms]) & (wds != "")
    riv, nms, wds = riv[ok], nms[ok], wds[ok]
    uw, inv = np.unique(wds, return_inverse=True)
    tot = np.bincount(inv, weights=km[riv])
    # each item's label: its most common name
    uniq = np.empty(len(uw), dtype=object)
    for i in np.argsort(inv, kind="stable"):
        pass
    best_name = {}
    for ii, n_, kk in zip(inv, nms, km[riv]):
        d = best_name.setdefault(ii, {})
        d[n_] = d.get(n_, 0.0) + kk
    for ii, d in best_name.items():
        uniq[ii] = max(d, key=d.get)
    major = tot >= MAJOR_KM
    sel = major[inv]
    riv, inv = riv[sel], inv[sel]
    # both passes of each edge
    t_up = tf[goff[riv]]
    t_dn = tb[goff[riv + 1] - 1]
    T = np.r_[t_up, t_dn]
    I = np.r_[inv, inv]
    K = np.r_[km[riv], km[riv]]
    f = np.isfinite(T)
    T, I, K = T[f], I[f], K[f]
    nb = int(np.ceil(T.max() / BIN)) + 1
    b = (T / BIN).astype(int)
    # per bin: the named river with the largest whole-course length among those swum for at least 30 km
    swum = {}
    for bb, ii, kk in zip(b, I, K):
        swum[(bb, ii)] = swum.get((bb, ii), 0.0) + kk
    best = [None] * nb
    win = {}
    for (bb, ii), kk in swum.items():
        for w_ in (bb - 1, bb, bb + 1):
            if 0 <= w_ < nb:
                win[(w_, ii)] = win.get((w_, ii), 0.0) + kk
    for (bb, ii), kk in win.items():
        if kk < 40:
            continue
        if best[bb] is None or tot[ii] > tot[best[bb]]:
            best[bb] = ii
    out = []
    for i in range(nb):
        if best[i] is None:
            continue
        nm_ = str(uniq[best[i]])
        if out and out[-1][2] == nm_ and i * BIN - out[-1][1] <= BIN + 1e-9:
            out[-1][1] = (i + 1) * BIN
        else:
            out.append([i * BIN, (i + 1) * BIN, nm_])
    # keep labels long enough to read
    out = [o for o in out if o[1] - o[0] >= 0.99]
    json.dump(out, open(os.path.join(DATA, "now_labels.json"), "w"), indent=0, ensure_ascii=False)
    print(len(out), "labels:", ", ".join(f"{o[0]:.1f} {o[2]}" for o in out))


if __name__ == "__main__":
    main()
