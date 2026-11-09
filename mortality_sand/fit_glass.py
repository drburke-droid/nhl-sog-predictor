"""How tightly each vessel's piles must be packed to lie as natural sand inside its glass.

    python fit_glass.py          # -> data/pack.json (after build_data.py; before film.py)

At story.GRAIN people a grain, a few regions hold more people than their land can carry as sand: the
countryside fills to its height limit everywhere, thinly peopled land included (the Thar desert, Borneo),
and the coast becomes a wall. For each such vessel this finds the largest factor k (the share of its
people's grains it keeps) at which, in its most populous year, at most SAT_MAX of its countryside stands
at the limit. Everyone else keeps k = 1. The same k holds in every year, so a region's ups and downs keep
their proportions; lifted grains stay GRAIN people everywhere (sand.Sand.lift).
"""
import json, os
import numpy as np
import sand as Sd
import story as S

SAT_MAX = 0.15     # East Asia (13% at the limit: the Yangtze delta, Japan) stays unpacked


def saturation(sim, k, year):
    sim.year = year
    sim.counts = np.round(Sd.people_grains(year) * k).astype(np.int64)
    if hasattr(sim, "steep"):
        del sim.steep
    sim.cell_targets()
    out = np.zeros(Sd.NV)
    for i in range(Sd.NV):
        c = Sd.VCELLS[i]
        out[i] = (sim.bgf[c] >= S.BACKGROUND_CAP - 0.5).mean() if len(c) else 0
    return out


def main():
    sim = Sd.Sand()
    z = np.load(os.path.join(Sd.DATA, "series.npz"))
    year = float(z["years"][np.argmax(z["pop"].sum(0))])
    sat = saturation(sim, np.ones(Sd.NV), year)
    need = sat > SAT_MAX
    lo, hi = np.where(need, 0.05, 1.0), np.ones(Sd.NV)
    for _ in range(12):
        mid = np.where(need, (lo + hi) / 2, 1.0)
        ok = saturation(sim, mid, year) <= SAT_MAX
        lo = np.where(need & ok, mid, lo)
        hi = np.where(need & ~ok, mid, hi)
    pack = {}
    for i, v in enumerate(Sd.ORDER):
        if need[i]:
            k = float(np.floor(lo[i] * 100) / 100)
            pack[v] = k
            print(f"{S.VESSELS[v]['label']:22s} at the limit {sat[i] * 100:5.1f}% -> pack {k:.2f} "
                  f"(1 grain = {S.GRAIN / k:,.0f} people in its piles)")
    json.dump(pack, open(os.path.join(Sd.DATA, "pack.json"), "w"), indent=1)
    print("wrote data/pack.json:", pack)


if __name__ == "__main__":
    main()
