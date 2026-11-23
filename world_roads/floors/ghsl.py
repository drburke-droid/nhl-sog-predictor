"""Global floor area from GHSL (JRC Global Human Settlement Layer, R2023A, epoch 2025, 30 arcsec ~ 1 km).

GHS-BUILT-V gives building volume (m^3) per cell, GHS-BUILT-S built-up surface (m^2), and the _NRES files
the non-residential share. Floor area = volume / storey height. The storey height is calibrated on New
York City, where the city's tax-lot data (PLUTO) gives the real floor area (see nyc_prep.py).

    python ghsl.py          -> data/ghsl/agg10.npz (sums on a 10x coarser grid) and printed totals
"""
import os, json
import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "ghsl")
F = {"v": "GHS_BUILT_V_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif",
     "vn": "GHS_BUILT_V_NRES_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif",
     "s": "GHS_BUILT_S_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"}
K = 10                      # aggregation factor: 30 arcsec -> 5 arcmin


def main():
    srcs = {k: rasterio.open(os.path.join(D, f)) for k, f in F.items()}
    r = srcs["v"]
    Wd, Hd = r.width, r.height
    T = r.transform
    print("grid", Wd, Hd, "origin", T.c, T.f, "cell", T.a, T.e)
    aw, ah = Wd // K, Hd // K
    agg = {k: np.zeros((ah, aw), np.float64) for k in F}
    tot = {k: 0.0 for k in F}
    STRIP = K * 64
    for y0 in range(0, ah * K, STRIP):
        h = min(STRIP, ah * K - y0)
        for k, s in srcs.items():
            a = s.read(1, window=Window(0, y0, aw * K, h)).astype(np.float64)
            tot[k] += a.sum()
            agg[k][y0 // K:(y0 + h) // K] = a.reshape(h // K, K, aw, K).sum(axis=(1, 3))
        print(f"rows {y0 + h}/{ah * K}", flush=True)
    # cells beyond the aggregated area (the last column / rows)
    for k, s in srcs.items():
        if aw * K < Wd:
            tot[k] += s.read(1, window=Window(aw * K, 0, Wd - aw * K, Hd)).astype(np.float64).sum()
        if ah * K < Hd:
            tot[k] += s.read(1, window=Window(0, ah * K, aw * K, Hd - ah * K)).astype(np.float64).sum()
    np.savez_compressed(os.path.join(D, "agg10.npz"), v=agg["v"].astype(np.float32), vn=agg["vn"].astype(np.float32),
                        s=agg["s"].astype(np.float32), x0=T.c, y0=T.f, cell=T.a * K)
    out = dict(volume_m3=tot["v"], volume_nres_m3=tot["vn"], surface_m2=tot["s"])
    json.dump(out, open(os.path.join(D, "totals.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
