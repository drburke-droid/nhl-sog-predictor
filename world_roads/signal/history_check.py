"""Do the beacon chains of history pass our sightline test? Each station's usual identification is
snapped to the highest summit within SNAP km in the global terrain grid, then each hop is checked
with the same physics as the search (no clearance margin, 10 m fires).

    python history_check.py   -> data/history.json
"""
import os, json, math
import numpy as np
import chain as CH
import edges as ED

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SNAP = 6.0
AESCHYLUS = [("Mount Ida", 39.705, 26.863), ("Lemnos (Hermes' crag)", 39.93, 25.25), ("Mount Athos", 40.157, 24.326),
             ("Makistos (Euboea)", 38.62, 23.85), ("Messapion", 38.43, 23.50), ("Kithairon", 38.18, 23.25),
             ("Aigiplanktos", 38.02, 23.12), ("Arachnaion", 37.62, 22.95), ("Argos (the palace)", 37.6385, 22.7163)]


def snap(mos, row0, W, lat, lon, r_km, keep=False):
    if keep:
        P = CH.xyz(np.array([lat]), np.array([lon]))[0]
        return lat, lon, CH.terrain(mos, row0, W, *P) * 1000
    best = (-1, lat, lon)
    for dy in np.arange(-r_km, r_km + 0.01, 0.3):
        for dx in np.arange(-r_km, r_km + 0.01, 0.3):
            if dx * dx + dy * dy > r_km * r_km:
                continue
            la = lat + dy / 111.2; lo = lon + dx / (111.2 * math.cos(math.radians(lat)))
            P = CH.xyz(np.array([la]), np.array([lo]))[0]
            h = CH.terrain(mos, row0, W, *P) * 1000
            if h > best[0]:
                best = (h, la, lo)
    return best[1], best[2], best[0]


def main():
    mos, row0, W = ED.load_global()
    out = []
    pts = []
    for i, (nm, la, lo) in enumerate(AESCHYLUS):
        la2, lo2, h = snap(mos, row0, W, la, lo, SNAP, keep=(i == len(AESCHYLUS) - 1))
        pts.append((nm, la2, lo2, h))
    for i, (nm, la, lo, h) in enumerate(pts):
        rec = dict(name=nm, lat=la, lon=lo, h_m=round(h))
        if i:
            a, b = pts[i - 1], pts[i]
            Pa = CH.xyz(np.array([a[1]]), np.array([a[2]]))[0]; Pb = CH.xyz(np.array([b[1]]), np.array([b[2]]))[0]
            ha, hb = a[3] / 1000 + CH.PYRE_KM, b[3] / 1000 + CH.PYRE_KM
            ok = CH.los(mos, row0, W, Pa, ha, Pb, hb, CH.REFF, 0.25, 0.5, 0.0)
            d = float(CH.gc(Pa, Pb))
            rec.update(hop_km=round(d, 1), visible=bool(ok), horizons_km=round(float(CH.reach(ha) + CH.reach(hb)), 1))
        out.append(rec)
        print(rec)
    json.dump({"aeschylus": out}, open(os.path.join(DATA, "history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
