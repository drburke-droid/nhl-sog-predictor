"""Build one modeled day of scheduled flights from the OpenFlights route database.

OpenFlights lists which airline flies which airport pair (a 2014 snapshot), but not how often or when.
So the schedule is MODELED:
  * operated (non-codeshare, nonstop) airline routes only;
  * flights/day per airline route from a gravity model (busier airports and shorter routes get more
    frequencies), scaled so the day totals TARGET_FLIGHTS (~109,000, the 2025 daily average);
  * departures spread over a typical local-time day (06:00-23:00, long-haul in morning/evening banks),
    converted to UTC with each airport's offset; block time = distance / 820 km/h + 30 min.
A route and its reverse get identical frequencies, so each airport's departures balance its arrivals.
"""
import os, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TARGET_FLIGHTS = 109_000
R_EARTH = 6371.0088
rng = np.random.default_rng(2026)


def load():
    ap = {}
    for r in csv.reader(open(os.path.join(DATA, "airports.dat"), encoding="utf-8")):
        try:
            ap[r[0]] = dict(iata=r[4], name=r[1], city=r[2], country=r[3], lat=float(r[6]), lon=float(r[7]),
                            tz=float(r[9]) if r[9] not in ("\\N", "") else round(float(r[7]) / 15))
        except ValueError:
            continue
    rows = []
    for r in csv.reader(open(os.path.join(DATA, "routes.dat"), encoding="utf-8")):
        al, s, d, cs, stops = r[0], r[3], r[5], r[6], r[7]
        if cs == "Y" or stops != "0" or s not in ap or d not in ap or s == d:
            continue
        rows.append((al, s, d))
    rows = sorted(set(rows))
    return ap, rows


def gc_km(lat1, lon1, lat2, lon2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def local_departures(n, dist):
    """n departure times (local hours) for one airline route."""
    if dist > 6000 and n <= 2:
        banks = rng.choice([0, 1], size=n, p=[0.45, 0.55])
        return np.where(banks == 0, rng.uniform(8.5, 13.5, n), rng.uniform(17.5, 23.8, n))
    base = 6.0 + (np.arange(n) + rng.uniform(0.15, 0.85)) * (17.0 / n)
    return np.clip(base + rng.normal(0, 0.35, n), 5.5, 23.9)


def main():
    ap, rows = load()
    ids = sorted({a for _, s, d in rows for a in (s, d)})
    idx = {a: i for i, a in enumerate(ids)}
    lat = np.array([ap[a]["lat"] for a in ids]); lon = np.array([ap[a]["lon"] for a in ids])
    tz = np.array([ap[a]["tz"] for a in ids])
    S = np.array([idx[s] for _, s, _ in rows]); D = np.array([idx[d] for _, _, d in rows])
    deg = np.bincount(np.r_[S, D], minlength=len(ids)).astype(float)
    dist = gc_km(lat[S], lon[S], lat[D], lon[D])
    df = np.select([dist < 1000, dist < 3000, dist < 7000], [1.0, 0.7, 0.4], 0.25)
    w = (deg[S] * deg[D]) ** 0.4 * df
    lo, hi = 1e-4, 10.0
    for _ in range(60):                                  # scale so the day totals TARGET_FLIGHTS
        k = (lo + hi) / 2
        tot = np.maximum(1, np.round(k * w)).clip(max=40).sum()
        lo, hi = (k, hi) if tot < TARGET_FLIGHTS else (lo, k)
    freq = np.maximum(1, np.round(k * w)).clip(max=40).astype(int)
    fs, fd, dep, dk = [], [], [], []
    for s, d, n, km in zip(S, D, freq, dist):
        loc = local_departures(n, km)
        utc = (loc - tz[s]) % 24.0
        fs.append(np.full(n, s)); fd.append(np.full(n, d)); dep.append(utc); dk.append(np.full(n, km))
    fs, fd, dep, dk = map(np.concatenate, (fs, fd, dep, dk))
    dur = dk / 820.0 + 0.5
    o = np.argsort(dep)
    fs, fd, dep, dk, dur = fs[o], fd[o], dep[o], dk[o], dur[o]
    np.savez(os.path.join(DATA, "schedule.npz"), src=fs, dst=fd, dep=dep, dur=dur, km=dk,
             lat=lat, lon=lon, deg=deg, iata=np.array([ap[a]["iata"] for a in ids]))
    stats = {"airline_routes": len(rows), "airports": len(ids), "flights": int(len(fs)),
             "total_km": float(dk.sum()), "mean_km": float(dk.mean()), "median_km": float(np.median(dk)),
             "mean_block_h": float(dur.mean()), "avg_airborne": float(dur.sum() / 24)}
    json.dump(stats, open(os.path.join(DATA, "schedule_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
