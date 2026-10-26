"""A random suburb for each metro:  python homes.py  -> ../media/commute/homes.json

From GeoNames (cities5000, CC BY 4.0): towns of 25,000+ people, 18-35 km in a straight line from the metro's
office, that are their own place (not a neighbourhood of the core city, GeoNames feature PPLX). One is drawn per
metro with its own fixed seed, so the draw is random but repeatable, and fixing one metro leaves the others alone.
"""
import os, io, json, math, zipfile, random, urllib.request
from metros import METROS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
URL = "https://download.geonames.org/export/dump/cities5000.zip"
RING = (18.0, 35.0)              # km from the office
MIN_POP = 25_000
SEED = 20261117
# GeoNames entries left out, and why
EXCLUDE = {4502434: "'Jackson, NJ' carries Jackson Township's population (Ocean County) at a point in Camden County"}


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def towns():
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, "cities5000.txt")
    if not os.path.exists(p):
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=120).read()))
        open(p, "wb").write(z.read("cities5000.txt"))
    out = []
    for line in open(p, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if f[6] != "P" or f[7] == "PPLX" or int(f[14] or 0) < MIN_POP:
            continue
        out.append(dict(geonameid=int(f[0]), name=f[1], lat=float(f[4]), lon=float(f[5]), country=f[8],
                        admin1=f[10], feature=f[7], population=int(f[14])))
    return out


def main():
    T = towns()
    homes = []
    for rank, name, cc, tz, (oname, olat, olon), tt in METROS:
        cand = sorted([dict(t, km=round(km((olat, olon), (t["lat"], t["lon"])), 1)) for t in T
                       if t["country"] == cc and t["geonameid"] not in EXCLUDE
                       and RING[0] <= km((olat, olon), (t["lat"], t["lon"])) <= RING[1]],
                      key=lambda t: t["geonameid"])
        pick = random.Random(f"{SEED}-{rank}").choice(cand)
        homes.append(dict(rank=rank, metro=name, tz=tz, tomtom=tt, office=dict(name=oname, lat=olat, lon=olon),
                          home=pick, candidates=len(cand)))
        print(f"{rank:2d} {name:24s} {pick['name']:28s} pop {pick['population']:>9,}  {pick['km']:5.1f} km  (of {len(cand)} suburbs)")
    os.makedirs(MEDIA, exist_ok=True)
    json.dump(dict(source="GeoNames cities5000 (CC BY 4.0), https://www.geonames.org", ring_km=RING, min_population=MIN_POP,
                   seed=SEED, metros=homes), open(os.path.join(MEDIA, "homes.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
