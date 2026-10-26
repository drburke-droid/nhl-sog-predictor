"""A random suburb for each metro:  python homes.py [--more]  -> ../media/commute/homes.json (homes_more.json)

From GeoNames (cities5000, CC BY 4.0): towns of 25,000+ people, 18-35 km in a straight line from the metro's
office, that are their own place (not a neighbourhood of the core city, GeoNames feature PPLX). One is drawn per
metro with its own fixed seed, so the draw is random but repeatable, and fixing one metro leaves the others alone.

--more adds a suburb in each other compass direction (90 degree sectors centred on N, E, S, W; the main suburb
keeps its own): towns of 10,000+ people, 15-40 km out, one drawn per metro and sector with its own seed. Each draw
is checked with OSRM on empty roads (router.project-osrm.org): a drive that takes a ferry, or is more than 1.6
times the straight line (round a bay or a lake), is set aside and the next town in the draw is tried. A sector
with no such town (open water, a border) is left empty.
"""
import os, io, sys, json, math, time, zipfile, random, urllib.request
from metros import METROS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
URL = "https://download.geonames.org/export/dump/cities5000.zip"
RING = (18.0, 35.0)              # km from the office
MIN_POP = 25_000
SEED = 20261117
SECTORS = ["N", "E", "S", "W"]
MORE_RING, MORE_POP, MAX_DETOUR = (15.0, 40.0), 10_000, 1.6
OSRM = "https://router.project-osrm.org/route/v1/driving/{o};{h}?overview=false&steps=true&alternatives=false"
UA = "Claude-User (commute research for a video; one request per suburb)"
# GeoNames entries left out, and why
EXCLUDE = {4502434: "'Jackson, NJ' carries Jackson Township's population (Ocean County) at a point in Camden County"}


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def towns(min_pop=MIN_POP):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, "cities5000.txt")
    if not os.path.exists(p):
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=120).read()))
        open(p, "wb").write(z.read("cities5000.txt"))
    out = []
    for line in open(p, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if f[6] != "P" or f[7] == "PPLX" or int(f[14] or 0) < min_pop:
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


def sector(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return SECTORS[int(((math.degrees(math.atan2(y, x)) + 405) % 360) // 90)]


def drive(rank, office, t):
    """OSRM's quickest drive on empty roads, office -> town: (km, takes a ferry)"""
    p = os.path.join(DATA, "osrm_more", f"{rank}_{t['geonameid']}.json")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        url = OSRM.format(o=f"{office[1]},{office[0]}", h=f"{t['lon']},{t['lat']}")
        r = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60).read())
        if r.get("code") != "Ok":
            raise SystemExit(f"OSRM refused {t['name']}: {r.get('code')} {r.get('message')}")
        json.dump(r, open(p, "w"))
        time.sleep(1.5)
    r = json.load(open(p))["routes"][0]
    return r["distance"] / 1000, any(s["mode"] == "ferry" for leg in r["legs"] for s in leg["steps"])


def more():
    T = towns(MORE_POP)
    H = {m["rank"]: m for m in json.load(open(os.path.join(MEDIA, "homes.json")))["metros"]}
    out = []
    for rank, name, cc, tz, (oname, olat, olon), tt in METROS:
        office, m = (olat, olon), H[rank]
        main = dict(m["home"], sector=sector(office, (m["home"]["lat"], m["home"]["lon"])), main=True)
        homes = [main]
        for sec in SECTORS:
            if sec == main["sector"]:
                continue
            cand = sorted([dict(t, km=round(km(office, (t["lat"], t["lon"])), 1)) for t in T
                           if t["country"] == cc and t["geonameid"] not in EXCLUDE and sector(office, (t["lat"], t["lon"])) == sec
                           and MORE_RING[0] <= km(office, (t["lat"], t["lon"])) <= MORE_RING[1]], key=lambda t: t["geonameid"])
            random.Random(f"{SEED}-{rank}-{sec}").shuffle(cand)
            skipped = []
            for t in cand:
                road, ferry = drive(rank, office, t)
                if ferry or road > MAX_DETOUR * t["km"]:
                    skipped.append(f"{t['name']} ({'ferry' if ferry else f'{road:.0f} km by road'})")
                    continue
                homes.append(dict(t, sector=sec, main=False, road_km=round(road, 1), candidates=len(cand), skipped=skipped))
                break
            pick = homes[-1] if homes[-1]["sector"] == sec else None
            print(f"{rank:3d} {name:24s} {sec}  " + (f"{pick['name']:26s} pop {pick['population']:>9,} {pick['km']:5.1f} km "
                  f"(road {pick['road_km']:5.1f}; of {len(cand)})" if pick else f"{'-':26s} (of {len(cand)})")
                  + (f"  set aside: {', '.join(skipped)}" if skipped else ""))
        out.append(dict(rank=rank, metro=name, tz=tz, tomtom=tt, office=dict(name=oname, lat=olat, lon=olon), homes=homes))
    json.dump(dict(source="GeoNames cities5000 (CC BY 4.0), https://www.geonames.org; drives checked with OSRM "
                          "(router.project-osrm.org, OpenStreetMap data, ODbL)", ring_km=MORE_RING, min_population=MORE_POP,
                   max_detour=MAX_DETOUR, seed=SEED, metros=out),
              open(os.path.join(MEDIA, "homes_more.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    more() if "--more" in sys.argv[1:] else main()
