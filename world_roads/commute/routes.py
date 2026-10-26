"""The drive home on empty roads, office -> suburb, from OSRM on OpenStreetMap:  python routes.py
-> data/osrm/<rank>.json (cached responses), ../media/commute/routes.json

The public OSRM demo server (router.project-osrm.org, car profile) gives the quickest route on empty roads and,
for every stretch of it, its length and its time at the profile's speeds. That is the free-flow drive; tomtom.py
gives how much slower each hour of the day is. 20 requests in all, one every 1.5 s, each sent only once.
"""
import os, json, time, math, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "osrm")
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
URL = ("https://router.project-osrm.org/route/v1/driving/{o};{h}"
       "?overview=full&geometries=geojson&annotations=distance,duration&alternatives=false&steps=false")
UA = "Claude-User (commute research for a video; 20 requests)"


def km(a, b):
    """great-circle distance between (lat, lon) points"""
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    return 2 * 6371.0 * math.asin(math.sqrt(math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def fetch(m):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, f"{m['rank']}.json")
    if not os.path.exists(p):
        o, h = m["office"], m["home"]
        url = URL.format(o=f"{o['lon']},{o['lat']}", h=f"{h['lon']},{h['lat']}")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        if r.get("code") != "Ok":
            raise SystemExit(f"OSRM refused {m['metro']}: {r.get('code')} {r.get('message')}")
        json.dump(r, open(p, "w"))
        time.sleep(1.5)
    return json.load(open(p))


def main():
    H = json.load(open(os.path.join(MEDIA, "homes.json")))["metros"]
    out = []
    for m in H:
        r = fetch(m)["routes"][0]
        leg = r["legs"][0]["annotation"]
        pts = [(lat, lon) for lon, lat in r["geometry"]["coordinates"]]
        office = (m["office"]["lat"], m["office"]["lon"])
        # each stretch: its free-flow seconds and how far its midpoint is from the office, in a straight line
        seg = [dict(s=round(t, 2), m=round(d, 1), r=round(km(office, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)), 3))
               for t, d, a, b in zip(leg["duration"], leg["distance"], pts, pts[1:])]
        out.append(dict(rank=m["rank"], metro=m["metro"], tz=m["tz"], tomtom=m["tomtom"], office=m["office"], home=m["home"],
                        km=round(r["distance"] / 1000, 2), free_flow_min=round(r["duration"] / 60, 2),
                        line=[[round(a, 5), round(b, 5)] for a, b in pts], stretches=seg))
        print(f"{m['rank']:2d} {m['metro']:24s} -> {m['home']['name']:24s} {r['distance'] / 1000:5.1f} km  {r['duration'] / 60:5.1f} min "
              f"({len(seg)} stretches)")
    json.dump(dict(source="OSRM (router.project-osrm.org, car profile) on OpenStreetMap data, (c) OpenStreetMap contributors, ODbL",
                   fetched=time.strftime("%Y-%m-%d"), metros=out),
              open(os.path.join(MEDIA, "routes.json"), "w"), ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
