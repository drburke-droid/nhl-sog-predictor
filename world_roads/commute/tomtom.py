"""Hour-by-hour traffic for each metro from the TomTom Traffic Index:  python tomtom.py
-> data/tomtom/<city>.html (cached pages), ../media/commute/tomtom_weekhour.json

Each city page of the Traffic Index (tomtom.com/traffic-index, public; robots.txt allows agents) carries, for
the metro area and for the city centre, 168 values, one per weekday and hour: the average speed v and the
free-flow speed fv (km/h) and the congestion level c (% extra travel time, = fv / v - 1). They are averages over
the index's measurement year. This script fetches the pages once, slowly (the routed metros and the map's
extra ones, metros.MAP_ONLY), and keeps only those series.
"""
import os, re, json, time, html, urllib.request
from metros import METROS, MAP_ONLY

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "tomtom")
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
URL = "https://www.tomtom.com/traffic-index/city/{key}/"
UA = "Claude-User (traffic-index research for a video; one fetch per city)"
DAYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]


def page(key):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, f"{key}.html")
    if not os.path.exists(p):
        req = urllib.request.Request(URL.format(key=key), headers={"User-Agent": UA})
        open(p, "wb").write(urllib.request.urlopen(req, timeout=60).read())
        time.sleep(2.0)
    return open(p, encoding="utf-8", errors="replace").read()


def de(v):
    """Astro's serialised props: [0, x] is a plain value (or an object of encoded values), [1, [...]] an array"""
    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], int):
        t, x = v
        if t == 0:
            return {k: de(w) for k, w in x.items()} if isinstance(x, dict) else x
        if t == 1:
            return [de(w) for w in x]
        return x
    if isinstance(v, dict):
        return {k: de(w) for k, w in v.items()}
    return v


def series(raw):
    """{'metro': {...}, 'center': {...}} with 'DAY-H' -> (v, fv, c), from the DrivingPatterns component"""
    for url, p in re.findall(r'<astro-island[^>]*?component-url="([^"]+)"[^>]*?props="([^"]*)"', raw):
        if "DrivingPatterns" not in url:
            continue
        d = de(json.loads(html.unescape(p)))
        wh = d.get("weekHourData", {})
        out = {}
        for area in ("metro", "center"):
            rows = wh.get(area) or []
            out[area] = {r["time"]: (r["v"], r["fv"], r["c"]) for r in rows}
        if all(len(out[a]) == 168 for a in out):
            return out
    raise ValueError("no weekday-hour series on the page")


def year(raw):
    m = re.search(r'YearOption (20\d\d)" class="[^"]* bg-grey5"', raw)     # the year selected on the page
    return int(m.group(1)) if m else None


def main():
    out = {}
    for rank, name, cc, tz, office, key in METROS + MAP_ONLY:
        raw = page(key)
        s = series(raw)
        out[key] = dict(rank=rank, name=name, index_year=year(raw), areas={a: {k: list(v) for k, v in s[a].items()} for a in s})
        wed = s["metro"]["WEDNESDAY-17"]
        print(f"{rank:2d} {name:24s} {key:18s} Wed 5 p.m. metro: {wed[0]:5.1f} of {wed[1]:5.1f} km/h (+{wed[2]:.0f}%)")
    os.makedirs(MEDIA, exist_ok=True)
    json.dump(dict(source="TomTom Traffic Index city pages (tomtom.com/traffic-index), weekday-hour averages",
                   fetched=time.strftime("%Y-%m-%d"), cities=out), open(os.path.join(MEDIA, "tomtom_weekhour.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
