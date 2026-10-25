"""Drive times home by departure time, from Google's Routes API:  python fetch.py [--dry-run] [rank ...]
-> data/routes/<rank>/<date>_<HHMM>.json (one cached response per request), then ../media/commute/travel_times.json

For each metro: office -> its random suburb (homes.json), departing every 5 minutes from 3:30 to 7:30 p.m. local
time on three ordinary weekdays ahead (Tue-Thu, 17-19 Nov 2026: clear of US, Canadian and Mexican holidays and of
the clock changes). Google predicts each from its traffic history (routingPreference TRAFFIC_AWARE_OPTIMAL,
trafficModel BEST_GUESS) and returns the quickest route for that departure.

Needs GOOGLE_MAPS_API_KEY with the Routes API enabled. Requests already cached are never sent again, so a run can
be stopped and resumed.
"""
import os, sys, json, time, hashlib, urllib.request, urllib.error
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "routes")
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
API = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELDS = "routes.duration,routes.staticDuration,routes.distanceMeters,routes.description,routes.polyline.encodedPolyline"
DAYS = [date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)]
FIRST, LAST, STEP = (15, 30), (19, 30), 5          # local departure times, minutes


def departures():
    t = FIRST[0] * 60 + FIRST[1]
    while t <= LAST[0] * 60 + LAST[1]:
        yield t // 60, t % 60
        t += STEP


def plan():
    H = json.load(open(os.path.join(MEDIA, "homes.json")))["metros"]
    jobs = []
    for m in H:
        tz = ZoneInfo(m["tz"])
        for d in DAYS:
            for hh, mm in departures():
                local = datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz)
                jobs.append(dict(rank=m["rank"], metro=m["metro"], day=d.isoformat(), local=f"{hh:02d}:{mm:02d}",
                                 utc=local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 o=(m["office"]["lat"], m["office"]["lon"]), h=(m["home"]["lat"], m["home"]["lon"])))
    return jobs


def body(j):
    ll = lambda p: {"location": {"latLng": {"latitude": p[0], "longitude": p[1]}}}
    return {"origin": ll(j["o"]), "destination": ll(j["h"]), "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE_OPTIMAL", "trafficModel": "BEST_GUESS",
            "departureTime": j["utc"], "computeAlternativeRoutes": False, "units": "METRIC"}


def path(j):
    return os.path.join(DATA, str(j["rank"]), f"{j['day']}_{j['local'].replace(':', '')}.json")


def call(j, key):
    req = urllib.request.Request(API, data=json.dumps(body(j)).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELDS})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:400]
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** (attempt + 1))
                continue
            raise SystemExit(f"Routes API refused {j['metro']} {j['day']} {j['local']} ({e.code}): {msg}")


def secs(s):
    return float(s.rstrip("s")) if s else None


def collect(jobs):
    out = {}
    for j in jobs:
        p = path(j)
        if not os.path.exists(p):
            continue
        r = json.load(open(p)).get("routes", [{}])[0]
        poly = r.get("polyline", {}).get("encodedPolyline", "")
        out.setdefault(str(j["rank"]), dict(metro=j["metro"], rows=[]))["rows"].append(
            dict(day=j["day"], depart=j["local"], minutes=round(secs(r.get("duration")) / 60, 2) if r.get("duration") else None,
                 free_flow_minutes=round(secs(r.get("staticDuration")) / 60, 2) if r.get("staticDuration") else None,
                 km=round(r.get("distanceMeters", 0) / 1000, 2), via=r.get("description", ""),
                 route=hashlib.sha1(poly.encode()).hexdigest()[:10] if poly else None))
    json.dump(dict(source="Google Maps Platform Routes API (computeRoutes, TRAFFIC_AWARE_OPTIMAL, BEST_GUESS)",
                   fetched=datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d"), days=[d.isoformat() for d in DAYS],
                   metros=out), open(os.path.join(MEDIA, "travel_times.json"), "w"), indent=1, ensure_ascii=False)
    return sum(len(v["rows"]) for v in out.values())


def main(args):
    jobs = plan()
    only = {int(a) for a in args if a.isdigit()}
    if only:
        jobs = [j for j in jobs if j["rank"] in only]
    todo = [j for j in jobs if not os.path.exists(path(j))]
    print(f"{len(jobs)} requests ({len(jobs) - len(todo)} cached, {len(todo)} to send)")
    if "--dry-run" in args:
        print(json.dumps(body(jobs[0]), indent=1))
        return
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if todo and not key:
        raise SystemExit("GOOGLE_MAPS_API_KEY is not set: add it to the environment's secrets, then start a new session")

    def one(j):
        os.makedirs(os.path.dirname(path(j)), exist_ok=True)
        r = call(j, key)
        json.dump(r, open(path(j), "w"))
        return j

    done = 0
    with ThreadPoolExecutor(4) as ex:
        for j in ex.map(one, todo):
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}  last: {j['metro']} {j['day']} {j['local']}", flush=True)
    print("rows:", collect(jobs), "->", os.path.join(MEDIA, "travel_times.json"))


if __name__ == "__main__":
    main(sys.argv[1:])
