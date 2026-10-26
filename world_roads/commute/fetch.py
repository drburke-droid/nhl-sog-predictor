"""Drive times by departure time on each commute, from Google's Routes API:  python fetch.py [--dry-run] [rank ...]
-> data/routes/<rank>/<direction>_<date>_<HHMM>.json (one cached response per request),
   then ../media/commute/drive_times_google.json

The same departures as tomtom_routes.py (plan.py: both commutes, every 5 minutes, Tue-Thu 17-19 Nov 2026).
Google predicts each from its traffic history (routingPreference TRAFFIC_AWARE_OPTIMAL, trafficModel BEST_GUESS)
and returns the quickest route for that departure.

Needs GOOGLE_MAPS_API_KEY with the Routes API enabled. Requests already cached are never sent again, so a run can
be stopped and resumed.
"""
import os, sys, json, time, hashlib, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from plan import jobs, DAYS, MEDIA, HERE

DATA = os.path.join(HERE, "data", "routes")
API = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELDS = "routes.duration,routes.staticDuration,routes.distanceMeters,routes.description,routes.polyline.encodedPolyline"


def body(j):
    ll = lambda p: {"location": {"latLng": {"latitude": p[0], "longitude": p[1]}}}
    return {"origin": ll(j["o"]), "destination": ll(j["d"]), "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE_OPTIMAL", "trafficModel": "BEST_GUESS",
            "departureTime": j["utc"], "computeAlternativeRoutes": False, "units": "METRIC"}


def path(j):
    return os.path.join(DATA, str(j["rank"]), f"{j['direction']}_{j['day']}_{j['local'].replace(':', '')}.json")


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
            raise SystemExit(f"Routes API refused {j['metro']} {j['direction']} {j['day']} {j['local']} ({e.code}): {msg}")


def secs(s):
    return float(s.rstrip("s")) if s else None


def collect(all_jobs):
    out = {}
    for j in all_jobs:
        p = path(j)
        if not os.path.exists(p):
            continue
        r = json.load(open(p)).get("routes", [{}])[0]
        if not r.get("duration"):
            continue
        poly = r.get("polyline", {}).get("encodedPolyline", "")
        out.setdefault(str(j["rank"]), dict(metro=j["metro"], rows=[]))["rows"].append(
            dict(direction=j["direction"], day=j["day"], depart=j["local"], minutes=round(secs(r["duration"]) / 60, 2),
                 free_flow_minutes=round(secs(r.get("staticDuration")) / 60, 2) if r.get("staticDuration") else None,
                 km=round(r.get("distanceMeters", 0) / 1000, 2), via=r.get("description", ""),
                 route=hashlib.sha1(poly.encode()).hexdigest()[:10] if poly else None))
    json.dump(dict(source="Google Maps Platform Routes API (computeRoutes, TRAFFIC_AWARE_OPTIMAL, BEST_GUESS)",
                   fetched=time.strftime("%Y-%m-%d"), days=[d.isoformat() for d in DAYS], metros=out),
              open(os.path.join(MEDIA, "drive_times_google.json"), "w"), indent=1, ensure_ascii=False)
    return sum(len(v["rows"]) for v in out.values())


def main(args):
    only = {int(a) for a in args if a.isdigit()}
    every = jobs(only)
    todo = [j for j in every if not os.path.exists(path(j))]
    print(f"{len(every)} requests ({len(every) - len(todo)} cached, {len(todo)} to send)")
    if "--dry-run" in args:
        print(json.dumps(body(every[0]), indent=1))
        return
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if todo and not key:
        raise SystemExit("GOOGLE_MAPS_API_KEY is not set: add it in the environment's settings, then start a new session")

    def one(j):
        os.makedirs(os.path.dirname(path(j)), exist_ok=True)
        json.dump(call(j, key), open(path(j), "w"))
        return j

    done = 0
    with ThreadPoolExecutor(4) as ex:
        for j in ex.map(one, todo):
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}  last: {j['metro']} {j['direction']} {j['day']} {j['local']}", flush=True)
    print("rows:", collect(every), "->", os.path.join(MEDIA, "drive_times_google.json"))


if __name__ == "__main__":
    main(sys.argv[1:])
