"""Drive times by departure time on each commute, from TomTom's Routing API:  python tomtom_routes.py [--dry-run] [--check] [--more] [rank ...]
-> data/tomtom_routes/<rank>/<direction>_<date>_<HHMM>.json (one trimmed response per request),
   then ../media/commute/drive_times_tomtom.json
   --more: the extra suburbs (plan.more_jobs) -> data/tomtom_routes/<rank>/<sector>/..., drive_times_tomtom_more.json

For every departure in plan.py (both commutes, every 5 minutes, Tue-Thu 17-19 Nov 2026), TomTom's Calculate
Route gives the quickest route for that departure and its drive time from TomTom's time-dependent historic
traffic on that route (summary.historicTrafficTravelTimeInSeconds; the dates are weeks ahead, so no live
traffic is in it), plus the same route on empty roads.

Needs the TomTom key as TOMTOM_API_KEY or API_KEY (developer.tomtom.com: free account, no credit card; the Routing API's free allowance is
20,000 requests a month, and the whole run is 6,600). Requests already cached are never sent again, so a run can
be stopped and resumed. --check sends just one request (Toronto, Tuesday, 4:50 p.m.) and prints it.
"""
import os, sys, json, time, hashlib, threading, urllib.request, urllib.error, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from plan import jobs, more_jobs, DAYS, MEDIA, HERE

DATA = os.path.join(HERE, "data", "tomtom_routes")
API = "https://api.tomtom.com/routing/1/calculateRoute/{o[0]},{o[1]}:{d[0]},{d[1]}/json"
PARAMS = dict(travelMode="car", routeType="fastest", traffic="true", computeTravelTimeFor="all",
              routeRepresentation="polyline", maxAlternatives="0")
RATE = 4.0                                              # requests per second, at most
KEEP = 160                                              # points of each route kept, for maps and route changes
# BC and Alberta keep daylight time from 1 Nov 2026 (tzdata 2026e), but TomTom still turns their clocks back, and its
# historic traffic goes by its own local time. So for them ask for the local clock time at TomTom's offset.
TOMTOM_OFFSET = {"America/Vancouver": "-08:00", "America/Edmonton": "-07:00"}


def depart_at(j):
    return j["at"][:-6] + TOMTOM_OFFSET[j["tz"]] if j["tz"] in TOMTOM_OFFSET else j["at"]


def check(j, s):
    """TomTom's departure, in its local time, must be the clock time asked for"""
    if s["departureTime"][11:16] != j["local"]:
        raise SystemExit(f"{j['metro']} {j['direction']} {j['day']}: asked for {j['local']} local, TomTom departs "
                         f"{s['departureTime']}; check its time zone for {j['tz']}")


def url(j, key):
    q = dict(PARAMS, departAt=depart_at(j))
    if key:
        q["key"] = key
    return API.format(o=j["o"], d=j["d"]) + "?" + urllib.parse.urlencode(q)


def path(j):
    return os.path.join(DATA, str(j["rank"]), j.get("sector", ""), f"{j['direction']}_{j['day']}_{j['local'].replace(':', '')}.json")


class Throttle:
    def __init__(self, rate):
        self.gap, self.next, self.lock = 1.0 / rate, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


def trim(r):
    """keep the summary and a thinned line, plus a fingerprint of the full line to spot route changes"""
    route = r["routes"][0]
    pts = [(p["latitude"], p["longitude"]) for leg in route["legs"] for p in leg["points"]]
    step = max(1, len(pts) // KEEP)
    return dict(summary=route["summary"], points=len(pts), line=[[round(a, 5), round(b, 5)] for a, b in pts[::step] + pts[-1:]],
                route=hashlib.sha1(json.dumps([[round(a, 4), round(b, 4)] for a, b in pts]).encode()).hexdigest()[:10])


def call(j, key, throttle):
    for attempt in range(6):
        throttle.wait()
        try:
            with urllib.request.urlopen(url(j, key), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if key:
                msg = msg.replace(key, "<key>")
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** (attempt + 1))
                continue
            raise SystemExit(f"TomTom refused {j['metro']} {j['direction']} {j['day']} {j['local']} ({e.code}): {msg}")


def collect(all_jobs, dst):
    out = {}
    for j in all_jobs:
        p = path(j)
        if not os.path.exists(p):
            continue
        c = json.load(open(p))
        s = c["summary"]
        check(j, s)
        m = out.setdefault(str(j["rank"]), dict(metro=j["metro"], rows=[]))
        historic = "historicTrafficTravelTimeInSeconds" in s
        m["rows"].append(dict(direction=j["direction"], day=j["day"], depart=j["local"],
                              **{k: j[k] for k in ("sector", "home") if k in j},
                              minutes=round(s["historicTrafficTravelTimeInSeconds" if historic else "travelTimeInSeconds"] / 60, 2),
                              historic=historic,
                              planned_minutes=round(s["travelTimeInSeconds"] / 60, 2),
                              free_flow_minutes=round(s["noTrafficTravelTimeInSeconds"] / 60, 2),
                              km=round(s["lengthInMeters"] / 1000, 2), route=c["route"]))
    json.dump(dict(source="TomTom Routing API (Calculate Route, routeType fastest, traffic true, departAt; "
                          "historicTrafficTravelTimeInSeconds)",
                   fetched=time.strftime("%Y-%m-%d"), days=sorted({j["day"] for j in all_jobs}), metros=out),
              open(dst, "w"), indent=1, ensure_ascii=False)
    return sum(len(v["rows"]) for v in out.values())


def main(args):
    only = {int(a) for a in args if a.isdigit()}
    plan = more_jobs if "--more" in args else jobs
    dst = os.path.join(MEDIA, "drive_times_tomtom_more.json" if "--more" in args else "drive_times_tomtom.json")
    every = plan(only)
    if "--check" in args:
        every = [j for j in jobs({7}) if j["direction"] == "evening" and j["day"] == "2026-11-17" and j["local"] == "16:50"]
    todo = [j for j in every if not os.path.exists(path(j))]
    print(f"{len(every)} requests ({len(every) - len(todo)} cached, {len(todo)} to send, about "
          f"{len(todo) / RATE / 60:.0f} min)")
    if "--dry-run" in args:
        print(url(every[0], None) + "&key=...")
        return
    key = os.environ.get("TOMTOM_API_KEY") or os.environ.get("API_KEY")
    if todo and not key:
        raise SystemExit("No TomTom key: set TOMTOM_API_KEY (or API_KEY) in the environment's settings, then start a new session")
    throttle = Throttle(RATE)

    def one(j):
        os.makedirs(os.path.dirname(path(j)), exist_ok=True)
        c = trim(call(j, key, throttle))
        check(j, c["summary"])
        json.dump(c, open(path(j), "w"))
        return j

    done = 0
    with ThreadPoolExecutor(4) as ex:
        for j in ex.map(one, todo):
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}  last: {j['metro']} {j['direction']} {j['day']} {j['local']}", flush=True)
    if "--check" in args:
        s = json.load(open(path(every[0])))["summary"]
        print(f"Toronto, Tue 17 Nov, leave King and Bay 4:50 p.m.: {s['historicTrafficTravelTimeInSeconds'] / 60:.0f} min "
              f"(empty roads {s['noTrafficTravelTimeInSeconds'] / 60:.0f} min, {s['lengthInMeters'] / 1000:.1f} km), "
              f"home {s['arrivalTime']}")
        return
    print("rows:", collect(plan(only), dst), "->", dst)


if __name__ == "__main__":
    main(sys.argv[1:])
