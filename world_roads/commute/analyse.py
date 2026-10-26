"""Time home (and time at work) by departure time, from a router's predictions:  python analyse.py [tomtom|google] [--input FILE]
-> ../media/commute/curves_<router>.json

Works on the 5-minute departures as fetched (no interpolation), for each metro and commute:
* the drive for each departure averaged over Tue-Thu, and its range over the three days;
* worst: the departure with the longest drive, and the rush window: departures within WITHIN minutes of it;
* best and worst half hour: leave 30 minutes later; how much later do you arrive? (the least and the most);
* reference: leave 35 minutes after the classic time (5:00 p.m. home, 7:00 a.m. to work);
* flattest: the 5-minute step where arrival moves least per minute you wait;
* checks: first-in, first-out (a later departure arriving earlier than an earlier one, on any day) and the
  route changes along the evening (the router can switch to another road as traffic builds).
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
WITHIN = 5.0
REFERENCE = {"evening": 17 * 60, "morning": 7 * 60}
LATER = 35


def minute(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def hm(t):
    h, m = divmod(int(round(t)), 60)
    return f"{(h - 1) % 12 + 1}:{m:02d}"


def series(rows, direction):
    rows = [r for r in rows if r["direction"] == direction]
    days = sorted({r["day"] for r in rows})
    dep = np.array(sorted({minute(r["depart"]) for r in rows}), dtype=float)
    T = np.full((len(days), len(dep)), np.nan)
    route = [[None] * len(dep) for _ in days]
    for r in rows:
        i, k = days.index(r["day"]), int(np.searchsorted(dep, minute(r["depart"])))
        T[i, k], route[i][k] = r["minutes"], r.get("route")
    return days, dep, T, route, rows


def analyse(dep, T):
    step = int(round(dep[1] - dep[0]))
    n = 30 // step
    mean = np.nanmean(T, axis=0)
    A = dep + mean
    k = int(np.argmax(mean))
    rush = np.flatnonzero(mean >= mean[k] - WITHIN)
    gap = A[n:] - A[:-n]
    b, w = int(np.argmin(gap)), int(np.argmax(gap))
    slope = np.diff(A) / step
    f = int(np.argmin(slope))
    return dict(worst=dict(depart=dep[k], drive=mean[k], arrive=A[k], rush_from=dep[rush[0]], rush_to=dep[rush[-1]]),
                best_half_hour=dict(depart=dep[b], then=dep[b + n], later=float(gap[b])),
                worst_half_hour=dict(depart=dep[w], then=dep[w + n], later=float(gap[w])),
                flattest=dict(depart=dep[f], then=dep[f + 1], slope=float(slope[f])),
                first=dict(depart=dep[0], drive=mean[0]), last=dict(depart=dep[-1], drive=mean[-1]))


def reference(dep, T, direction):
    mean = np.nanmean(T, axis=0)
    a, b = REFERENCE[direction], REFERENCE[direction] + LATER
    i, j = int(np.searchsorted(dep, a)), int(np.searchsorted(dep, b))
    if i >= len(dep) or j >= len(dep) or dep[i] != a or dep[j] != b:
        return None
    return dict(depart=a, then=b, drive=mean[i], drive_then=mean[j], later=float(b + mean[j] - a - mean[i]))


def checks(days, dep, T, route):
    fifo = []
    for i, d in enumerate(days):
        A = dep + T[i]
        for k in np.flatnonzero(np.diff(A) < 0):
            fifo.append(dict(day=d, depart=dep[k], then=dep[k + 1], earlier_by=float(A[k] - A[k + 1])))
    switches = []
    for i, d in enumerate(days):
        for k in range(1, len(dep)):
            if route[i][k] and route[i][k - 1] and route[i][k] != route[i][k - 1]:
                switches.append(dict(day=d, at=dep[k]))
    return fifo, switches


def main(args):
    router = next((a for a in args if a in ("tomtom", "google")), "tomtom")
    src = args[args.index("--input") + 1] if "--input" in args else os.path.join(MEDIA, f"drive_times_{router}.json")
    D = json.load(open(src))
    H = {m["rank"]: m for m in json.load(open(os.path.join(MEDIA, "homes.json")))["metros"]}
    out = []
    for direction in ("evening", "morning"):
        head = "leave work" if direction == "evening" else "leave home"
        print(f"\n{direction}: {head}, {D['source']}")
        print(f"{'':26s} {'free':>5s} | {'worst':>5s} {'drive':>5s} | {'rush window':>11s} | best +30 -> later | "
              f"worst +30 | {hm(REFERENCE[direction])}>+{LATER} | flattest     | days spread | FIFO  routes")
        for rank in sorted(D["metros"], key=int):
            m = D["metros"][rank]
            days, dep, T, route, rows = series(m["rows"], direction)
            if not rows:
                continue
            a = analyse(dep, T)
            ref = reference(dep, T, direction)
            fifo, switches = checks(days, dep, T, route)
            free = np.nanmedian([r["free_flow_minutes"] for r in rows if r.get("free_flow_minutes")])
            spread = float(np.nanmax(np.nanmax(T, 0) - np.nanmin(T, 0)))
            w, bh, wh, fl = a["worst"], a["best_half_hour"], a["worst_half_hour"], a["flattest"]
            print(f"{int(rank):2d} {m['metro']:23s} {free:5.1f} | {hm(w['depart']):>5s} {w['drive']:5.1f} | "
                  f"{hm(w['rush_from']):>5s}-{hm(w['rush_to']):5s} | {hm(bh['depart']):>5s}  +{bh['later']:5.1f}    | "
                  f"{hm(wh['depart']):>5s} +{wh['later']:4.1f} | "
                  f"{('+%.1f' % ref['later']) if ref else '  -  ':>7s}  | {fl['slope']:.2f} at {hm(fl['depart']):>5s} | "
                  f"{spread:5.1f} min   | {len(fifo):3d}  {len(set(r.get('route') for r in rows))}")
            out.append(dict(rank=int(rank), metro=m["metro"], home=H[int(rank)]["home"]["name"], direction=direction,
                            days=days, depart=dep.tolist(), drive=np.round(np.nanmean(T, 0), 2).tolist(),
                            drive_min=np.round(np.nanmin(T, 0), 2).tolist(), drive_max=np.round(np.nanmax(T, 0), 2).tolist(),
                            free_flow_min=round(float(free), 2), reference=ref, fifo=fifo, route_switches=switches, **a))
    dst = os.path.join(MEDIA, f"curves_{router}.json")
    json.dump(dict(source=D["source"], fetched=D.get("fetched"), metros=out), open(dst, "w"), ensure_ascii=False,
              separators=(",", ":"), default=float)
    print("\n->", dst)


if __name__ == "__main__":
    main(sys.argv[1:])
