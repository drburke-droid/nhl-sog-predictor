"""The city-average baseline (README: not used for findings), 3 to 8 p.m.:  python model.py [--radius KM]
-> ../media/commute/curves.json

The car moves along its free-flow route (routes.py) at a fraction of free-flow speed that depends on where it is
and what time it is: TomTom's city-centre profile within RADIUS km of the office (straight line), its metro-area
profile beyond. That fraction is 1 / (1 + c / 100), where c is TomTom's congestion level (extra travel time) for
the hour, averaged over Tuesday, Wednesday and Thursday. Within the day the extra-time factor 1 + c / 100 is a
smooth curve whose average over each hour is exactly TomTom's figure for that hour (a periodic cubic spline through
its running total), so no hour is flattened and no corner falls on the hour. Every departure's drive comes from
integrating that progress to the end of the route, so a car can never pass one that left earlier: time home can
flatten but never stall or fall.

For each city:
* peak: the departure with the longest drive (the worst time to leave), and the rush window around it: the
  departures whose drive is within WITHIN minutes of the worst.
* best half hour: the 30 minutes to stay on at work that cost the least evening (time home moves least), and the
  worst half hour, which costs the most.
* 5:00 vs 5:35: leave 35 minutes later than the classic quitting time; how much later are you home?
* the flattest point: where time home rises slowest per minute you stay (1 would be no help, 0 the "free" minute).
"""
import os, sys, json
import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
DAYS = ("TUESDAY", "WEDNESDAY", "THURSDAY")
FIRST, LAST = 15 * 60, 20 * 60           # departures, minutes after midnight, every minute
RADIUS = 5.0                             # km: the "centre" part of the route
WITHIN = 5.0                             # min: the rush window is the departures this close to the worst drive
DT = 2.0                                 # s, integration step


def factor(area):
    """minute of the day -> extra-time factor, Tue-Thu, smooth and true to every hourly average (wraps at midnight)"""
    f = np.array([np.mean([1 + area[f"{d}-{h}"][2] / 100 for d in DAYS]) for h in range(24)])
    run = 60 * (np.r_[0, np.cumsum(f - f.mean())])                # running total (factor x minutes) less its trend
    run[-1] = 0.0                                                   # ... which comes back to zero at midnight
    rate = CubicSpline(np.arange(25) * 60.0, run, bc_type="periodic").derivative()
    return lambda minute: np.maximum(1.0, f.mean() + rate(np.mod(minute, 1440)))


def slowdown(area):
    """minute of the day -> fraction of free-flow speed"""
    k = factor(area)
    return lambda minute: 1.0 / k(minute)


def drive(route, prof, radius):
    """drive minutes for each departure in FIRST..LAST"""
    s = np.array([x["s"] for x in route["stretches"]])
    s *= route["free_flow_min"] * 60 / s.sum()           # the stretches leave out turn costs: spread them evenly
    centre = np.array([x["r"] <= radius for x in route["stretches"]])
    edges = np.r_[0, np.cumsum(s)]                       # free-flow seconds at the start of each stretch
    total = edges[-1]
    rc, rm = slowdown(prof["center"]), slowdown(prof["metro"])
    dep = np.arange(FIRST, LAST + 1, dtype=float)
    p = np.zeros_like(dep)                               # free-flow seconds of route covered
    t = np.zeros_like(dep)                               # seconds since departure
    done = np.full(dep.shape, np.nan)
    while np.isnan(done).any():
        now = dep + t / 60
        i = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(s) - 1)
        rate = np.where(centre[i], rc(now), rm(now))
        q = p + rate * DT
        hit = np.isnan(done) & (q >= total)
        done[hit] = t[hit] + (total - p[hit]) / rate[hit]
        p, t = q, t + DT
    return dep, done / 60, total / 60, float(s[centre].sum() / 60)


def analyse(dep, T):
    A = dep + T
    k = int(np.argmax(T))
    rush = np.flatnonzero(T >= T[k] - WITHIN)
    gap = A[30:] - A[:-30]                               # time home moves if you leave 30 minutes later
    b, w = int(np.argmin(gap)), int(np.argmax(gap))
    at = lambda hh, mm: int(hh * 60 + mm - FIRST)
    slope = np.gradient(A, dep)                          # minutes later home per minute later leaving
    m = int(np.argmin(slope))
    return dict(peak=dict(depart=dep[k], drive=T[k], home=A[k], rush_from=dep[rush[0]], rush_to=dep[rush[-1]]),
                best_half_hour=dict(depart=dep[b], then=dep[b] + 30, later_home=float(gap[b])),
                worst_half_hour=dict(depart=dep[w], then=dep[w] + 30, later_home=float(gap[w])),
                five_vs_535=dict(drive_500=T[at(17, 0)], drive_535=T[at(17, 35)], later_home=A[at(17, 35)] - A[at(17, 0)]),
                at_3=T[0], at_8=T[-1], flattest=dict(depart=dep[m], slope=float(slope[m])))


def hm(minute):
    h, m = divmod(int(round(minute)), 60)
    return f"{(h - 1) % 12 + 1}:{m:02d}"


def main(args):
    radius = float(args[args.index("--radius") + 1]) if "--radius" in args else RADIUS
    R = json.load(open(os.path.join(MEDIA, "routes.json")))["metros"]
    TT = json.load(open(os.path.join(MEDIA, "tomtom_weekhour.json")))["cities"]
    out = []
    print(f"centre radius {radius} km; departures {hm(FIRST)}-{hm(LAST)} p.m.\n")
    print(f"{'':26s} {'free':>5s} | {'3pm':>4s} {'worst':>5s} {'drive':>5s} {'8pm':>4s} | {'rush window':>11s} | "
          f"best +30   home  | worst +30  home | 5:00>5:35 | flattest")
    for r in R:
        dep, T, ff, ffc = drive(r, TT[r["tomtom"]]["areas"], radius)
        a = analyse(dep, T)
        pk, bh, wh, f5 = a["peak"], a["best_half_hour"], a["worst_half_hour"], a["five_vs_535"]
        print(f"{r['rank']:2d} {r['metro']:23s} {ff:5.1f} | {a['at_3']:4.0f} {hm(pk['depart']):>5s} {pk['drive']:5.1f} {a['at_8']:4.0f} | "
              f"{hm(pk['rush_from']):>5s}-{hm(pk['rush_to']):5s} | {hm(bh['depart']):>5s}  +{bh['later_home']:4.1f} | "
              f"{hm(wh['depart']):>5s}  +{wh['later_home']:4.1f} |   +{f5['later_home']:4.1f}  | {a['flattest']['slope']:.2f} at "
              f"{hm(a['flattest']['depart'])}")
        out.append(dict(rank=r["rank"], metro=r["metro"], home=r["home"]["name"], km=r["km"], free_flow_min=round(ff, 2),
                        centre_free_flow_min=round(ffc, 2), depart=dep.tolist(), drive=np.round(T, 3).tolist(), **a))
    json.dump(dict(model=dict(days=DAYS, centre_radius_km=radius, first=FIRST, last=LAST, step_s=DT),
                   sources=["TomTom Traffic Index 2025 (weekday-hour congestion, metro and city centre)",
                            "OSRM on OpenStreetMap (free-flow route)", "GeoNames (suburbs)"],
                   metros=out), open(os.path.join(MEDIA, "curves.json"), "w"), ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1:])
