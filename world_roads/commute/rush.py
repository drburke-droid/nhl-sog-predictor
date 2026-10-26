"""When is rush hour in each metro, and why does it differ?  python rush.py
-> ../media/commute/rush.json, rush_clock.png, rush_routes.png, rush_trip_length.png, rush_friday.png

From three sources already fetched:
* the TomTom Traffic Index (tomtom_weekhour.json, tomtom.py): city-wide congestion (% extra travel time) for each
  weekday hour, averaged over 2025, metro area. Each value is taken as the middle of its hour, and a peak's time is
  refined with a parabola through the peak hour and its neighbours, so it is good to about a quarter of an hour.
* the routed drive times of every suburb, main (drive_times_tomtom.json) and others (drive_times_tomtom_more.json,
  up to four per metro, one per compass direction), all taken alike: Wednesday 18 Nov, every 10 minutes. A route's
  worst departure is refined with a parabola through its neighbours too.
* the main suburbs' Tue-Thu averages (curves_tomtom.json, analyse.py), for the costliest half hour.

What it measures:
* rush hour, city-wide: the Tue-Thu morning (5-11 a.m.) and evening (noon-9 p.m.) peaks, the hours each stays above
  half its peak, a midday peak between them if there is one, the worst hour of the week, and Friday's evening and
  Monday's morning against Tue-Thu;
* rush hour by route: each metro's worst time to leave on each of its routes, their median and spread, and the
  middle of the trip at the worst departure against the city-wide peak;
* trip length: the worst time to leave against how long the drive is then (a longer trip has to start earlier to
  meet the same peak), over all US and Canadian routes and within each metro (each route against its metro's mean);
* the sun: the clock time of solar noon (2025 average, daylight time included), against the city-wide peaks.
"""
import os, json
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from findings import MEDIA, SURFACE, INK, INK2, MUTED, GRID, BLUE, CONTEXT, hm, plain
from metros import METROS
from plan import MORE_DAY, MORE_STEP

MX = {1, 12, 13}                                        # the Mexican metros, set apart in the fits
MIDWEEK = ["TUESDAY", "WEDNESDAY", "THURSDAY"]
WEEK = ["MONDAY"] + MIDWEEK + ["FRIDAY", "SATURDAY", "SUNDAY"]


def minute(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def profile(s, days):
    return np.array([np.mean([s[f"{d}-{h}"][2] for d in days]) for h in range(24)])


def refine(a, b, c):
    """offset of a parabola's top from the middle of three equally spaced values, in steps"""
    return float(np.clip(0.5 * (a - c) / (a - 2 * b + c), -0.5, 0.5)) if a - 2 * b + c else 0.0


def peak(p, h0, h1):
    """(clock minutes, congestion) of the highest hour in h0..h1, refined with a parabola through its neighbours"""
    k = h0 + int(np.argmax(p[h0:h1 + 1]))
    return (k + 0.5 + refine(p[k - 1], p[k], p[(k + 1) % 24])) * 60, float(p[k])


def solar_noon(tz, lon):
    """clock time of mean solar noon, averaged over 2025 (the equation of time averages out over a year)"""
    z = ZoneInfo(tz)
    offs = [(datetime(2025, 1, 1, 12, tzinfo=z) + timedelta(days=n)).utcoffset().total_seconds() / 3600 for n in range(365)]
    return 12 * 60 + float(np.mean([(o * 15 - lon) * 4 for o in offs]))


def city(key, w, lon, tz):
    s = w["areas"]["metro"]
    p, fri, mon, sat = profile(s, MIDWEEK), profile(s, ["FRIDAY"]), profile(s, ["MONDAY"]), profile(s, ["SATURDAY"])
    am, amv = peak(p, 5, 11)
    pm, pmv = peak(p, 12, 21)
    fpm, fpmv = peak(fri, 12, 21)
    _, mamv = peak(mon, 5, 11)
    _, satv = peak(sat, 8, 21)
    mid = [h for h in range(int(am // 60) + 1, int(pm // 60)) if p[h] > p[h - 1] and p[h] > p[h + 1]]
    worst = max(((d, h) for d in WEEK for h in range(24)), key=lambda dh: s[f"{dh[0]}-{dh[1]}"][2])
    return dict(key=key, am=am, am_level=amv, pm=pm, pm_level=pmv, gap=pm - am,
                am_hours=int(sum(p[4:12] >= amv / 2)), pm_hours=int(sum(p[12:24] >= pmv / 2)),
                midday=[dict(hour=h, level=round(float(p[h]), 1)) for h in mid],
                friday_pm=fpm, friday_shift=fpm - pm, friday_level=fpmv / pmv - 1, monday_am_level=mamv / amv - 1,
                saturday_vs_weekday=satv / pmv, worst_hour=dict(day=worst[0], hour=worst[1], level=s[f"{worst[0]}-{worst[1]}"][2]),
                solar_noon=solar_noon(tz, lon), profile=p.round(1).tolist())


def routes():
    """one dict per route and commute: its Wednesday drives every 10 minutes and its worst departure"""
    H = {m["rank"]: m for m in json.load(open(os.path.join(MEDIA, "homes_more.json")))["metros"]}
    out = []
    for f in ("drive_times_tomtom.json", "drive_times_tomtom_more.json"):
        for rank, m in json.load(open(os.path.join(MEDIA, f)))["metros"].items():
            main = next(h for h in H[int(rank)]["homes"] if h["main"])
            rows = [r for r in m["rows"] if r["day"] == MORE_DAY.isoformat() and minute(r["depart"]) % MORE_STEP == 0]
            for sector, direction in sorted({(r.get("sector", main["sector"]), r["direction"]) for r in rows}):
                R = sorted((r for r in rows if r.get("sector", main["sector"]) == sector and r["direction"] == direction),
                           key=lambda r: minute(r["depart"]))
                dep, T = np.array([minute(r["depart"]) for r in R], float), np.array([r["minutes"] for r in R])
                k = int(np.argmax(T))
                d = refine(T[k - 1], T[k], T[k + 1]) if 0 < k < len(T) - 1 else 0.0
                worst = dep[k] + d * MORE_STEP
                out.append(dict(rank=int(rank), metro=m["metro"], sector=sector, home=R[0].get("home", main["name"]),
                                main="sector" not in R[0], direction=direction, worst=worst, drive=float(T[k]),
                                free=float(np.median([r["free_flow_minutes"] for r in R])), mid_trip=worst + float(T[k]) / 2,
                                km=float(np.median([r["km"] for r in R]))))
    return out


def fit(x, y):
    b, a = np.polyfit(x, y, 1)
    return dict(slope=float(b), intercept=float(a), r=float(np.corrcoef(x, y)[0, 1]), n=len(x))


def within(R):
    """the fit of worst departure on drive within metros: each route against its metro's mean (metros with 2+ routes)"""
    xs, ys = [], []
    for rank in {r["rank"] for r in R}:
        S = [r for r in R if r["rank"] == rank]
        if len(S) > 1:
            x, y = np.array([r["drive"] for r in S]), np.array([r["worst"] for r in S])
            xs += list(x - x.mean()); ys += list(y - y.mean())
    xs, ys = np.array(xs), np.array(ys)
    return dict(slope=float((xs * ys).sum() / (xs * xs).sum()), r=float(np.corrcoef(xs, ys)[0, 1]), n=len(xs))


def clock_chart(rows, dst):
    rows = sorted(rows, key=lambda r: r["pm"])
    Z = np.array([r["profile"] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 9), dpi=160)
    cmap = LinearSegmentedColormap.from_list("congestion", ["#f4f7fb", "#9cc0ec", BLUE, "#0b2a55"])
    im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=0, vmax=100, extent=(0, 24, len(rows), 0), interpolation="nearest")
    for i, r in enumerate(rows):
        for t in (r["am"], r["pm"]):
            ax.plot(t / 60, i + 0.5, "o", ms=5, color=SURFACE, mec=INK, mew=0.8)
        ax.text(24.25, i + 0.5, hm(r["pm"]), va="center", fontsize=9, color=INK2)
    ax.text(24.25, -0.35, "evening\npeak", fontsize=8.5, color=MUTED, va="bottom")
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([r["metro"] for r in rows], fontsize=9.5)
    ax.set_xticks(range(4, 25, 2))
    ax.set_xticklabels(["4", "6", "8", "10", "noon", "2", "4", "6", "8", "10", "midnight"])
    ax.set_xlim(4, 24)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.06, aspect=50)
    cb.set_label("extra travel time, % over empty roads (metro area, 2025 average)", color=INK2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5)
    fig.suptitle("When is rush hour? Congestion by hour, Tuesday-Thursday", x=0.06, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.06, 0.925, "Sorted by the evening peak; dots mark the morning and evening peaks. TomTom Traffic Index, "
             "city-wide averages.", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.17, right=0.93, top=0.9, bottom=0.08)
    fig.savefig(dst, facecolor=SURFACE)
    plt.close(fig)


def routes_chart(rows, routed, dst):
    """each metro's routes: the worst time to leave (dots) and the middle of the trip then (rings), against the
    city-wide peak (bar)"""
    rows = sorted(rows, key=lambda r: r["pm"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), dpi=160, sharey=True)
    for ax, direction, peak_key, (lo, hi), title in zip(axes, ("evening", "morning"), ("pm", "am"),
                                                       ((15 * 60, 19 * 60), (6 * 60 + 30, 9 * 60 + 30)),
                                                       ("Leave work (evening)", "Leave home (morning)")):
        for i, r in enumerate(rows):
            R = [x for x in routed if x["rank"] == r["rank"] and x["direction"] == direction]
            ax.plot([min(x["worst"] for x in R), max(x["worst"] for x in R)], [i, i], color=CONTEXT, lw=2, zorder=1,
                    solid_capstyle="round")
            ax.plot([x["worst"] for x in R], [i] * len(R), "o", ms=6.5, color=BLUE, mec=SURFACE, mew=1.2, zorder=3,
                    label="worst time to leave, one dot per route" if i == 0 else None)
            ax.plot([x["mid_trip"] for x in R], [i] * len(R), "o", ms=6.5, mfc="none", mec=BLUE, mew=1, zorder=2,
                    label="middle of the trip, leaving then" if i == 0 else None)
            ax.plot([r[peak_key]] * 2, [i - 0.35, i + 0.35], color=INK, lw=2, zorder=4, solid_capstyle="butt",
                    label="city-wide peak (Traffic Index)" if i == 0 else None)
        ticks = np.arange(lo, hi + 1, 30)
        ax.set_xticks(ticks)
        ax.set_xticklabels([hm(t) for t in ticks])
        ax.set_xlim(lo, hi)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.7)
        plain(ax)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r["metro"] for r in rows], fontsize=9.5)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.16, 0.905), frameon=False,
               fontsize=9.5, ncol=3)
    fig.suptitle("Rush hour by route: up to four suburbs per metro", x=0.05, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.05, 0.93, "Office downtown to a suburb in each compass direction, Wednesday 18 Nov, every 10 minutes. "
             "TomTom Routing API; city-wide peak from the TomTom Traffic Index.", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.85, bottom=0.05, wspace=0.12)
    fig.savefig(dst, facecolor=SURFACE)
    plt.close(fig)


def trip_chart(routed, fits, dst):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.75), dpi=160)
    for ax, direction, title in zip(axes, ("evening", "morning"), ("Leave work (evening)", "Leave home (morning)")):
        R = [r for r in routed if r["direction"] == direction]
        for mx in (True, False):
            S = [r for r in R if (r["rank"] in MX) == mx]
            ax.scatter([r["drive"] for r in S], [r["worst"] for r in S], s=40, color=CONTEXT if mx else BLUE,
                       edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Mexico" if mx else "US and Canada")
        f, w = fits[direction]["pooled"], fits[direction]["within"]
        us = [r["drive"] for r in R if r["rank"] not in MX]
        x = np.array([min(us), max(us)])
        ax.plot(x, f["intercept"] + f["slope"] * x, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(0.98, 0.96, f"US and Canada, {f['n']} routes: {abs(f['slope']) * 10:.0f} min earlier for each\n10 min more "
                f"driving (r = {f['r']:.2f}); within each metro, {abs(w['slope']) * 10:.0f} min (r = {w['r']:.2f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK)
        lo, hi = min(r["worst"] for r in R), max(r["worst"] for r in R)
        ticks = np.arange(30 * np.floor(lo / 30), hi + 30, 30)
        ax.set_yticks(ticks)
        ax.set_yticklabels([hm(t) for t in ticks])
        ax.set_ylim(ticks[0] - 5, ticks[-1] + 25)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_xlabel("drive at the worst departure (minutes)")
        ax.grid(color=GRID, lw=0.7)
        plain(ax)
    axes[0].set_ylabel("worst time to leave")
    axes[1].legend(loc="lower left", frameon=False, fontsize=9.5)
    fig.suptitle("The longer the drive, the earlier the worst time to leave", x=0.05, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.05, 0.9, "Office downtown to up to four suburbs per metro, one dot per route, Wednesday 18 Nov. "
             "TomTom Routing API.", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.1, wspace=0.18)
    fig.savefig(dst, facecolor=SURFACE)
    plt.close(fig)


def friday_chart(rows, dst):
    rows = sorted(rows, key=lambda r: r["friday_shift"])
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=160)
    for i, r in enumerate(rows):
        ax.plot([r["pm"], r["friday_pm"]], [i, i], color=CONTEXT, lw=2, zorder=1, solid_capstyle="round")
        ax.plot(r["pm"], i, "o", ms=8, color=CONTEXT, mec=SURFACE, mew=1.5, zorder=2, label="Tuesday-Thursday" if i == 0 else None)
        ax.plot(r["friday_pm"], i, "o", ms=8, color=BLUE, mec=SURFACE, mew=1.5, zorder=3, label="Friday" if i == 0 else None)
        ax.text(19.6 * 60, i, f"{r['friday_shift']:+.0f} min", va="center", ha="right", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["metro"] for r in rows], fontsize=9.5)
    ax.invert_yaxis()
    ticks = np.arange(15 * 60, 19 * 60 + 1, 30)
    ax.set_xticks(ticks)
    ax.set_xticklabels([hm(t) for t in ticks])
    ax.set_xlim(15 * 60 - 10, 19.6 * 60)
    ax.grid(axis="x", color=GRID, lw=0.7)
    plain(ax)
    ax.set_xlabel("evening peak, city-wide (p.m.)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.19, 0.9), frameon=False, fontsize=10, ncol=2)
    fig.suptitle("On Fridays, the evening rush comes earlier", x=0.05, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.05, 0.915, "Peak hour of congestion, metro area, Friday against Tuesday-Thursday. TomTom Traffic Index, "
             "2025 averages.", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.85, bottom=0.08)
    fig.savefig(dst, facecolor=SURFACE)
    plt.close(fig)


def main():
    W = json.load(open(os.path.join(MEDIA, "tomtom_weekhour.json")))["cities"]
    C = json.load(open(os.path.join(MEDIA, "curves_tomtom.json")))["metros"]
    rows = []
    for rank, name, cc, tz, (oname, lat, lon), key in METROS:
        rows.append(dict(rank=rank, metro=name, **city(key, W[key], lon, tz)))
    routed = routes()
    us = [r for r in rows if r["rank"] not in MX]
    fits = {d: dict(pooled=fit([r["drive"] for r in routed if r["direction"] == d and r["rank"] not in MX],
                               [r["worst"] for r in routed if r["direction"] == d and r["rank"] not in MX]),
                    within=within([r for r in routed if r["direction"] == d and r["rank"] not in MX])) for d in ("evening", "morning")}
    sun = {p: fit([r["solar_noon"] for r in us], [r[p] for r in us]) for p in ("am", "pm")}
    by_city = []
    for r in rows:
        for direction, p in (("evening", "pm"), ("morning", "am")):
            R = [x for x in routed if x["rank"] == r["rank"] and x["direction"] == direction]
            by_city.append(dict(rank=r["rank"], metro=r["metro"], direction=direction, routes=len(R),
                                worst_median=float(np.median([x["worst"] for x in R])), worst_from=min(x["worst"] for x in R),
                                worst_to=max(x["worst"] for x in R), mid_trip_median=float(np.median([x["mid_trip"] for x in R])),
                                city_peak=r[p]))

    print(f"{'':27s} {'morning':>13s} {'evening':>13s}  gap    hours>half  midday      Friday pm     Mon am  Sat/wkday  "
          f"worst hour of the week   solar noon")
    for r in sorted(rows, key=lambda r: r["pm"]):
        md = ", ".join(f"{m['hour']}h {m['level']:.0f}%" for m in r["midday"]) or "-"
        w = r["worst_hour"]
        print(f"{r['rank']:3d} {r['metro']:23s} {hm(r['am']):>5s} {r['am_level']:5.1f}%  {hm(r['pm']):>5s} {r['pm_level']:5.1f}%  "
              f"{hm(r['gap']):>5s}  {r['am_hours']:2d} / {r['pm_hours']:2d}   {md:11s} {r['friday_shift']:+5.0f} min {r['friday_level']:+4.0%}  "
              f"{r['monday_am_level']:+4.0%}   {r['saturday_vs_weekday']:.2f}     {w['day'][:3].title()} {w['hour']:2d}h {w['level']:5.1f}%"
              f"        {hm(r['solar_noon'])}")
    print(f"\nby route, Wed 18 Nov every {MORE_STEP} min: worst time to leave (median of routes, range), middle of the trip "
          f"then (median), city-wide peak")
    for r in sorted(rows, key=lambda r: r["pm"]):
        e, m = (next(b for b in by_city if b["rank"] == r["rank"] and b["direction"] == d) for d in ("evening", "morning"))
        print(f"{r['rank']:3d} {r['metro']:23s} {e['routes']} routes | evening {hm(e['worst_median']):>5s} ({hm(e['worst_from'])}-"
              f"{hm(e['worst_to'])}) mid {hm(e['mid_trip_median'])} city {hm(e['city_peak'])} | morning {hm(m['worst_median'])} "
              f"({hm(m['worst_from'])}-{hm(m['worst_to'])}) mid {hm(m['mid_trip_median'])} city {hm(m['city_peak'])}")
    fr = [r["friday_shift"] for r in us]
    print(f"\nUS and Canada (n={len(us)}): evening peak {hm(min(r['pm'] for r in us))}-{hm(max(r['pm'] for r in us))}, "
          f"morning {hm(min(r['am'] for r in us))}-{hm(max(r['am'] for r in us))}; Mexico evening "
          f"{hm(min(r['pm'] for r in rows if r['rank'] in MX))}-{hm(max(r['pm'] for r in rows if r['rank'] in MX))}")
    print(f"evening peak above the morning's in {sum(r['pm_level'] > r['am_level'] for r in rows)} of {len(rows)}; "
          f"evening at least as long in {sum(r['pm_hours'] >= r['am_hours'] for r in rows)}")
    print(f"Friday evening peak, US and Canada: {np.mean(fr):+.0f} min on average (median {np.median(fr):+.0f}, earlier in "
          f"{sum(f < 0 for f in fr)} of {len(fr)}); Monday morning peak lower than Tue-Thu in "
          f"{sum(r['monday_am_level'] < 0 for r in rows)} of {len(rows)} (average {np.mean([r['monday_am_level'] for r in rows]):+.0%})")
    for d, f in fits.items():
        p, w = f["pooled"], f["within"]
        print(f"trip length, {d}: worst departure {p['slope']:+.2f} min per minute of drive, r = {p['r']:+.2f} ({p['n']} US and "
              f"Canadian routes); within metros {w['slope']:+.2f}, r = {w['r']:+.2f}")
    for d in ("evening", "morning"):
        B = [b for b in by_city if b["direction"] == d]
        gap = np.array([b["mid_trip_median"] - b["city_peak"] for b in B])
        print(f"{d}: middle of the trip at the worst departure minus the city-wide peak, median of metros {np.median(gap):+.0f} min "
              f"(middle half {np.percentile(gap, 25):+.0f} to {np.percentile(gap, 75):+.0f}); worst departure minus city peak "
              f"{np.median([b['worst_median'] - b['city_peak'] for b in B]):+.0f} min")
    for p, f in sun.items():
        print(f"solar noon vs city-wide {p} peak, US and Canada: slope {f['slope']:+.2f}, r = {f['r']:+.2f}")
    worst30 = max((c for c in C if c["direction"] == "morning"), key=lambda c: c["worst_half_hour"]["later"])
    w30 = worst30["worst_half_hour"]
    print(f"the costliest half hour: {worst30['metro']}, leaving home at {hm(w30['then'])} instead of {hm(w30['depart'])} a.m. "
          f"gets you in {w30['later']:.1f} min later")

    json.dump(dict(sources=["TomTom Traffic Index city pages, 2025 weekday-hour averages, metro area",
                            "TomTom Routing API, historic traffic, Wed 18 Nov 2026 every 10 minutes, up to four suburbs per metro"],
                   cities=rows, routes=routed, by_city=by_city, trip_length=fits, solar_noon=sun),
              open(os.path.join(MEDIA, "rush.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    clock_chart(rows, os.path.join(MEDIA, "rush_clock.png"))
    routes_chart(rows, routed, os.path.join(MEDIA, "rush_routes.png"))
    trip_chart(routed, fits, os.path.join(MEDIA, "rush_trip_length.png"))
    friday_chart(rows, os.path.join(MEDIA, "rush_friday.png"))


if __name__ == "__main__":
    main()
