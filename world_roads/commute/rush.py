"""When is rush hour in each metro, and why does it differ?  python rush.py
-> ../media/commute/rush.json, rush_clock.png, rush_trip_length.png, rush_friday.png

From two sources already fetched:
* the TomTom Traffic Index (tomtom_weekhour.json, tomtom.py): city-wide congestion (% extra travel time) for each
  weekday hour, averaged over 2025, metro area. Each value is taken as the middle of its hour, and a peak's time is
  refined with a parabola through the peak hour and its neighbours, so it is good to about a quarter of an hour.
* the routed drive times (curves_tomtom.json, analyse.py): one commute per metro, every 5 minutes, Tue-Thu.

What it measures:
* rush hour, city-wide: the Tue-Thu morning (5-11 a.m.) and evening (noon-9 p.m.) peaks, the hours each stays above
  half its peak, a midday peak between them if there is one, the worst hour of the week, and Friday's evening and
  Monday's morning against Tue-Thu;
* trip length: on one route, the worst time to leave against how long the drive is then (a longer trip has to start
  earlier to meet the same peak), fitted over the US and Canada;
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

MX = {1, 12, 13}                                        # the Mexican metros, set apart in the fits
MIDWEEK = ["TUESDAY", "WEDNESDAY", "THURSDAY"]
WEEK = ["MONDAY"] + MIDWEEK + ["FRIDAY", "SATURDAY", "SUNDAY"]
LABEL = ["New York", "Toronto", "Vancouver", "Detroit", "Houston", "Atlanta", "Calgary", "Edmonton", "Boston"]
LEFT = {("Monterrey", "morning")}                       # labels set to the left of their dot, clear of a neighbour


def profile(s, days):
    return np.array([np.mean([s[f"{d}-{h}"][2] for d in days]) for h in range(24)])


def peak(p, h0, h1):
    """(clock minutes, congestion) of the highest hour in h0..h1, refined with a parabola through its neighbours"""
    k = h0 + int(np.argmax(p[h0:h1 + 1]))
    a, b, c = p[k - 1], p[k], p[(k + 1) % 24]
    d = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0.0
    return (k + 0.5 + float(np.clip(d, -0.5, 0.5))) * 60, float(b)


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


def fit(x, y):
    b, a = np.polyfit(x, y, 1)
    return dict(slope=float(b), intercept=float(a), r=float(np.corrcoef(x, y)[0, 1]), n=len(x))


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


def trip_chart(routed, fits, dst):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.75), dpi=160)
    for ax, direction, title in zip(axes, ("evening", "morning"), ("Leave work (evening)", "Leave home (morning)")):
        R = [r for r in routed if r["direction"] == direction]
        for mx in (True, False):
            S = [r for r in R if (r["rank"] in MX) == mx]
            ax.scatter([r["drive"] for r in S], [r["worst"] for r in S], s=60, color=CONTEXT if mx else BLUE,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Mexico" if mx else "US and Canada")
        f = fits[direction]
        x = np.array([min(r["drive"] for r in R if r["rank"] not in MX), max(r["drive"] for r in R if r["rank"] not in MX)])
        ax.plot(x, f["intercept"] + f["slope"] * x, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        right = max(r["drive"] for r in R) - 8
        for r in R:
            if r["metro"] in LABEL or r["rank"] in MX:
                left = r["drive"] > right or (r["metro"], direction) in LEFT
                ax.annotate(r["metro"], (r["drive"], r["worst"]), xytext=(-6 if left else 6, 3), textcoords="offset points",
                            fontsize=9, color=INK2, ha="right" if left else "left")
        ax.text(0.98, 0.96, f"US and Canada: {abs(f['slope']) * 10:.0f} min earlier\nfor each 10 min more driving (r = {f['r']:.2f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK)
        lo, hi = min(r["worst"] for r in R), max(r["worst"] for r in R)
        ticks = np.arange(15 * np.floor(lo / 15), hi + 15, 15)
        ax.set_yticks(ticks)
        ax.set_yticklabels([hm(t) for t in ticks])
        ax.set_ylim(ticks[0] - 5, ticks[-1] + 5)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_xlabel("drive at the worst departure (minutes)")
        ax.grid(color=GRID, lw=0.7)
        plain(ax)
    axes[0].set_ylabel("worst time to leave")
    axes[1].legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.suptitle("The longer the drive, the earlier the worst time to leave", x=0.05, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.05, 0.9, "One commute per metro, office downtown and a random suburb, Tuesday-Thursday average. "
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
    routed = [dict(rank=c["rank"], metro=c["metro"], direction=c["direction"], worst=c["worst"]["depart"],
                   drive=c["worst"]["drive"], mid_trip=c["worst"]["depart"] + c["worst"]["drive"] / 2,
                   worst_half_hour=c["worst_half_hour"]["later"], worst_half_hour_at=c["worst_half_hour"]["depart"]) for c in C]
    us = [r for r in rows if r["rank"] not in MX]
    fits = {d: fit([r["drive"] for r in routed if r["direction"] == d and r["rank"] not in MX],
                   [r["worst"] for r in routed if r["direction"] == d and r["rank"] not in MX]) for d in ("evening", "morning")}
    sun = {p: fit([r["solar_noon"] for r in us], [r[p] for r in us]) for p in ("am", "pm")}

    print(f"{'':27s} {'morning':>13s} {'evening':>13s}  gap    hours>half  midday      Friday pm     Mon am  Sat/wkday  "
          f"worst hour of the week   solar noon")
    for r in sorted(rows, key=lambda r: r["pm"]):
        md = ", ".join(f"{m['hour']}h {m['level']:.0f}%" for m in r["midday"]) or "-"
        w = r["worst_hour"]
        print(f"{r['rank']:3d} {r['metro']:23s} {hm(r['am']):>5s} {r['am_level']:5.1f}%  {hm(r['pm']):>5s} {r['pm_level']:5.1f}%  "
              f"{hm(r['gap']):>5s}  {r['am_hours']:2d} / {r['pm_hours']:2d}   {md:11s} {r['friday_shift']:+5.0f} min {r['friday_level']:+4.0%}  "
              f"{r['monday_am_level']:+4.0%}   {r['saturday_vs_weekday']:.2f}     {w['day'][:3].title()} {w['hour']:2d}h {w['level']:5.1f}%"
              f"        {hm(r['solar_noon'])}")
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
        print(f"trip length, {d}: worst departure {f['slope']:+.2f} min per minute of drive, r = {f['r']:+.2f} (n={f['n']})")
    for p, f in sun.items():
        print(f"solar noon vs city-wide {p} peak, US and Canada: slope {f['slope']:+.2f}, r = {f['r']:+.2f}")
    worst30 = max((r for r in routed if r["direction"] == "morning"), key=lambda r: r["worst_half_hour"])
    print(f"the costliest half hour: {worst30['metro']}, leaving home at {hm(worst30['worst_half_hour_at'] + 30)} instead of "
          f"{hm(worst30['worst_half_hour_at'])} a.m. gets you in {worst30['worst_half_hour']:.1f} min later")

    json.dump(dict(sources=["TomTom Traffic Index city pages, 2025 weekday-hour averages, metro area",
                            "TomTom Routing API, historic traffic, Tue-Thu 17-19 Nov 2026"],
                   cities=rows, routed=routed, trip_length=fits, solar_noon=sun),
              open(os.path.join(MEDIA, "rush.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    clock_chart(rows, os.path.join(MEDIA, "rush_clock.png"))
    trip_chart(routed, fits, os.path.join(MEDIA, "rush_trip_length.png"))
    friday_chart(rows, os.path.join(MEDIA, "rush_friday.png"))


if __name__ == "__main__":
    main()
