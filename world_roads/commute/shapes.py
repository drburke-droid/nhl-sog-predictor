"""Which metros share a weekday traffic shape?  python shapes.py
-> ../media/commute/shapes.json, shapes_buckets.png, shapes_axes.png, shapes_map.png

A metro's shape: its Tue-Thu congestion by hour (TomTom Traffic Index, metro area, 2025; tomtom_weekhour.json),
4 a.m. to midnight, divided by its own peak, so a mild rush and a severe one with the same timing look alike. How bad
the traffic gets (the peak) is kept apart.

* Groups: Ward's method on the shapes, with the silhouette for 2 to 8 groups, and the same split redone on z-scored
  shapes and with average linkage, to see which groups hold up.
* Buckets: the number of Ward groups (3 to 6) whose split each day taken alone (Tue, Wed, Thu) repeats best; a metro
  is kept in its group only where at least two of the three days put it there too, the rest are "in between". Each
  group is named from its mean shape: the Mexican metros' group, the busiest middays, the quietest, the earlier
  evening peak of the two left, and the one left (the earliest mornings).
* Axes: the two main ways the US-shaped metros differ (principal components of their shapes), each set against a
  plain measure: midday traffic as a share of the peak (11 a.m.-3 p.m.) and early-morning (6-8 a.m.) minus
  late-evening (8-11 p.m.) traffic as a share of the peak.
* Peak timing against severity: the evening peak's time against peak congestion, US-shaped metros.
"""
import os, json
from itertools import permutations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
from findings import MEDIA, SURFACE, INK, INK2, MUTED, GRID, CONTEXT, hm, plain
from rush import profile, peak
import rush_map as rm

MIDWEEK = ["TUESDAY", "WEDNESDAY", "THURSDAY"]
H0 = 4                                                  # the shape runs 4 a.m. to midnight
# three categorical slots that validate all-pairs (light, dark surfaces), the sibling of "two sharp peaks" in a
# lighter step of its blue, the Mexican metros in ink / the light-speed film's warm white, in-between grey and hollow;
# all pairs clear the colour-blind and normal-vision floors, and every bucket has its own marker too
STYLE = {"sharp": dict(name="Two sharp peaks", light="#2a78d6", dark="#3987e5", marker="o"),
         "finish": dict(name="Early finishers", light="#86b6ef", dark="#9ec5f4", marker="v"),
         "allday": dict(name="Busy all day", light="#eb6834", dark="#d95926", marker="s"),
         "early": dict(name="Early starters", light="#1baf7a", dark="#199e70", marker="^"),
         "mexico": dict(name="Late and long", light=INK, dark="#f6ecd6", marker="D"),
         "between": dict(name="In between", light="#8a8984", dark="#8a93a3", marker="o")}
ORDER = ["sharp", "finish", "early", "allday", "mexico", "between"]
WORD = {3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def shape(P):
    return (P / P.max(1, keepdims=True))[:, H0:]


def silhouette(X, lab):
    D = squareform(pdist(X))
    s = []
    for i in range(len(X)):
        same = (lab == lab[i]) & (np.arange(len(X)) != i)
        if not same.any():
            s.append(0.0)
            continue
        a = D[i, same].mean()
        b = min(D[i, lab == c].mean() for c in set(lab) if c != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def ari(a, b):
    """adjusted Rand index of two labellings"""
    c2 = lambda n: n * (n - 1) / 2
    n = np.array([[np.sum((a == i) & (b == j)) for j in sorted(set(b))] for i in sorted(set(a))])
    s, sa, sb = c2(n).sum(), c2(n.sum(1)).sum(), c2(n.sum(0)).sum()
    e = sa * sb / c2(len(a))
    return float((s - e) / (0.5 * (sa + sb) - e))


def align(a, b):
    """b's labels renamed to match a's as well as they can"""
    best = max(permutations(sorted(set(b))), key=lambda p: sum(dict(zip(p, sorted(set(a))))[x] == y for x, y in zip(b, a)))
    m = dict(zip(best, sorted(set(a))))
    return np.array([m[x] for x in b])


def groups(X, k, method="ward"):
    return fcluster(linkage(X, method), k, "maxclust")


def features(p):
    q = p / p.max()
    return dict(am=peak(p, 5, 11)[0], pm=peak(p, 12, 21)[0], peak=float(p.max()), am_pm=float(p[5:12].max() / p[12:22].max()),
                midday=float(q[11:15].mean()), early_late=float(q[6:8].mean() - q[20:23].mean()))


def buckets_chart(rows, means, X, dst):
    fig, axes = plt.subplots(1, len(ORDER), figsize=(19, 6.6), dpi=150, sharey=True)
    hours = np.arange(H0, 24) + 0.5
    for ax, key in zip(axes, ORDER):
        st, R = STYLE[key], [r for r in rows if r["bucket"] == key]
        for r in R:
            ax.plot(hours, X[r["i"]], color=CONTEXT, lw=1, zorder=1)
        ax.plot(hours, means[key], color=st["light"], lw=2.6, zorder=3, solid_capstyle="round",
                ls=(0, (4, 2)) if key == "between" else "-")
        ax.set_title(f"{st['name']} ({len(R)})", loc="left", fontsize=12.5, color=INK, fontweight="bold")
        ax.text(0.0, -0.17, "\n".join(sorted(rm.SHORT.get(r["metro"], r["metro"]) for r in R)), transform=ax.transAxes,
                va="top", fontsize=9, color=INK2, linespacing=1.35)
        ax.set_xticks([6, 9, 12, 15, 18, 21, 24])
        ax.set_xticklabels(["6", "9", "noon", "3", "6", "9", "12"])
        ax.set_xlim(H0, 24)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=GRID, lw=0.7)
        plain(ax)
    axes[0].set_ylabel("congestion, share of the metro's own peak")
    fig.suptitle(f"{WORD[len(ORDER) - 1]} shapes of a weekday", x=0.04, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.905, f"{len(rows)} metros grouped by the shape of their Tuesday-Thursday traffic, each scaled to its own peak; "
             "thin lines: each metro, thick: the group's mean. TomTom Traffic Index, 2025.", color=INK2, fontsize=10.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.36, wspace=0.12)
    fig.savefig(dst, facecolor=SURFACE)
    plt.close(fig)


def axes_chart(rows, dst):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), dpi=150, sharex=True)
    for ax, (ykey, ylabel) in zip(axes, (("early_late", "early starters  →   6-8 a.m. minus 8-11 p.m. traffic, share of the peak"),
                                         ("pm", "evening peak, city-wide (p.m.)"))):
        for key in ORDER:
            st, R = STYLE[key], [r for r in rows if r["bucket"] == key]
            ax.scatter([r["midday"] for r in R], [r[ykey] for r in R], s=46, marker=st["marker"], zorder=3,
                       facecolor="none" if key == "between" else st["light"],
                       edgecolor=st["light"] if key == "between" else (INK2 if key == "finish" else SURFACE),
                       linewidth=1.2 if key == "between" else 0.6, label=f"{st['name']} ({len(R)})")
            for r in R:
                ax.annotate(rm.SHORT.get(r["metro"], r["metro"]), (r["midday"], r[ykey]), xytext=(5, 3),
                            textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_xlabel("busy all day  →   midday traffic (11 a.m.-3 p.m.) as a share of the peak")
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID, lw=0.7)
        plain(ax)
    ticks = np.arange(16.5 * 60, 17.5 * 60 + 1, 15)
    axes[1].set_yticks(ticks)
    axes[1].set_yticklabels([hm(t) for t in ticks])
    axes[1].set_ylim(16.5 * 60, 17.6 * 60)
    M = [r for r in rows if r["group"] == "mexico"]
    axes[1].text(0.99, 0.985, f"above: the Mexican metros, {hm(min(r['pm'] for r in M))}-{hm(max(r['pm'] for r in M))} p.m. ↑",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9.5, color=INK)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9.5)
    fig.suptitle("How weekday traffic shapes differ", x=0.05, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.05, 0.915, "Each metro's Tuesday-Thursday congestion, scaled to its own peak: how busy the middle of the day "
             "stays, against early mornings over late evenings (left) and the time of the evening peak (right). "
             "TomTom Traffic Index, 2025.", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.87, bottom=0.08, wspace=0.16)
    fig.savefig(dst, facecolor=SURFACE)
    plt.close(fig)


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def map_chart(rows, means, dst):
    L = rm.build()
    E = L["roads"][..., None] * rm.CYAN * 0.2
    for r in rows:
        x, y = rm.px(r["lon"], r["lat"])
        col = np.array(rgb(STYLE[r["bucket"]]["dark"]), np.float32) / 255
        rad = 40
        y0, y1, x0, x1 = int(y) - rad, int(y) + rad + 1, int(x) - rad, int(x) + rad + 1
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        g = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * 10.0 ** 2)) * (0.25 if r["bucket"] == "between" else 0.7)
        E[y0:y1, x0:x1] += g[..., None] * col
    img = 1 - np.exp(-(L["bg"] + E) * 1.15)
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    ov = Image.new("RGBA", (rm.OW, rm.OH), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    F = dict(title=rm.font("Inter-SemiBold.otf", 46), sub=rm.font("Inter-Regular.otf", 25), label=rm.font("Inter-Medium.otf", 15),
             leg=rm.font("Inter-Medium.otf", 20), small=rm.font("Inter-Medium.otf", 15), foot=rm.font("Inter-Regular.otf", 14),
             tick=rm.font("Inter-Regular.otf", 12))

    def marker(x, y, key, s=7):
        st = STYLE[key]
        c = rgb(st["dark"]) + (255,)
        o = (2, 5, 12, 255)
        if st["marker"] == "s":
            d.rectangle((x - s + 1, y - s + 1, x + s - 1, y + s - 1), fill=c, outline=o, width=2)
        elif st["marker"] == "^":
            d.polygon([(x, y - s - 1), (x + s + 1, y + s), (x - s - 1, y + s)], fill=c, outline=o, width=2)
        elif st["marker"] == "v":
            d.polygon([(x, y + s + 1), (x + s + 1, y - s), (x - s - 1, y - s)], fill=c, outline=o, width=2)
        elif st["marker"] == "D":
            d.polygon([(x, y - s - 1), (x + s + 1, y), (x, y + s + 1), (x - s - 1, y)], fill=c, outline=o, width=2)
        elif key == "between":
            d.ellipse((x - s, y - s, x + s, y + s), outline=c, width=3)
        else:
            d.ellipse((x - s, y - s, x + s, y + s), fill=c, outline=o, width=2)

    for r in rows:
        x, y = rm.px(r["lon"], r["lat"])
        marker(x, y, r["bucket"])
        dx, dy, anchor = rm.LABEL[r["metro"]]
        d.text((x + dx, y + dy), rm.SHORT.get(r["metro"], r["metro"]), font=F["label"], anchor=anchor,
               fill=(206, 218, 236, 235), stroke_width=3, stroke_fill=(2, 5, 12, 200))
    d.text((58, 40), f"{WORD[len(ORDER) - 1]} shapes of a weekday", font=F["title"], fill=(236, 242, 250, 255))
    d.text((60, 100), f"{len(rows)} metros grouped by the shape of their Tuesday-Thursday traffic, not by how bad it gets",
           font=F["sub"], fill=(150, 176, 205, 255))
    x0, y0, w, gap = 1545, 300, 300, 100
    d.text((x0, y0 - 34), "SHAPE OF THE DAY, 4 A.M.-MIDNIGHT", font=F["small"], fill=(130, 150, 175, 255))
    for j, key in enumerate(ORDER):
        st, n = STYLE[key], sum(r["bucket"] == key for r in rows)
        y = y0 + j * gap
        marker(x0 + 9, y + 12, key)
        d.text((x0 + 28, y + 12), f"{st['name']}", font=F["leg"], fill=(236, 242, 250, 255), anchor="lm")
        d.text((x0 + w, y + 12), f"{n}", font=F["leg"], fill=(160, 178, 200, 255), anchor="rm")
        sx = x0 + (np.arange(H0, 24) + 0.5 - H0) / (24 - H0) * w
        sy = y + 80 - means[key] * 46
        d.line([(x0, y + 80), (x0 + w, y + 80)], fill=(50, 66, 88, 255), width=1)
        for hh in (6, 12, 18):
            tx = x0 + (hh - H0) / (24 - H0) * w
            d.line([(tx, y + 80), (tx, y + 84)], fill=(80, 100, 125, 255), width=1)
        d.line(list(zip(sx, sy)), fill=rgb(st["dark"]) + (255,), width=3)
    for hh, lab in ((6, "6 a.m."), (12, "noon"), (18, "6 p.m.")):
        d.text((x0 + (hh - H0) / (24 - H0) * w, y0 + (len(ORDER) - 1) * gap + 87), lab, font=F["tick"], fill=(120, 140, 165, 255), anchor="ma")
    d.text((60, 1046), "Congestion: TomTom Traffic Index, metro areas, Tue-Thu averages of 2025, each metro scaled to its own peak "
           "· Map: Natural Earth", font=F["foot"], fill=(120, 140, 165, 255))
    im.paste(ov, (0, 0), ov)
    im.save(dst)


def main():
    W = json.load(open(os.path.join(MEDIA, "tomtom_weekhour.json")))["cities"]
    S = [W[m[5]]["areas"]["metro"] for m in rm.ALL]
    names = [m[1] for m in rm.ALL]
    P = np.array([profile(s, MIDWEEK) for s in S])
    X = shape(P)
    two = groups(X, 2)
    day_shapes = {d: shape(np.array([profile(s, [d]) for s in S])) for d in ["MONDAY"] + MIDWEEK + ["FRIDAY"]}
    stability = {}
    for k in range(3, 7):
        base = groups(X, k)
        stability[k] = float(np.mean([np.mean(align(base, groups(day_shapes[d], k)) == base) for d in MIDWEEK]))
    K = max(stability, key=stability.get)
    four = groups(X, K)
    Xz = (P[:, H0:] - P[:, H0:].mean(1, keepdims=True)) / P[:, H0:].std(1, keepdims=True)
    sil = {k: silhouette(X, groups(X, k)) for k in range(2, 9)}
    robust = {k: dict(zscore=ari(groups(X, k), groups(Xz, k)), average=ari(groups(X, k), groups(X, k, "average"))) for k in (2, 4, 5)}
    days = {d: align(four, groups(day_shapes[d], K)) for d in MIDWEEK}
    agree = sum((days[d] == four).astype(int) for d in MIDWEEK)
    two_days = {d: ari(two, groups(day_shapes[d], 2)) for d in day_shapes}

    assert K == 5, f"the most stable split has {K} groups, not the 5 the names below are for: {stability}"
    mx = four[names.index("Mexico City")]
    mid = {g: float(X[four == g][:, 11 - H0:15 - H0].mean()) for g in set(four)}
    pm = {g: float(np.mean([peak(P[i], 12, 21)[0] for i in np.nonzero(four == g)[0]])) for g in set(four)}
    rest = sorted((g for g in set(four) if g != mx), key=lambda g: mid[g])
    finish, early = sorted(rest[1:3], key=lambda g: pm[g])
    key_of = {mx: "mexico", rest[0]: "sharp", rest[3]: "allday", finish: "finish", early: "early"}
    rows = []
    for i, m in enumerate(rm.ALL):
        rows.append(dict(i=i, rank=m[0], metro=m[1], country=m[2], lat=m[4][1], lon=m[4][2], group=key_of[four[i]],
                         days_agree=int(agree[i]), bucket=key_of[four[i]] if agree[i] >= 2 else "between", **features(P[i])))
    el = {k: np.mean([r["early_late"] for r in rows if r["group"] == k]) for k in ("sharp", "finish", "early", "allday")}
    assert el["early"] == max(el.values()), f"the group left over is not the earliest: {el}"
    means = {k: X[[r["i"] for r in rows if r["bucket"] == k]].mean(0) for k in ORDER}

    us = np.array([r["i"] for r in rows if r["group"] != "mexico"])
    Xc = X[us] - X[us].mean(0)
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:2].T
    var = sv ** 2 / (sv ** 2).sum()
    R = [rows[i] for i in us]
    pc_match = dict(midday=float(np.corrcoef(pcs[:, 0], [r["midday"] for r in R])[0, 1]),
                    early_late=float(np.corrcoef(pcs[:, 1], [r["early_late"] for r in R])[0, 1]))
    lv, pm = np.array([r["peak"] for r in R]), np.array([r["pm"] for r in R])
    timing = dict(r=float(np.corrcoef(lv, pm)[0, 1]), minutes_per_10_points=float(np.polyfit(lv, pm, 1)[0] * 10), n=len(R))

    print("silhouette by number of groups: " + ", ".join(f"{k}: {v:.2f}" for k, v in sil.items()))
    print("each day alone repeats the Tue-Thu split, by number of groups: " + ", ".join(f"{k}: {v:.0%}" for k, v in stability.items()))
    for k, v in robust.items():
        print(f"{k} groups, same split with z-scored shapes: ARI {v['zscore']:.2f}; with average linkage: {v['average']:.2f}")
    print("two groups, each day alone against Tue-Thu: " + ", ".join(f"{d[:3].title()} {v:.2f}" for d, v in two_days.items()))
    print(f"{K} groups, each day alone: " + ", ".join(f"{d[:3].title()} {np.mean(days[d] == four):.0%} same" for d in MIDWEEK)
          + f"; {int(np.sum(agree >= 2))} of {len(rows)} metros kept")
    print(f"US-shaped metros: first two components {var[0]:.0%} and {var[1]:.0%} of the variation; midday share r = "
          f"{pc_match['midday']:+.2f} with the first, early-minus-late r = {pc_match['early_late']:+.2f} with the second")
    print(f"evening peak time against peak congestion: r = {timing['r']:+.2f}, {timing['minutes_per_10_points']:+.1f} min per 10 points")
    print(f"\n{'':22s} {'n':>2s}  am peak  pm peak  am/pm  midday  early-late  peak")
    for key in ORDER:
        B = [r for r in rows if r["bucket"] == key]
        print(f"{STYLE[key]['name']:22s} {len(B):2d}   {hm(np.mean([r['am'] for r in B])):>5s}    {hm(np.mean([r['pm'] for r in B])):>5s}   "
              f"{np.mean([r['am_pm'] for r in B]):.2f}    {np.mean([r['midday'] for r in B]):.2f}     {np.mean([r['early_late'] for r in B]):+.2f}   "
              f"{np.mean([r['peak'] for r in B]):3.0f}%  " + ", ".join(f"{r['metro']}" + ("" if key != "between" else f" ({STYLE[r['group']]['name'].lower()}, {r['days_agree']}/3)")
                                                          for r in B))
    json.dump(dict(source="TomTom Traffic Index, metro area, Tue-Thu averages of 2025, 4 a.m.-midnight, each metro scaled to its peak",
                   silhouette=sil, day_stability=stability, groups=K, robustness=robust, two_groups_by_day=two_days, components=[float(v) for v in var[:4]],
                   component_match=pc_match, timing=timing, buckets={k: STYLE[k]["name"] for k in ORDER},
                   mean_shapes={k: np.round(v, 3).tolist() for k, v in means.items()},
                   metros=[{k: v for k, v in r.items() if k != "i"} for r in rows]),
              open(os.path.join(MEDIA, "shapes.json"), "w"), ensure_ascii=False, indent=1)
    buckets_chart(rows, means, X, os.path.join(MEDIA, "shapes_buckets.png"))
    axes_chart(rows, os.path.join(MEDIA, "shapes_axes.png"))
    map_chart(rows, means, os.path.join(MEDIA, "shapes_map.png"))


if __name__ == "__main__":
    main()
