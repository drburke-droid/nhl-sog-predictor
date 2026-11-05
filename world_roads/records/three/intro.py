"""The opening of each companion film: why performances can't be compared across eras as they stand, and how the
score allows for it, in four or five short scenes built from intro.json (python facts_intro.py).

    improve   world records since 1950 keep rising (science, nutrition, training, technique)
    tiles     the same number in different eras (each film its own example)
    rate      step 1: every season converted to the scoring level of the season being judged (hockey's 1980s)
    method    step 2: compared with the best of everyone else in the 15 seasons either side; the score is how many
              standard deviations above that standard
    throws    (film B) the women's throws of the 1980s: when the outlier is an era
"""
import json, math, os

import numpy as np
import matplotlib

import film as FM
from look import (W, H, BG, INK, INK2, MUTED, GRID, AXIS, GOLD, BLUE, px, fs, clamp, ease, ease_out, fade, ax_px, text,  # noqa
                  title, style_axes, ring, outlined)
from matplotlib.ticker import MaxNLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
I = json.load(open(os.path.join(HERE, "intro.json"), encoding="utf-8"))
TREND = FM.TREND


def mmss(sec):
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:05.2f}" if m else f"{sec:.2f}"


def captions(fig, t, al, caps, x=1190, y=330, width=40, size=26, page=None):
    """captions stacked as they arrive; with page (a time), the ones before it fade out then and the rest start again
    at the top"""
    cur = max((c[0] for c in caps if c[0] <= t), default=None)
    y0 = y
    for group in ([c for c in caps if page is None or c[0] < page], [c for c in caps if page is not None and c[0] >= page]):
        y = y0
        for t0, s in group:
            a = al * fade(t, t0, page if page is not None and t0 < page else 99, 0.8, 0.6)
            if a > 0:
                y = FM.para(fig, x, y, s, width=width, size=size, color=INK if t0 == cur else INK2, alpha=a) + 20
    return y


def head(fig, al, kicker, name):
    text(fig, 96, 92, kicker, size=21, color=MUTED, weight="semibold", alpha=al)
    title(fig, 96, 160, name, size=58, alpha=al)


# ------------------------------------------------------------------ scenes
def improve(fig, t, al, spec):
    d = I["improve"]
    head(fig, al, "WHY THIS IS HARD", "Athletes keep getting better")
    ax = ax_px(fig, 150, 330, 1100, 930)
    style_axes(ax)
    yrs = np.array(d["years"])
    top = max(max(v) for v in d["lines"].values()) * 1.06
    ax.set_xlim(1948, 2028)
    ax.set_ylim(-1, top)
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"+{v:.0f}%" if v > 0 else "1950"))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(al)
    ax.spines["bottom"].set_alpha(al)
    ax.text(0, 1.04, f"World records in {d['n_events']} athletics and swimming events: how far above the 1950 record",
            transform=ax.transAxes, fontsize=fs(19), color=MUTED, alpha=al)
    sweep = 1950 + (2026 - 1950) * ease(clamp((t - 1.0) / 8.0))
    m = yrs <= sweep
    if m.sum() > 1:
        for v in d["lines"].values():
            ax.plot(yrs[m], np.array(v)[m], color="#8b919a", lw=px(1.2), alpha=0.28 * al, zorder=2)
        ax.fill_between(yrs[m], np.array(d["q25"])[m], np.array(d["q75"])[m], color=BLUE, alpha=0.22 * al, lw=0, zorder=3)
        ax.plot(yrs[m], np.array(d["median"])[m], color=GOLD, lw=px(3.5), alpha=al, zorder=5)
    if sweep >= 2025.9:
        ax.annotate(f"the typical event: +{d['median'][-1]:.0f}%", (2026, d["median"][-1]), xytext=(px(-10), px(14)),
                    textcoords="offset points", ha="right", fontsize=fs(22), color=GOLD, weight="semibold", alpha=al,
                    path_effects=outlined(6), zorder=7)
    captions(fig, t, al, spec["caps"])


def tiles(fig, t, al, spec):
    head(fig, al, "SAME NUMBER, DIFFERENT ERA", spec["title"])
    for i, tl in enumerate(spec["tiles"]):
        x = 380 + 580 * i
        a = al * fade(t, 0.8 + 1.4 * i, 99, 0.7, 0)
        title(fig, x, 470, tl["big"], size=118, color=INK, ha="center", alpha=a)
        text(fig, x, 535, tl["label"], size=27, color=INK2, ha="center", alpha=a)
        text(fig, x, 580, tl["context"], size=25, color=GOLD, ha="center", alpha=a, weight="semibold")
    y = 700
    cur = max((c[0] for c in spec["caps"] if c[0] <= t), default=None)
    for t0, s in spec["caps"]:
        a = al * fade(t, t0, 99, 0.8, 0)
        if a > 0:
            y = FM.para(fig, 160, y, s, width=92, size=30, color=INK if t0 == cur else INK2, alpha=a) + 24


def rate(fig, t, al, spec):
    head(fig, al, "STEP 1", "Put every season on the same scale")
    pts = np.array(I["nhl_rate"], float)
    ax = ax_px(fig, 150, 330, 1100, 930)
    style_axes(ax)
    ax.set_xlim(pts[0, 0] - 2, pts[-1, 0] + 2)
    ax.set_ylim(0, 5.6)
    ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(al)
    ax.spines["bottom"].set_alpha(al)
    ax.text(0, 1.04, "NHL goals per team per game, each season", transform=ax.transAxes, fontsize=fs(19), color=MUTED, alpha=al)
    at = spec["at"]
    band = al * fade(t, at["band"], 99, 0.8, 0)
    if band > 0:
        ax.axvspan(1979.5, 1989.5, color=GOLD, alpha=0.12 * band, lw=0, zorder=1)
        ax.text(1984.5, 5.45, "run-and-gun\n1980s", ha="center", va="top", fontsize=fs(19), color=INK2, alpha=band)
    sweep = pts[0, 0] + (pts[-1, 0] - pts[0, 0] + 1) * ease(clamp((t - 0.8) / 5.0))
    m = pts[:, 0] <= sweep
    if m.sum() > 1:
        ax.plot(pts[m, 0], pts[m, 1], color=BLUE, lw=px(2.8), alpha=al, zorder=4)
    nh = spec["nhl"]
    if t >= at["band"]:
        for y_, lab in ((1986, f"{nh['1986']:.2f} in 1985–86"), (2004, f"{nh['2004']:.2f} in 2003–04")):
            v = nh[str(y_)]
            ring(ax, y_, v, GOLD, size=13, alpha=al * fade(t, at["band"], 99, 0.5, 0), z=6)
            ax.annotate(lab, (y_, v), xytext=(px(10), px(14) if y_ == 1986 else px(-26)), textcoords="offset points",
                        fontsize=fs(21), color=INK, alpha=al * fade(t, at["band"], 99, 0.5, 0), path_effects=outlined(6),
                        zorder=7)
    captions(fig, t, al, spec["caps"])


def method(fig, t, al, spec):
    ex = spec["example"]
    head(fig, al, "STEP 2", "Compare with the best of their time")
    L = ex.get("length")
    conv = (lambda v: L / v) if L else (lambda v: v)                     # races: speed -> time, faster at the top
    ax = ax_px(fig, 150, 330, 1100, 930)
    style_axes(ax)
    pts = [(s, [conv(v) for v in vs]) for s, vs in ex["pts"]]
    allv = [v for _, vs in pts for v in vs] + [conv(ex["x"]), conv(ex["base"])]
    lo, hi = min(allv), max(allv)
    pad = 0.1 * (hi - lo)
    ys = [s for s, _ in pts] + [ex["year"]]
    ax.set_xlim(min(ys) - 1, max(ys) + 1)
    ax.set_ylim(*((hi + pad, lo - pad) if L else (lo - pad, hi + pad)))
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if L:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: mmss(v)[:-1]))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(al)
    ax.spines["bottom"].set_alpha(al)
    ax.text(0, 1.04, spec["chart_title"], transform=ax.transAxes, fontsize=fs(19), color=MUTED, alpha=al)
    first, last = min(ys), max(ys)
    at = spec["at"]
    sweep = first + (last - first + 1.5) * ease(clamp((t - 1.0) / max(2.0, at["curve"] - 1.5)))
    for s, vs in pts:
        if s <= sweep:
            pop = ease_out(clamp((sweep - s) / 1.5))
            for j, v in enumerate(sorted(vs)):
                dx = [0, -0.22, 0.22, -0.44, 0.44][min(j, 4)] if len(vs) > 1 else 0
                ring(ax, s + dx, v, BLUE, size=(9 if len(vs) > 1 else 12) * (0.5 + 0.5 * pop), alpha=al * pop, z=5)
    tr = np.array(ex["trend"])
    tc = clamp((t - at["curve"]) / 3.0)
    if tc > 0:
        k = max(2, int(len(tr) * tc))
        ax.plot(tr[:k, 0], [conv(v) for v in tr[:k, 1]], color=TREND, lw=px(3.2), alpha=al, zorder=4)
        if tc >= 1:
            ax.text(tr[-1, 0], conv(tr[-1, 1]), "  the curve", ha="left", va="center", fontsize=fs(18), color=INK2,
                    alpha=al, path_effects=outlined(5))
    ba = al * fade(t, at["band"], 99, 1.0, 0)
    if ba > 0:
        up = [conv(v + ex["step"]) for v in tr[:, 1]]
        dn = [conv(v - ex["step"]) for v in tr[:, 1]]
        ax.fill_between(tr[:, 0], dn, up, color=TREND, alpha=0.16 * ba, lw=0, zorder=2)
        if t < at["example"]:
            xm = tr[len(tr) // 4, 0]
            ax.annotate("", (xm, conv(np.interp(xm, tr[:, 0], tr[:, 1]) + ex["step"])),
                        (xm, conv(np.interp(xm, tr[:, 0], tr[:, 1]))),
                        arrowprops=dict(arrowstyle="<->", color=INK2, lw=px(2)), alpha=ba)
            ax.text(xm + 0.6, conv(np.interp(xm, tr[:, 0], tr[:, 1]) + ex["step"] * 0.5), "one standard deviation",
                    ha="left", va="center", fontsize=fs(19), color=INK2, alpha=ba, path_effects=outlined(5))
    la = al * fade(t, at["example"], 99, 0.6, 0)
    if la > 0:
        y0 = ex["year"]
        ring(ax, y0, conv(ex["raw"] if not L else ex["x"]), GOLD, size=17, alpha=la, z=8)
        ax.annotate(ex["label"], (y0, conv(ex["raw"] if not L else ex["x"])), xytext=(px(16), px(8)),
                    textcoords="offset points", fontsize=fs(22), color=INK, weight="semibold", alpha=la,
                    path_effects=outlined(6), zorder=9)
        prog = ease(clamp((t - at["example"] - 1.0) / 3.5))
        topv = ex["base"] + (ex["x"] - ex["base"]) * prog
        ax.plot([y0, y0], [conv(ex["base"]), conv(topv)], color=GOLD, lw=px(2.5), ls=(0, (2, 2)), alpha=la, zorder=7)
        for kk in range(1, int((topv - ex["base"]) / ex["step"] + 1e-9) + 1):
            yk = conv(ex["base"] + kk * ex["step"])
            ax.plot([y0 - 0.5, y0 + 0.5], [yk, yk], color=GOLD, lw=px(2.2), alpha=la, zorder=7)
            ax.text(y0 - 0.8, yk, f"{kk}", ha="right", va="center", fontsize=fs(17), color=GOLD, alpha=la, zorder=7,
                    path_effects=outlined(4))
        cnt = ex["z"] * prog
        title(fig, 150, 1040, f"{cnt:.1f}", size=64, color=GOLD, alpha=la)
        text(fig, 270, 1035, "standard deviations above the curve", size=24, color=INK2, alpha=la)
    captions(fig, t, al, spec["caps"], size=25, page=at.get("page"))


def throws(fig, t, al, spec):
    th = I["b"]["throws"]
    sp, dt = th["ath_W_SP"], th["ath_W_DT"]
    head(fig, al, "WHEN THE OUTLIER IS AN ERA", "The women's throws of the 1980s")
    ax = ax_px(fig, 150, 330, 1100, 930)
    style_axes(ax)
    leads = np.array(sp["leads"], float)
    ax.set_xlim(leads[0, 0] - 1, leads[-1, 0] + 1)
    ax.set_ylim(19.4, 23.2)
    ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(al)
    ax.spines["bottom"].set_alpha(al)
    ax.text(0, 1.04, "Women's shot put: the longest put of each season (metres)", transform=ax.transAxes, fontsize=fs(19),
            color=MUTED, alpha=al)
    at = spec["at"]
    band = al * fade(t, at["band"], 99, 1.0, 0)
    if band > 0:
        ax.axvspan(1979.5, 1989.5, color=GOLD, alpha=0.10 * band, lw=0, zorder=1)
        ax.text(1984.5, 23.1, "1980s", ha="center", va="top", fontsize=fs(20), color=INK2, alpha=band)
    sweep = leads[0, 0] + (leads[-1, 0] - leads[0, 0] + 1) * ease(clamp((t - 1.0) / 6.0))
    m = leads[:, 0] <= sweep
    if m.sum() > 1:
        ax.plot(leads[m, 0], leads[m, 1], color=BLUE, lw=px(2.2), alpha=al, zorder=4)
        ax.scatter(leads[m, 0], leads[m, 1], s=px(7) ** 2, color=BLUE, edgecolors=BG, linewidths=px(1.2), alpha=al, zorder=5)
    ru = sp["ruler"]
    ca = al * fade(t, at["curve"], 99, 0.8, 0)
    if ca > 0:
        tr = np.array(ru["trend"])
        ax.fill_between(tr[:, 0], tr[:, 1] - ru["step"], tr[:, 1] + ru["step"], color=TREND, alpha=0.16 * ca, lw=0, zorder=2)
        ax.plot(tr[:, 0], tr[:, 1], color=TREND, lw=px(3), alpha=ca, zorder=3)
        ax.text(tr[-1, 0], tr[-1, 1], "  the curve for 1987", ha="left", va="center", fontsize=fs(17), color=INK2,
                alpha=ca, path_effects=outlined(5))
        ring(ax, sp["year"], sp["value"], GOLD, size=16, alpha=ca, z=8)
        ax.annotate(f"{sp['name']}, {sp['value']:.2f} m (world record, {sp['year']})", (sp["year"], sp["value"]),
                    xytext=(px(14), px(6)), textcoords="offset points", fontsize=fs(21), color=INK, alpha=ca,
                    path_effects=outlined(6), zorder=9)
    captions(fig, t, al, spec["caps"], size=24)
    FM.para(fig, 150, 1000, "Random out-of-competition drug testing began in athletics in the late 1980s.", width=90, size=20,
            color=MUTED, alpha=al * fade(t, at["note"], 99, 0.8, 0))


SCENES = {"improve": improve, "tiles": tiles, "rate": rate, "method": method, "throws": throws}


class Intro:
    """the scenes in order, each fading in and out"""

    def __init__(self, scenes):
        self.scenes = scenes                                   # [(name, duration, spec)]
        self.dur = sum(d for _, d, _ in scenes)

    def draw(self, fig, t):
        t0 = 0.0
        for name, d, spec in self.scenes:
            if t0 <= t < t0 + d or (name == self.scenes[-1][0] and t >= t0):
                SCENES[name](fig, t - t0, fade(t - t0, 0, d, 0.6, 0.6), spec)
                return
            t0 += d

    def starts(self):
        out, t0 = {}, 0.0
        for name, d, _ in self.scenes:
            out[name] = t0
            t0 += d
        return out
