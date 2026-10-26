"""Charts of what analyse.py found:  python findings.py [tomtom|google] [--input FILE] [--out DIR]
-> ../media/commute/findings_<router>_<direction>_half_hour.png and ..._drive.png, for both commutes
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, BAND, CONTEXT, MAGIC = "#2a78d6", "#d6e4f5", "#c9c8c3", "#f1e3c4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE})
WORDS = {"evening": dict(leave="leave work", arrive="home", ampm="p.m.", trip="Office downtown to a random suburb"),
         "morning": dict(leave="leave home", arrive="at work", ampm="a.m.", trip="A random suburb to an office downtown")}


def hm(minute):
    h, m = divmod(int(round(minute)), 60)
    return f"{(h - 1) % 12 + 1}:{m:02d}"


def ticks(ax, lo, hi):
    ax.xaxis.set_major_locator(FixedLocator(np.arange(lo, hi + 1, 60)))
    ax.set_xticklabels([hm(t) for t in np.arange(lo, hi + 1, 60)])
    ax.set_xlim(lo, hi)


def plain(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def half_hour(C, direction, source, dst):
    w = WORDS[direction]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    best = min(C, key=lambda c: c["best_half_hour"]["later"])
    top = 0
    for c in C:
        dep, A = np.array(c["depart"]), np.array(c["depart"]) + np.array(c["drive"])
        n = int(round(30 / (dep[1] - dep[0])))
        gap = A[n:] - A[:-n]
        top = max(top, gap.max())
        hi = c is best
        ax.plot(dep[:-n], gap, color=BLUE if hi else CONTEXT, lw=2.2 if hi else 1.2, zorder=3 if hi else 2,
                solid_capstyle="round")
    lo_x, hi_x = C[0]["depart"][0], C[0]["depart"][-1] - 30
    ax.axhline(30, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(hi_x - 2, 30.5, "30 min: traffic gives nothing back", color=INK2, fontsize=10, va="bottom", ha="right")
    ax.axhspan(0, 1.5, color=MAGIC, zorder=0)
    ax.text(lo_x + 4, 2.2, f"0 min: the \"magic\" departure, {w['arrive']} no later for leaving later", color=INK2,
            fontsize=10, va="bottom")
    b = best["best_half_hour"]
    ax.annotate(f"{best['metro']} comes closest: {w['leave']} at {hm(b['then'])} instead of {hm(b['depart'])},\n"
                f"{w['arrive']} {b['later']:.1f} min later ({30 - b['later']:.1f} min less driving)",
                xy=(b["depart"], b["later"]), xytext=(lo_x + 20, max(6, b["later"] - 12)), color=INK, fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ticks(ax, lo_x, hi_x)
    ymax = max(40, 10 * np.ceil((top + 3) / 10))
    ax.set_ylim(min(0, b["later"] - 3), ymax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    plain(ax)
    ax.set_xlabel(f"if you would {w['leave']} at ...  ({w['ampm']}, local time)")
    ax.set_ylabel(f"minutes later {w['arrive']}, for leaving 30 min later")
    fig.suptitle(f"{w['leave'].capitalize()} 30 minutes later: how much later are you {w['arrive']}?", x=0.06, ha="left",
                 fontsize=16, fontweight="bold")
    fig.text(0.06, 0.875, f"{w['trip']}, typical Tuesday-Thursday, in {len(C)} North American metros (gray: the other {len(C) - 1})."
             f"\nPredicted drive times every 5 minutes: {source}.", color=INK2, fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.1)
    fig.savefig(dst, facecolor=SURFACE)
    plt.close(fig)


def grid(C, direction, source, dst):
    w = WORDS[direction]
    top = max(max(c["drive_max"]) for c in C)
    ymax = 30 * np.ceil((top + 18) / 30)
    rows = -(-len(C) // 5)
    fig, axes = plt.subplots(rows, 5, figsize=(16, 2.75 * rows), dpi=140, sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        if k >= len(C):
            ax.set_visible(False)
            axes.flat[k - 5].xaxis.set_tick_params(labelbottom=True)
    for ax, c in zip(axes.flat, sorted(C, key=lambda c: c["rank"])):
        dep, T = np.array(c["depart"]), np.array(c["drive"])
        ax.fill_between(dep, c["drive_min"], c["drive_max"], color=BAND, lw=0, zorder=1)
        ax.plot(dep, T, color=BLUE, lw=2, solid_capstyle="round", zorder=3)
        ax.axhline(c["free_flow_min"], color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=2)
        wk = c["worst"]
        ax.plot([wk["depart"]], [wk["drive"]], "o", ms=6, color=BLUE, mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"worst {hm(wk['depart'])}\n{wk['drive']:.0f} min", (wk["depart"], wk["drive"]),
                    xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_ylim(0, ymax)
        ax.set_yticks(np.arange(0, ymax + 1, 30))
        ax.set_title(f"{c['metro']}  >  {c['home']}" if direction == "evening" else f"{c['home']}  >  {c['metro']}",
                     fontsize=9.5, loc="left", color=INK)
        ax.text(dep[-1] - 5, c["free_flow_min"] - 2.5, f"empty roads {c['free_flow_min']:.0f}", fontsize=7.5,
                color=MUTED, ha="right", va="top")
        ticks(ax, dep[0], dep[-1])
        ax.grid(axis="y", color=GRID, lw=0.7)
        plain(ax)
        ax.tick_params(labelsize=8.5)
    fig.suptitle(f"The drive by departure time, {hm(C[0]['depart'][0])}-{hm(C[0]['depart'][-1])} {w['ampm']} (minutes)",
                 x=0.04, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.04, 0.935, f"{w['trip']}. Line: Tuesday-Thursday average; band: the range over the three days. "
             f"{source}.", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.89, bottom=0.04, hspace=0.42, wspace=0.18)
    fig.savefig(dst, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    router = next((a for a in args if a in ("tomtom", "google")), "tomtom")
    src = args[args.index("--input") + 1] if "--input" in args else os.path.join(MEDIA, f"curves_{router}.json")
    out = args[args.index("--out") + 1] if "--out" in args else MEDIA
    D = json.load(open(src))
    label = {"tomtom": "TomTom Routing API", "google": "Google Routes API"}[router]
    for direction in ("evening", "morning"):
        C = [c for c in D["metros"] if c["direction"] == direction]
        if C:
            half_hour(C, direction, label, os.path.join(out, f"findings_{router}_{direction}_half_hour.png"))
            grid(C, direction, label, os.path.join(out, f"findings_{router}_{direction}_drive.png"))
