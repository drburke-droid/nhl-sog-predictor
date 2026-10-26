"""Charts of what model.py found:  python findings.py
-> ../media/commute/findings_half_hour.png, ../media/commute/findings_drive.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, CONTEXT = "#2a78d6", "#c9c8c3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE})


def hm(minute):
    h, m = divmod(int(round(minute)), 60)
    return f"{(h - 1) % 12 + 1}:{m:02d}"


def ticks(ax, lo, hi):
    ax.xaxis.set_major_locator(FixedLocator(np.arange(lo, hi + 1, 60)))
    ax.set_xticklabels([f"{hm(t)}" for t in np.arange(lo, hi + 1, 60)])
    ax.set_xlim(lo, hi)


def half_hour(C):
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    best = min(C, key=lambda c: c["best_half_hour"]["later_home"])
    for c in C:
        dep, A = np.array(c["depart"]), np.array(c["depart"]) + np.array(c["drive"])
        gap = A[30:] - A[:-30]
        hi = c is best
        ax.plot(dep[:-30], gap, color=BLUE if hi else CONTEXT, lw=2.2 if hi else 1.2, zorder=3 if hi else 2,
                solid_capstyle="round")
    ax.axhline(30, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(19 * 60 + 28, 30.5, "30 min: traffic gives nothing back", color=INK2, fontsize=10, va="bottom", ha="right")
    ax.axhspan(0, 1.5, color="#f1e3c4", zorder=0)
    ax.text(15 * 60 + 4, 2.2, "0 min: the \"magic\" departure, home no later for leaving later (never happens)",
            color=INK2, fontsize=10, va="bottom")
    b = best["best_half_hour"]
    ax.annotate(f"{best['metro']} is the closest: leave {hm(b['then'])} instead of {hm(b['depart'])},\n"
                f"home {b['later_home']:.1f} min later (saves {30 - b['later_home']:.1f} min of driving)",
                xy=(b["depart"], b["later_home"]), xytext=(b["depart"] + 25, 14), color=INK, fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.text(15 * 60 + 4, 34.0, "the other 19 metros", color=MUTED, fontsize=10)
    ticks(ax, 15 * 60, 19 * 60 + 30)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("if you would have left at ...  (p.m., local time)")
    ax.set_ylabel("minutes later home, for staying 30 min longer")
    fig.suptitle("Stay 30 minutes longer at work: how much later do you get home?", x=0.06, ha="left", fontsize=16,
                 fontweight="bold")
    fig.text(0.06, 0.875, "Office downtown to a random suburb 22-40 km out, typical Tuesday-Thursday, in 20 North "
             "American metros.\nModel: TomTom Traffic Index 2025 hourly congestion applied to an OSRM route on "
             "OpenStreetMap.", color=INK2, fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.1)
    fig.savefig(os.path.join(MEDIA, "findings_half_hour.png"), facecolor=SURFACE)


def grid(C):
    fig, axes = plt.subplots(4, 5, figsize=(16, 11), dpi=140, sharex=True, sharey=True)
    for ax, c in zip(axes.flat, sorted(C, key=lambda c: c["rank"])):
        dep, T = np.array(c["depart"]), np.array(c["drive"])
        ax.plot(dep, T, color=BLUE, lw=2, solid_capstyle="round")
        ax.axhline(c["free_flow_min"], color=MUTED, lw=0.9, ls=(0, (3, 3)))
        pk = c["peak"]
        ax.plot([pk["depart"]], [pk["drive"]], "o", ms=6, color=BLUE, mec=SURFACE, mew=2)
        ax.annotate(f"worst {hm(pk['depart'])}\n{pk['drive']:.0f} min", (pk["depart"], pk["drive"]),
                    xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_ylim(0, 92)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_title(f"{c['metro']}  >  {c['home']}", fontsize=9.5, loc="left", color=INK)
        ax.text(19 * 60 + 55, c["free_flow_min"] - 2.5, f"empty roads {c['free_flow_min']:.0f}", fontsize=7.5,
                color=MUTED, ha="right", va="top")
        ticks(ax, 15 * 60, 20 * 60)
        ax.grid(axis="y", color=GRID, lw=0.7)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, labelsize=8.5)
    fig.suptitle("The drive home by departure time, 3-8 p.m. (minutes)", x=0.04, ha="left", fontsize=16,
                 fontweight="bold")
    fig.text(0.04, 0.935, "Typical Tuesday-Thursday. TomTom Traffic Index 2025 hourly congestion (city centre "
             "near the office, metro area beyond) applied to the OSRM route on OpenStreetMap.", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.89, bottom=0.04, hspace=0.42, wspace=0.18)
    fig.savefig(os.path.join(MEDIA, "findings_drive.png"), facecolor=SURFACE)


if __name__ == "__main__":
    C = json.load(open(os.path.join(MEDIA, "curves.json")))["metros"]
    half_hour(C)
    grid(C)
