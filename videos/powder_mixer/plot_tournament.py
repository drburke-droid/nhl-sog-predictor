"""Charts for tournament.py: each tool's mixing curve, and how the result
moves with loading pattern, speed and grain size."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from tools import CONTESTANTS, TOOLS

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc", "#d5d2ca"
BLUE = "#2a78d6"


def load():
    rows = {r["run"]: r for r in csv.DictReader(open(os.path.join(HERE, "experiments", "tournament.csv")))}
    curves = np.load(os.path.join(HERE, "experiments", "tournament_curves.npz"))
    return rows, curves


def fig_base(rows, curves):
    names = [t for t in CONTESTANTS if f"base_{t}" in rows]
    names.sort(key=lambda t: (float(rows[f"base_{t}"]["s_to_90"] or 1e9), -float(rows[f"base_{t}"]["final_percent"])))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, t in zip(axes.ravel(), names):
        ax.set_facecolor(SURFACE)
        for o in names:
            if o != t:
                ax.plot(curves[f"base_{o}_t"], 100 * curves[f"base_{o}_mix"], color=CONTEXT, lw=1.1, zorder=1)
        x, y = curves[f"base_{t}_t"], 100 * curves[f"base_{t}_mix"]
        ax.plot(x, y, color=BLUE, lw=2.2, zorder=3, solid_capstyle="round")
        r = rows[f"base_{t}"]
        if r["s_to_90"]:
            note = f"90% in {float(r['s_to_90']):.0f} s, {float(r['count_to_90']):.0f} {r['unit']}"
            ax.plot([float(r["s_to_90"])], [90], "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5, zorder=4)
        else:
            note = f"{float(r['final_percent']):.0f}% after {float(r['seconds']):.0f} s"
        ax.set_title(TOOLS[t]["label"], loc="left", color=INK, fontsize=11.5, fontweight="bold", pad=18)
        ax.text(0, 1.02, note, transform=ax.transAxes, color=INK2, fontsize=10, va="bottom")
        ax.set_ylim(0, 100); ax.set_xlim(0, 20.6)
        ax.set_yticks(range(0, 101, 20)); ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    for ax in axes[1]:
        ax.set_xlabel("Seconds of mixing", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("% mixed", color=INK2)
    g = int(rows[f"base_{names[0]}"]["grains"])
    fig.suptitle("Eight ways to mix dry ingredients in a round bowl", x=0.012, ha="left", color=INK, fontsize=16, fontweight="bold")
    fig.text(0.012, 0.925, f"Simulated bowl of six powders loaded as pie wedges, about {round(g, -3):,} grains. "
             "Blue: this tool. Grey: the other seven. Best first.", color=INK2, fontsize=10.5)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.84, bottom=0.08, wspace=0.12, hspace=0.42)
    out = os.path.join(HERE, "experiments", "tournament_base.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)
    return names


def fig_sensitivity(rows, curves, order):
    cols = [("Pie wedges\n(the film)", "base_{}"), ("Stripes", "load_stripes_{}"), ("Rings", "load_rings_{}"),
            ("Half speed", "speed0.5_{}"), ("Double speed", "speed2_{}"),
            ("Fine grains\n1.4 mm", "grain1.4_{}"), ("Coarse grains\n2.4 mm", "grain2.4_{}")]
    groups = [(0, 3, "How the bowl was loaded"), (3, 5, "Speed (same number of strokes)"), (5, 7, "Grain size")]
    val = np.full((len(order), len(cols)), np.nan)
    for i, t in enumerate(order):
        for j, (_, pat) in enumerate(cols):
            r = rows.get(pat.format(t))
            if r:
                val[i, j] = float(r["final_percent"])
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc3ef", "#2a78d6", "#134a91"])
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(order) + 2.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i in range(len(order)):
        for j in range(len(cols)):
            gap = 0.14 * sum(j >= g[0] for g in groups[1:])
            if np.isnan(val[i, j]):
                ax.text(j + gap + 0.5, i + 0.5, "not run", ha="center", va="center", color="#a09d96", fontsize=9)
                continue
            v = val[i, j]
            ax.add_patch(plt.Rectangle((j + gap + 0.03, i + 0.05), 0.94, 0.9, color=cmap(v / 100), lw=0))
            ax.text(j + gap + 0.5, i + 0.5, f"{v:.0f}%", ha="center", va="center",
                    color="white" if v > 62 else INK, fontsize=11.5, fontweight="bold")
    for j, (lab, _) in enumerate(cols):
        gap = 0.14 * sum(j >= g[0] for g in groups[1:])
        ax.text(j + gap + 0.5, -0.12, lab, ha="center", va="bottom", color=INK2, fontsize=10)
    for a, b, lab in groups:
        gap = 0.14 * sum(a >= g[0] for g in groups[1:])
        ax.plot([a + gap + 0.03, b + gap - 0.03], [-0.95, -0.95], color=INK2, lw=1)
        ax.text((a + b) / 2 + gap, -1.05, lab, ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
    for i, t in enumerate(order):
        ax.text(-0.12, i + 0.5, TOOLS[t]["label"], ha="right", va="center", color=INK, fontsize=11)
    ax.set_xlim(-3.1, len(cols) + 0.4); ax.set_ylim(len(order) + 0.1, -1.9)
    ax.axis("off")
    fig.suptitle("Does the winner change?  % mixed at the end of each run", x=0.012, ha="left",
                 color=INK, fontsize=15, fontweight="bold")
    fig.text(0.012, 0.9, "Every run is 20 s at normal speed (half speed 40 s, double speed 10 s, so the stroke count "
             "is the same). 1.8 mm grains unless stated.", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02)
    out = os.path.join(HERE, "experiments", "tournament_sensitivity.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    rows, curves = load()
    order = fig_base(rows, curves)
    fig_sensitivity(rows, curves, order)
