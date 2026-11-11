"""Chart for experiment.py: % mixed against paddle revolutions, one panel per question."""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
BASE = "#52514e"                                  # the film's twin gate paddles: same grey everywhere
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3

PANELS = [
    ("Paddle shape", [("twin_gate", "Gate (film)", BASE), ("twin_plate", "Solid plate", C1),
                      ("twin_fork", "Fork", C2)]),
    ("One paddle or two", [("twin_gate", "Two (film)", BASE), ("single_wide", "One wide", C1),
                           ("single_small", "One small", C2)]),
    ("How it is driven", [("twin_gate", "1 turn/s", BASE), ("twin_half_speed", "0.5 turn/s", C1),
                          ("twin_double_speed", "2 turns/s", C3),
                          ("twin_alternate", "Reversing", C2)]),
]


def main():
    rows = defaultdict(list)
    with open(os.path.join(HERE, "experiments", "mixing_vs_revolutions.csv")) as fh:
        for r in csv.DictReader(fh):
            rows[r["design"]].append((float(r["paddle_revolutions"]), float(r["percent_mixed"])))
    grains = None
    with open(os.path.join(HERE, "experiments", "mixing_vs_revolutions.csv")) as fh:
        grains = int(next(csv.DictReader(fh))["grains"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True, facecolor=SURFACE)
    xmax = max(x for d in rows.values() for x, _ in d)
    for ax, (title, series) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ends = []
        for name, label, col in series:
            if name not in rows:
                continue
            xs, ys = zip(*rows[name])
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.2,
                    zorder=3 if col != BASE else 2, solid_capstyle="round")
            ends.append([ys[-1], xs[-1], f"{label}  {ys[-1]:.0f}%", col])
        if title == "Paddle shape" and "twin_gate_seed" in rows:   # run-to-run noise
            xs, ys = zip(*rows["twin_gate_seed"])
            ax.plot(xs, ys, color=BASE, lw=1.2, ls=(0, (3, 3)), zorder=1)
            ax.annotate("dashed: same paddles,\ndifferent random fill", (xs[3], ys[3]), xytext=(18, -34),
                        textcoords="offset points", color=INK2, fontsize=9.5,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        # direct labels at the line ends, nudged apart so they never collide
        ends.sort(reverse=True)
        ends[0][0] = min(ends[0][0], 98.5)
        for i in range(1, len(ends)):
            ends[i][0] = min(ends[i][0], ends[i - 1][0] - 5.5)
        for y, x, text, col in ends:
            ax.plot([x + 0.4, x + 1.2], [y, y], color=col, lw=2, solid_capstyle="round", clip_on=False)
            ax.text(x + 1.6, y, text, color=INK, va="center", fontsize=10.5, clip_on=False)
        ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(0, xmax * 1.02)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.set_xlabel("Paddle revolutions in the powder", color=INK2)
    axes[0].set_ylabel("% mixed", color=INK2)
    fig.suptitle("How many paddle turns until the powders are mixed?", x=0.012, ha="left",
                 color=INK, fontsize=16, fontweight="bold")
    fig.text(0.012, 0.905, f"Simulated dish of six powders, about {round(grains, -3):,} grains. Each dot is one pass under the mixer. "
             "100% = indistinguishable from a random shuffle.", color=INK2, fontsize=10.5)
    fig.subplots_adjust(left=0.045, right=0.87, top=0.8, bottom=0.12, wspace=0.72)
    out = os.path.join(HERE, "experiments", "mixing_vs_revolutions.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
