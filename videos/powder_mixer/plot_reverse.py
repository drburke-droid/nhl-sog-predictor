"""Charts for reverse_experiment.py: does stirring backwards un-mix?"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools import TOOL_COLOR, TOOLS

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
SURFACE, INK, INK2, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc", "#bdbab2"
ORDER = ["spoon", "whisk", "fork", "paddle_large", "twin_beaters", "spoon_centre"]
NAME = {"spoon": "Spoon", "whisk": "Whisk", "fork": "Fork", "paddle_large": "Stand mixer, large paddle",
        "twin_beaters": "Hand mixer", "spoon_centre": "Spoon twirled in the middle"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)


def load():
    rows = list(csv.DictReader(open(os.path.join(EXP, "reverse.csv"))))
    for r in rows:
        for k in ("grain", "strokes", "tool_count", "mix_turn", "mix_peak", "mix_after_plunge", "mix_before_lift", "mix_end",
                  "moved_turn_mm", "moved_end_mm", "way_back", "home_fraction"):
            r[k] = float(r[k])
    return rows, json.load(open(os.path.join(EXP, "reverse_curves.json")))


def key(tool, n, load="wedges", grain=1.8):
    return f"{tool}_{load}_g{grain:g}_n{n:g}"


def fig_loops(rows, cur):
    """% mixed against how far along its path the tool is: out along the grey line, back along the coloured ones."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4), sharey=True, facecolor=SURFACE)
    for ax, tool in zip(axes.ravel(), ORDER):
        style(ax)
        ns = sorted(r["strokes"] for r in rows if r["set"] == "tools" and r["tool"] == tool)
        col = TOOL_COLOR[tool]
        longest = cur[key(tool, ns[-1])]
        fm = longest["frame_mid"]
        scale = ns[-1] / longest["count"][fm]                      # tool's own count -> hand strokes' worth of time
        ax.plot(np.array(longest["count"][:fm + 1]) * scale, 100 * np.array(longest["mix"][:fm + 1]),
                color=CONTEXT, lw=2.4, solid_capstyle="round", zorder=2)
        for n in ns:
            c = cur[key(tool, n)]
            f0 = c["frame_mid"]
            x, y = np.array(c["count"][f0:]) * (n / c["count"][f0]), 100 * np.array(c["mix"][f0:])
            ax.plot(x, y, color=col, lw=1.8, zorder=3, solid_capstyle="round")
            ax.plot([x[0]], [y[0]], "o", ms=6.5, color=SURFACE, mec=col, mew=1.8, zorder=4)       # where it turned back
            ax.plot([0], [y[-1]], "o", ms=6.5, color=col, mec=SURFACE, mew=1.2, zorder=5)          # back at the start
        ax.set_title(NAME[tool], loc="left", color=INK, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(-0.6, ns[-1] + 0.6)
        ax.set_xticks(ns)
        ax.set_ylim(-2, 100)
        ax.set_yticks(range(0, 101, 20))
    for ax in axes[1]:
        ax.set_xlabel("Strokes away from the start (one stroke = 0.67 s of stirring)", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("% mixed", color=INK2)
    fig.suptitle("Stirring backwards does not un-mix", x=0.012, ha="left", color=INK, fontsize=16, fontweight="bold")
    fig.text(0.012, 0.925, "Grey: stirring forward. Colour: the tool retraces its path exactly, from the open circle back to where it "
             "started (filled circle).\nIf mixing were reversible, each coloured line would run back down the grey one to 0%.",
             color=INK2, fontsize=10.5, va="top")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.08, wspace=0.08, hspace=0.32)
    out = os.path.join(EXP, "reverse_loops.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


def fig_home(rows):
    """How far the grains come back, by tool and by grain size."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.55, 1]))
    style(a); style(b)
    ends = []
    for tool in ORDER[:5]:                  # the centre twirl barely moves anything, so "the way back" means nothing for it
        rs = sorted((r for r in rows if r["set"] == "tools" and r["tool"] == tool), key=lambda r: r["strokes"])
        x, y = [r["strokes"] for r in rs], [100 * r["way_back"] for r in rs]
        a.plot(x, y, "-o", color=TOOL_COLOR[tool], lw=2, ms=6, mec=SURFACE, mew=1.2, solid_capstyle="round")
        ends.append([y[-1], x[-1], tool])
    ends.sort()
    for i in range(1, len(ends)):           # keep the direct labels from overlapping
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 5.5) if ends[i][1] == ends[i - 1][1] else ends[i][0]
    for y, x, tool in ends:
        a.text(x * 1.09, y, NAME[tool], color=TOOL_COLOR[tool], fontsize=10, va="center", fontweight="bold")
    a.axhline(0, color=INK2, lw=1)
    a.set_xscale("log", base=2)
    a.set_xticks([1, 2, 4, 8, 16]); a.set_xticklabels(["1", "2", "4", "8", "16"])
    a.set_xlim(0.85, 42); a.set_ylim(-20, 80)
    a.set_yticks(range(-20, 81, 20)); a.set_yticklabels([f"{v}%" for v in range(-20, 81, 20)])
    a.set_xlabel("Strokes forward before turning back", color=INK2)
    a.set_title("How far the grains come back", loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    a.text(0, 1.02, "Share of the distance travelled that the backward strokes undo (typical grain). 100% = back home.",
           transform=a.transAxes, color=INK2, fontsize=10, va="bottom")

    grains = [1.4, 1.8, 2.4]
    for tool in ("spoon", "whisk"):
        for n, ls in ((2, "-"), (8, (0, (4, 2)))):
            y = []
            for g in grains:
                r = [r for r in rows if r["tool"] == tool and r["load"] == "wedges" and r["grain"] == g and r["strokes"] == n
                     and r["seed"] == "7" and r["set"] in ("tools", "grain")]
                y.append(100 * r[0]["way_back"])
            b.plot(grains, y, ls=ls, marker="o", color=TOOL_COLOR[tool], lw=2, ms=6, mec=SURFACE, mew=1.2)
            b.text(2.44, y[-1], f"{NAME[tool]}, {n} strokes", color=TOOL_COLOR[tool], fontsize=9.5, va="center", fontweight="bold")
    b.axhline(0, color=INK2, lw=1)
    b.set_xticks(grains); b.set_xticklabels(["1.4 mm\nfine", "1.8 mm", "2.4 mm\ncoarse"])
    b.set_xlim(1.3, 3.05); b.set_ylim(-20, 80)
    b.set_yticks(range(-20, 81, 20)); b.set_yticklabels([])
    b.set_title("Grain size", loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    b.text(0, 1.02, "Same measure. Solid: turned back after 2 strokes. Dashed: after 8.", transform=b.transAxes,
           color=INK2, fontsize=10, va="bottom")
    fig.suptitle("Grains come part of the way back, and less so the longer a spoon or whisk has stirred", x=0.012, ha="left",
                 color=INK, fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.80, bottom=0.13, wspace=0.05)
    out = os.path.join(EXP, "reverse_home.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


def top_view(P, ingr, R, d, size=420):
    """Top-down picture of the bowl from grain positions (same drawing as the film's score strip)."""
    from PIL import Image, ImageDraw
    from film_compose import GRAIN
    S = size * 2
    px = S / (2 * R)
    img = np.full((S, S, 3), 252, np.uint8)
    zb = np.full((S, S), -1.0, np.float32)
    x = ((P[:, 0] + R) * px).astype(np.int32)
    y = ((R - P[:, 1]) * px).astype(np.int32)
    r = max(int(round(d * px / 2)), 1)
    col = GRAIN[ingr]
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            xx = np.clip(x + dx, 0, S - 1); yy = np.clip(y + dy, 0, S - 1)
            m = P[:, 2] > zb[yy, xx]
            order = np.argsort(P[m, 2])
            img[yy[m][order], xx[m][order]] = col[m][order]
            zb[yy[m][order], xx[m][order]] = P[m, 2][order]
    im = Image.fromarray(img).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse((2, 2, S - 3, S - 3), fill=255)
    out = Image.new("RGB", (size, size), (252, 252, 251))
    out.paste(im, (0, 0), mask.resize((size, size), Image.LANCZOS))
    return out


def fig_topviews(rows, tool="spoon"):
    ns = sorted(r["strokes"] for r in rows if r["set"] == "tools" and r["tool"] == tool)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, axes = plt.subplots(len(ns), 3, figsize=(8.6, 2.75 * len(ns) + 1.1), facecolor=SURFACE)
    R, d = 55.0, 1.8
    for i, n in enumerate(ns):
        z = np.load(os.path.join(HERE, "cache", "reverse", key(tool, n), "reverse.npz"))
        r = next(r for r in rows if r["set"] == "tools" and r["tool"] == tool and r["strokes"] == n)
        for j, (k, cap) in enumerate((("start", "0% mixed"), ("mid", f"{r['mix_turn']:.0f}% mixed"), ("end", f"{r['mix_end']:.0f}% mixed"))):
            ax = axes[i, j]
            ax.imshow(top_view(z[k], z["ingredient"], R, d))
            ax.set_axis_off()
            ax.text(0.5, -0.04, cap, transform=ax.transAxes, ha="center", va="top", color=INK2, fontsize=10)
        axes[i, 0].text(-0.06, 0.5, f"{n:g} stroke{'s' if n != 1 else ''}", transform=axes[i, 0].transAxes, ha="right", va="center",
                        color=INK, fontsize=11, fontweight="bold", rotation=90)
    for j, head in enumerate(("Start", "Stirred forward", "Then stirred back\nthe same number of strokes")):
        axes[0, j].set_title(head, color=INK, fontsize=11.5, fontweight="bold", pad=10)
    fig.suptitle(f"{NAME[tool]}: forward, then exactly backwards", x=0.03, ha="left", color=INK, fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - 1.25 / (2.75 * len(ns) + 1.1), bottom=0.03, wspace=0.04, hspace=0.2)
    out = os.path.join(EXP, f"reverse_topviews_{tool}.jpg")
    fig.savefig(out, dpi=130, facecolor=SURFACE, pil_kwargs=dict(quality=90))
    print("wrote", out)


def table(rows, cur):
    print("\nbackward leg, base case (pie wedges, 1.8 mm):")
    print(f"{'tool':28s}{'strokes':>8s}{'at turn':>9s}{'lowest':>8s}{'back at start':>15s}{'way home':>10s}")
    for tool in ORDER:
        for r in sorted((r for r in rows if r["set"] == "tools" and r["tool"] == tool), key=lambda r: r["strokes"]):
            c = cur[key(tool, r["strokes"])]
            m = 100 * np.array(c["mix"][c["frame_mid"]:])
            print(f"{NAME[tool]:28s}{r['strokes']:8g}{r['mix_turn']:8.0f}%{m.min():7.0f}%{r['mix_end']:14.0f}%{100 * r['way_back']:9.0f}%")


if __name__ == "__main__":
    rows, cur = load()
    fig_loops(rows, cur)
    fig_home(rows)
    fig_topviews(rows, "spoon")
    table(rows, cur)
