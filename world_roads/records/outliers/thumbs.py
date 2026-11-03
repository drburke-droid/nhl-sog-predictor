"""Thumbnails A/B/C (1280x720), drawn on the film's 1080p canvas and downscaled.

    python thumbs.py       # writes ../../../videos/sports/thumbnails/outliers_thumb_{A,B,C}.png
"""
import os
import numpy as np
from PIL import Image
import film
from film import CARDS
from look import W, H, BG, INK, INK2, GOLD, BLUE, OUT, px, fs, new_fig, ax_px, text, title, plt

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, "..", "..", "..", "videos", "sports", "thumbnails")


def save(fig, name):
    os.makedirs(DST, exist_ok=True)
    tmp = os.path.join(OUT, f"_outliers_{name}.png")
    fig.savefig(tmp, dpi=100, facecolor=BG)
    plt.close(fig)
    Image.open(tmp).convert("RGB").resize((1280, 720), Image.LANCZOS).save(os.path.join(DST, f"outliers_thumb_{name}.png"))


def dots(ax, pts, x, year, trend=None):
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=px(30) ** 2, color=BLUE, edgecolors=BG, linewidths=px(4), zorder=4)
    if trend is not None:
        ax.plot(*trend, color="#9a9890", lw=px(6), zorder=3)
    ax.scatter([year], [x], s=px(62) ** 2, color=GOLD, edgecolors=BG, linewidths=px(6), zorder=6)
    ax.plot([year, year], [max(ys), x], color=GOLD, lw=px(4), ls=(0, (2, 2)), zorder=5)


def thumb_a():
    """Beamon: one gold dot far above every other season's best"""
    c = CARDS[1]
    pts = [(p["season"], p["value"]) for p in c["window"]]
    fig = new_fig(1)
    ax = ax_px(fig, 760, 140, 1840, 1000)
    ax.set_axis_off()
    ax.set_ylim(7.95, 9.0)
    ax.set_xlim(1952, 1984)
    dots(ax, pts, c["x_raw"], c["year"], trend=np.array(c["trend_raw_units"]).T)
    title(fig, 90, 420, "8.90 m", size=230, color=GOLD)
    title(fig, 96, 560, "The biggest", size=96, color=INK)
    title(fig, 96, 670, "outlier in", size=96, color=INK)
    title(fig, 96, 780, "sports history?", size=96, color=INK)
    text(fig, 96, 900, "Bob Beamon, 1968, against everyone else's best", size=40, color=INK2)
    save(fig, "A")


def thumb_b():
    """the four that never left the top ten"""
    fig = new_fig(1)
    items = [("8.90 m", "Beamon"), ("50.4", "Chamberlain"), ("56", "DiMaggio"), ("163", "Gretzky")]
    pos = [(500, 420), (1420, 420), (500, 860), (1420, 860)]
    for (big, who), (x, y) in zip(items, pos):
        title(fig, x, y, big, size=200, color=INK, ha="center")
        text(fig, x, y + 80, who, size=52, color=INK2, ha="center", weight="semibold")
    title(fig, W / 2, 140, "Which stood furthest above its era?", size=86, color=GOLD, ha="center")
    save(fig, "B")


def thumb_c():
    """Wilt: share of an average team's points, against everyone else's best"""
    c = CARDS[2]
    s = c["share"]
    pts = [(p[0], p[1]) for p in s["window"]]
    fig = new_fig(1)
    ax = ax_px(fig, 820, 150, 1840, 1000)
    ax.set_axis_off()
    lo = min(p[1] for p in pts)
    ax.set_ylim(lo - 0.03, s["x"] + 0.03)
    ax.set_xlim(1945, 1979)
    dots(ax, pts, s["x"], c["year"], trend=np.array(s["trend"]).T)
    title(fig, 90, 420, "50.4", size=260, color=GOLD)
    title(fig, 96, 560, "points a game", size=96, color=INK)
    title(fig, 96, 700, "Next best:", size=96, color=INK2)
    title(fig, 96, 810, f"{float(c['runner_up']['value']):.1f}", size=96, color=INK2)
    text(fig, 96, 930, "Wilt Chamberlain, 1961–62", size=40, color=INK2)
    save(fig, "C")


if __name__ == "__main__":
    thumb_a()
    thumb_b()
    thumb_c()
    print(os.path.abspath(DST))
