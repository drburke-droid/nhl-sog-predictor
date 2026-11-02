"""Thumbnails A/B/C (1280x720), drawn on the film's 1080p canvas and downscaled.

    python thumbs.py       # writes ../../videos/athletics/thumbnails/records_thumb_{A,B,C}.png
"""
import os
import numpy as np
from PIL import Image
from look import *                                              # noqa: F403
import film
import analysis as A

DST = os.path.join(HERE, "..", "..", "videos", "athletics", "thumbnails")


def canvas():
    fig = new_fig(1)
    ax = ax_px(fig, 0, 0, W, H)
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    return fig, ax


def save(fig, name):
    os.makedirs(DST, exist_ok=True)
    tmp = os.path.join(OUT, f"_{name}.png")
    fig.savefig(tmp, dpi=100, facecolor=BG)
    plt.close(fig)
    Image.open(tmp).convert("RGB").resize((1280, 720), Image.LANCZOS).save(os.path.join(DST, f"records_thumb_{name}.png"))


def thumb_a():
    race = film.Race()
    fig, ax = canvas()
    x0, x1 = 120, 1800
    ax.add_patch(matplotlib.patches.Rectangle((60, 600), 1800, 360, color=film.TRACK, zorder=1))
    for y in (600, 780, 960):
        ax.plot([60, 1860], [y, y], color=film.LANE, lw=px(3), alpha=0.6, zorder=2)
    ax.plot([x1, x1], [600, 960], color=INK, lw=px(6), zorder=2)
    xb = x0 + (x1 - x0) * race.bolt[1] / race.burke[1]
    ax.scatter([xb], [690], s=px(54) ** 2, color=BLUE, edgecolors=BG, linewidths=px(5), zorder=4)
    ax.scatter([x1], [870], s=px(54) ** 2, color=GOLD, edgecolors=BG, linewidths=px(5), zorder=4)
    ax.plot([xb - 260, xb - 30], [690, 690], color=BLUE, lw=px(22), alpha=0.25, solid_capstyle="round", zorder=3)
    ax.annotate("", (x1 - 8, 640), (xb + 40, 640), arrowprops=dict(arrowstyle="<->", color=INK, lw=px(6)))
    text(fig, 160, 700, "1896", size=58, weight="bold", color=INK)
    text(fig, 160, 880, "Bolt", size=58, weight="bold", color=INK)
    title(fig, 90, 250, f"{race.gap_m:.0f} metres", size=230, color=INK)
    text(fig, 96, 360, "The first Olympic champion, when the world record holder finishes", size=46, color=INK2)
    save(fig, "A")


def thumb_b():
    gm = film.GapChart("M")
    fig = new_fig(1)
    ax = ax_px(fig, 90, 380, 1830, 1000)
    ax.set_axis_off()
    ax.set_xlim(1910, 2026)
    ax.set_ylim(0.84, 1.005)
    ax.axhline(1.0, color=GOLD, lw=px(8))
    for d in gm.items:
        pts = gm.series(d)
        if len(pts) > 1:
            ax.plot(*zip(*pts), color=GREY, lw=px(2.5), alpha=0.45)
    ax.plot(*zip(*gm.med), color=INK, lw=px(11), solid_capstyle="round")
    text(fig, 1830, 520, "world record", size=40, color=INK2, ha="right")
    title(fig, 90, 260, "96%.", size=230, color=INK)
    title(fig, 760, 250, "For 112 years.", size=120, color=INK2)
    text(fig, 96, 345, "The middle of the Olympic final, against the world record", size=46, color=INK2)
    save(fig, "B")


def thumb_c():
    d = film.EventData("W", "DT")
    fig = new_fig(1)
    ax = ax_px(fig, 90, 360, 1830, 1010)
    ax.set_axis_off()
    ax.set_xlim(1950, 2026.5)
    ax.set_ylim(48, 80)
    m = d.rx >= 1900
    xs = np.append(d.rx, 2026.5)
    ys = np.append(d.ry, d.ry[-1])
    ax.step(xs, ys, where="post", color=GOLD, lw=px(10))
    for x in d.rows:
        if x["median"] and x["year"] >= 1952:
            ax.scatter([x["year"] + 0.6], [x["median"]], s=px(30) ** 2, color=BLUE, edgecolors=BG, linewidths=px(4),
                       zorder=5)
    ax.text(2026, 78.0, "76.80 m, 1988", ha="right", va="bottom", fontsize=fs(46), color=INK, fontweight="bold")
    ax.text(2026, 63.2, "Olympic field", ha="right", va="top", fontsize=fs(40), color=INK2)
    title(fig, 90, 230, "Untouched", size=200, color=INK)
    title(fig, 1170, 230, "since 1988", size=120, color=INK2)
    text(fig, 96, 320, "Women's discus: the Olympic field throws 85% of the record", size=46, color=INK2)
    save(fig, "C")


if __name__ == "__main__":
    thumb_a()
    thumb_b()
    thumb_c()
    print(DST)
